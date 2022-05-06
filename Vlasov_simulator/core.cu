/*---User APIs of 1-D electrostatic Vlasov simulator---*/

#include <iostream>
#include <fstream>
#include <iomanip>

#include "vlasov1d.h"

//#define DEBUG_A
//#define DEBUG_B
#define DEBUG_C

#define DISPOSE true

void writebinary(void *data, int len)
{
	
}

void distr::initialize(size_t size, size_t vngrids, size_t xngrids)
{
	cudaMalloc((void**)&f, size);
	cudaMalloc((void**)&fdv, size);
	cudaMalloc((void**)&fdx, size);

	cudaMalloc((void**)&S, size);
	cudaMalloc((void**)&Sdv, size);
	cudaMalloc((void**)&Sdx, size);

	cudaMalloc((void**)&v, vngrids * sizeof(double));
	cudaMalloc((void**)&rho, xngrids * sizeof(double));
	cudaMalloc((void**)&mid, xngrids * ((vngrids + K - 1) / K) * sizeof(double));

	dim[0] = xngrids; dim[1] = vngrids;
}

void distr::writecharge(std::string filename)
{
	double* rho_host = new double[dim[0]];

	cudaMemcpy(rho_host, rho, sizeof(double) * dim[0], cudaMemcpyDeviceToHost);

	save(filename, rho_host, &dim[0], 1);

	delete[] rho_host;
}

void distr::writedistr(std::string filename, char type)
	{
		double* f_host = new double[dim[0] * dim[1]];

		switch (type)
		{
		case 'f':
			cudaMemcpy(f_host, f, sizeof(double) * dim[0] * dim[1],
				cudaMemcpyDeviceToHost);
			break;
		case 'v':
			cudaMemcpy(f_host, fdv, sizeof(double) * dim[0] * dim[1],
				cudaMemcpyDeviceToHost);
			break;
		case 'x':
			cudaMemcpy(f_host, fdx, sizeof(double) * dim[0] * dim[1],
				cudaMemcpyDeviceToHost);
			break;
		case 'S':
			cudaMemcpy(f_host, S, sizeof(double) * dim[0] * dim[1],
				cudaMemcpyDeviceToHost);
			break;
		case 'u':
			cudaMemcpy(f_host, Sdv, sizeof(double) * dim[0] * dim[1],
				cudaMemcpyDeviceToHost);
			break;
		case 'w':
			cudaMemcpy(f_host, Sdx, sizeof(double) * dim[0] * dim[1],
				cudaMemcpyDeviceToHost);
			break;
		default:
			break;
		}

		save(filename, f_host, dim, 2);
		delete[] f_host;
	}

void distr::dispose()
{
	cudaFree(f);
	cudaFree(fdv);
	cudaFree(fdx);

	cudaFree(S);
	cudaFree(Sdv);
	cudaFree(Sdx);

	cudaFree(v);
	cudaFree(rho);
	cudaFree(mid);
}

void distr::vlinspace(double vmin, double vmax, double vngrids)
{
	double* v = new double[vngrids];

	double step = (vmax - vmin) / (vngrids - 1);
	for (size_t i = 0; i < vngrids; i++)
	{
		v[i] = vmin + i * step;
	}

	cudaMemcpy(this->v, v, sizeof(double) * vngrids, cudaMemcpyHostToDevice);

	delete[] v;
}

es1d_vsiumlator::es1d_vsiumlator(Solver solver, Particle particle)
{
	nspecies = particle.nspecies;

	distr_fcns = new distr[nspecies];
	distr_size = new size_t[nspecies];

	dx = solver.xmax / (solver.xngrids - 1);
	dv = new double[nspecies];

	ntsteps = solver.ntsteps;
	tstep = solver.tstep;

	xngrids = solver.xngrids;
	vngrids = new int[nspecies];

	charge = new double[nspecies];
	cmratio = new double[nspecies];
	density = new double[nspecies];

	vmax = new double[nspecies];
	vmin = new double[nspecies];

	isSave = new bool[nspecies];

	for (size_t i = 0; i < nspecies; i++)
	{
		charge[i] = particle.charge[i];
		cmratio[i] = particle.cmratio[i];
		density[i] = particle.density[i];

		isSave[i] = particle.isSave[i];

		vmax[i] = solver.vmax[i];
		vmin[i] = solver.vmin[i];

		vngrids[i] = solver.vngrids[i];
	}
	
	vars = new Diagvar[solver.nvars];
	for (size_t i = 0; i < solver.nvars; i++)
	{
		vars[i] = solver.vars[i];
	}
	nvars = solver.nvars;

	xmax = solver.xmax;

	devPtr_F = NULL;
	devPtr_Fs = NULL;

	devPtr_rk = NULL;
	devPtr_k = NULL;

	devPtr_rho = NULL;
	devPtr_phi = NULL;
}

void es1d_vsiumlator::initialize()
{
	for (size_t i = 0; i < nspecies; i++)
	{
		distr_size[i] = xngrids * vngrids[i];//Number of points in distribution functions

		dv[i] = (vmax[i] - vmin[i]) / (vngrids[i] - 1);

		distr_fcns[i].initialize(distr_size[i] * sizeof(double), vngrids[i], xngrids);
		distr_fcns[i].vlinspace(vmin[i], vmax[i], vngrids[i]);
	}

	cudaMalloc((void**)&devPtr_F, xngrids * sizeof(double));
	cudaMalloc((void**)&devPtr_Fs, xngrids * sizeof(double));

	//Allocate memory to total charge density
	cudaMalloc((void**)&devPtr_rho, (xngrids - 1) * sizeof(cufftDoubleComplex));

	//Allocate memory to electrostatic potential
	cudaMalloc((void**)&devPtr_phi, (xngrids - 1) * sizeof(cufftDoubleComplex));

	//Allocate memory to k and k^-2
	cudaMalloc((void**)&devPtr_rk, (xngrids - 1) * sizeof(double));
	cudaMalloc((void**)&devPtr_k, (xngrids - 1) * sizeof(double));

	kspace(xmax, xngrids - 1, devPtr_k);
	kspacesquare(xmax, xngrids - 1, devPtr_rk);

	pointwiseop(devPtr_rk, xngrids - 1, 'r');
}

void es1d_vsiumlator::start_simulation()
{
	//Initialize cufft plan
	cufftHandle plan;
	cufftPlan1d(&plan, xngrids - 1, CUFFT_Z2Z, 1);

	/*End of initialization*/

	//Initialize distributions and its partial derivatives
	for (size_t i = 0; i < nspecies; i++)
	{
		generate_distr(xmax, xngrids,
			vmin[i], vmax[i], vngrids[i], i, distr_fcns[i].f);

		forthcentral(distr_fcns[i].f, dx, vngrids[i],
			xngrids, 1, 'p', distr_fcns[i].fdx);

		forthcentral(distr_fcns[i].f, dv[i], vngrids[i],
			xngrids, 2, 'n', distr_fcns[i].fdv);
	}

#ifdef DEBUG_A
	distr_fcns[1].writedistr("C:\\Users\\59669\\Desktop\\f.dat", 'f');
	distr_fcns[1].writedistr("C:\\Users\\59669\\Desktop\\v.dat", 'v');
	distr_fcns[1].writedistr("C:\\Users\\59669\\Desktop\\x.dat", 'x');

	trapz(distr_fcns[0].f, dv[0],
		vngrids[0], xngrids, 2, distr_fcns[0].mid, distr_fcns[0].rho);
	trapz(distr_fcns[1].f, dv[1],
		vngrids[1], xngrids, 2, distr_fcns[1].mid, distr_fcns[1].rho);
	doublescl(distr_fcns[0].rho, charge[0], xngrids);

	for (size_t i = 0; i < nspecies - 1; i++)
	{
		pointwiseop(distr_fcns[i].rho, distr_fcns[nspecies - 1].rho, xngrids, '+');
	}

	distr_fcns[1].writecharge("C:\\Users\\59669\\Desktop\\rho.dat");

	real2cpx(distr_fcns[nspecies - 1].rho, devPtr_rho, xngrids - 1);

	cuPoisson(devPtr_rho, devPtr_phi, devPtr_rk, xngrids - 1, plan);
	cuGradient(devPtr_phi, devPtr_F, devPtr_k, xngrids - 1, plan);

	double* E = new double[xngrids];

	cudaMemcpy(E, devPtr_F, sizeof(double) * (xngrids),
		cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < xngrids; i++)
	{
		std::cout << std::fixed << std::setprecision(10) << E[i] << "\n";
	}

#ifndef DEBUG_B
	dispose();

	return;
#endif
#endif

	//First push
	for (size_t i = 0; i < nspecies; i++)
	{
		pchip_2d_het(distr_fcns[i].f, distr_fcns[i].fdx, distr_fcns[i].v,
			dx, tstep / 2, xngrids, vngrids[i], 1, distr_fcns[i].S, distr_fcns[i].Sdx);

		pchip_2d_dev(distr_fcns[i].fdv, distr_fcns[i].v,
			distr_fcns[i].Sdx, distr_fcns[i].fdx, dv[i], tstep / 2,
			xngrids, vngrids[i], 2, distr_fcns[i].Sdv);

		trapz(distr_fcns[i].S, dv[i],
			vngrids[i], xngrids, 2, distr_fcns[i].mid, distr_fcns[i].rho);

		doublescl(distr_fcns[i].rho, charge[i], xngrids);
	}

	//Collect charge and solve the Poisson equation
	for (size_t i = 0; i < nspecies - 1; i++)
	{
		pointwiseop(distr_fcns[i].rho, distr_fcns[nspecies - 1].rho, xngrids, '+');
	}

	real2cpx(distr_fcns[nspecies - 1].rho, devPtr_rho, xngrids - 1);

	cuPoisson(devPtr_rho, devPtr_phi, devPtr_rk, xngrids - 1, plan);
	cuGradient(devPtr_phi, devPtr_F, devPtr_k, xngrids - 1, plan);

	cudaMemcpy(devPtr_Fs, devPtr_F, sizeof(double) * xngrids,
		cudaMemcpyDeviceToDevice);

#ifdef DEBUG_B
	distr_fcns[1].writedistr("C:\\Users\\59669\\Desktop\\f.dat", 'S');
	distr_fcns[1].writedistr("C:\\Users\\59669\\Desktop\\v.dat", 'u');
	distr_fcns[1].writedistr("C:\\Users\\59669\\Desktop\\x.dat", 'w');
	distr_fcns[1].writecharge("C:\\Users\\59669\\Desktop\\rho.dat");

	double* E = new double[xngrids];

	cudaMemcpy(E, devPtr_F, sizeof(double) * (xngrids),
		cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < xngrids; i++)
	{
		std::cout << std::fixed << std::setprecision(10) << E[i] << "\n";
	}

#ifndef DEBUG_C
	dispose();

	return;
#endif
#endif

	//Preparing diagnostic variables
	Memorystream mstream_efield;
	Memorystream mstream_density;
	Memorystream mstream_distr;

	bool is_efield = false;
	bool is_density = false;
	bool is_distr = false;

	int efield_rate;
	int density_rate;
	int distr_rate;

	int bytes_density;
	int byte_distr[10];
	int distr_index = 0;

	std::string savedir = "C:\\Programmer\\Vlasov-simulation_new\\Diagnostics\\";
	int savedir_length = savedir.length();

	for (size_t i = 0; i < nvars; i++)
	{
		if (0 == strcmp(vars[i].name.c_str(), "Electric field")) {

			mstream_efield = Memorystream(savedir.append("efield.dat"),
				(ntsteps / vars[i].rate) * xngrids * sizeof(double));

			savedir = savedir.substr(0, savedir_length);

			efield_rate = vars[i].rate;

			is_efield = true;
			continue;
		}
		
		if (0 == strcmp(vars[i].name.c_str(), "Density")) {

			mstream_density = Memorystream(savedir.append("density.dat"),
				(ntsteps / vars[i].rate) * xngrids * sizeof(double) * nspecies);

			savedir = savedir.substr(0, savedir_length);

			density_rate = vars[i].rate;

			bytes_density = (ntsteps / vars[i].rate) * xngrids * sizeof(double);

			is_density = true;
			continue;
		}

		if (0 == strcmp(vars[i].name.c_str(), "Distribution"))
		{
			int nsave = 0;
			int length = 0;
			int index = 0;
			for (size_t k = 0; k < nspecies; k++)
				if (isSave[k]) {
					nsave++;
					byte_distr[index] = distr_size[k] * (ntsteps / vars[i].rate) * sizeof(double);
					length += byte_distr[index++];
				}

			mstream_distr = Memorystream(savedir.append("distribution.dat"), length);

			savedir = savedir.substr(0, savedir_length);

			distr_rate = vars[i].rate;

			is_distr = true;
			continue;
		}
	}

	int index = 0;
	//Main loop
	for (size_t i = 1; i < ntsteps + 1; i++)
	{
		if (is_efield && i % efield_rate == 0) {
			mstream_efield.save(devPtr_F, xngrids, 'y', 'y');
		}

		//x is v, y is x
		//Push each specie and obtain their charge density
		for (size_t k = 0; k < nspecies; k++)
		{
			doublescl(devPtr_Fs, cmratio[k], xngrids);

			/*---Push the distribution function---*/
			//S => f
			//Sdv => fdv
			pchip_2d_het(distr_fcns[k].S, distr_fcns[k].Sdv, devPtr_Fs, 
				dv[k], tstep, xngrids, vngrids[k], 2, distr_fcns[k].f, distr_fcns[k].fdv);

			cudaDeviceSynchronize();

			//update x partial derivative
			//Sdx => fdx
			pchip_2d_dev(distr_fcns[k].Sdx, devPtr_Fs,
				distr_fcns[k].Sdv, distr_fcns[k].fdv, dx, tstep,
				xngrids, vngrids[k], 1, distr_fcns[k].fdx);

			cudaDeviceSynchronize();

			//f => S
			//fdx => Sdx
			pchip_2d_het(distr_fcns[k].f, distr_fcns[k].fdx, distr_fcns[k].v,
				dx, tstep, xngrids, vngrids[k], 1, distr_fcns[k].S, distr_fcns[k].Sdx);

			cudaDeviceSynchronize();

			//update v partial derivative
			//fdv => Sdv
			pchip_2d_dev(distr_fcns[k].fdv, distr_fcns[k].v,
				distr_fcns[k].Sdx, distr_fcns[k].fdx, dv[k], tstep,
				xngrids, vngrids[k], 2, distr_fcns[k].Sdv);

			cudaDeviceSynchronize();
			/*---End of push---*/

			trapz(distr_fcns[k].S, dv[k],
				vngrids[k], xngrids, 2, distr_fcns[k].mid, distr_fcns[k].rho);

			if (is_density && i % density_rate == 0) {
				mstream_density.setoffset(k * bytes_density);
				mstream_density.save(distr_fcns[k].rho, xngrids, 'y', 'n');
				if (k == nspecies - 1)
					mstream_density.lift(xngrids * sizeof(double));
			}

			doublescl(distr_fcns[k].rho, charge[k], xngrids);

			cudaMemcpy(devPtr_Fs, devPtr_F, sizeof(double) * xngrids,
				cudaMemcpyDeviceToDevice);

			if (is_distr && isSave[k] && i % distr_rate == 0)
			{
				int totaloffset = 0;
				for (size_t l = 0; l < index; l++)
					totaloffset += byte_distr[l];

				mstream_distr.setoffset(totaloffset);
				mstream_distr.moveto(distr_index * distr_size[k] * sizeof(double));
				mstream_distr.save(distr_fcns[k].S, distr_size[k], 'y', 'n');

				index++;

				if (k == nspecies - 1)
					distr_index++;
			}
		}

		index = 0;

		for (size_t k = 0; k < nspecies - 1; k++)
		{
			pointwiseop(distr_fcns[k].rho, distr_fcns[nspecies - 1].rho, xngrids, '+');
		}

		real2cpx(distr_fcns[nspecies - 1].rho, devPtr_rho, xngrids - 1);

		cuPoisson(devPtr_rho, devPtr_phi, devPtr_rk, xngrids - 1, plan);
		cuGradient(devPtr_phi, devPtr_F, devPtr_k, xngrids - 1, plan);

		cudaMemcpy(devPtr_Fs, devPtr_F, sizeof(double) * xngrids,
			cudaMemcpyDeviceToDevice);
	}

#ifdef DEBUG_C
	distr_fcns[0].writedistr("C:\\Users\\59669\\Desktop\\f.dat", 'S');
	distr_fcns[0].writedistr("C:\\Users\\59669\\Desktop\\v.dat", 'u');
	distr_fcns[0].writedistr("C:\\Users\\59669\\Desktop\\x.dat", 'w');
	distr_fcns[0].writecharge("C:\\Users\\59669\\Desktop\\rho.dat");

	distr_fcns[1].writedistr("C:\\Users\\59669\\Desktop\\fp.dat", 'S');
	distr_fcns[1].writedistr("C:\\Users\\59669\\Desktop\\vp.dat", 'u');
	distr_fcns[1].writedistr("C:\\Users\\59669\\Desktop\\xp.dat", 'w');
	distr_fcns[1].writecharge("C:\\Users\\59669\\Desktop\\rhop.dat");
#endif

	mstream_density.moveto(0);
	mstream_efield.moveto(0);
	mstream_distr.moveto(0);

	mstream_density.save();
	mstream_efield.save();
	mstream_distr.save();

	mstream_density.dispose();
	mstream_efield.dispose();
	mstream_distr.dispose();

	dispose();
}

void es1d_vsiumlator::dispose()
{
	cudaFree(devPtr_F);
	cudaFree(devPtr_Fs);
	cudaFree(devPtr_k);
	cudaFree(devPtr_phi);
	cudaFree(devPtr_rho);
	cudaFree(devPtr_rk);

	delete[] distr_fcns;
	delete[] distr_size;
	delete[] dv;
	delete[] vngrids;
	delete[] charge;
	delete[] cmratio;
	delete[] density;
	delete[] vars;

	for (size_t i = 0; i < nspecies; i++)
	{
		distr_fcns[i].dispose();
	}
}

