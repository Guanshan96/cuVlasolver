#pragma once

#include <string>

#include "tinyxml2.h"

#include "cufft.h"
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"

#define K 16
#define K2 16

#define PI acos(-1.0)

using namespace tinyxml2;

struct Boundary
{
	std::string bc1;
	std::string bc2;
	double vbc1;
	double vbc2;
};

struct Diagvar
{
	std::string name;
	int rate;

public:
	void set(std::string name, int rate)
	{
		this->name = name;
		this->rate = rate;
	}
};

struct Particle
{
	double* charge;
	double* cmratio;
	double* density;

	int nspecies;

	bool* isSave;

	Particle(int nspecies)
	{
		charge = new double[nspecies];
		cmratio = new double[nspecies];
		density = new double[nspecies];

		this->nspecies = nspecies;

		isSave = new bool[nspecies];
	}

public:
	void setcharge(double* charge)
	{
		for (size_t i = 0; i < nspecies; i++)
		{
			this->charge[i] = charge[i];
		}
	}

	void setcmratio(double* cmratio)
	{
		for (size_t i = 0; i < nspecies; i++)
		{
			this->cmratio[i] = cmratio[i];
		}
	}

	void setdensity(double* density)
	{
		for (size_t i = 0; i < nspecies; i++)
		{
			this->density[i] = density[i];
		}
	}

	void setsaveflag(bool* isSave)
	{
		for (size_t i = 0; i < nspecies; i++)
		{
			this->isSave[i] = isSave[i];
		}
	}

	void dispose()
	{
		delete[] charge;
		delete[] cmratio;
		delete[] density;
		delete[] isSave;
	}
};

struct Solver
{
	double xmax;

	double* vmin;
	double* vmax;

	double tstep;

	int ntsteps;
	int xngrids;
	int* vngrids;

	int nspecies;

	int nvars;

	std::string path;
	Diagvar* vars;

	Solver(int nspecies)
	{
		vmin = new double[nspecies];
		vmax = new double[nspecies];

		vngrids = new int[nspecies];

		vars = NULL;

		xmax = 0;
		xngrids = 0;

		ntsteps = 0;
		tstep = 0;

		this->nspecies = nspecies;

		nvars = 0;
	}

public:
	void settemporal(double tstep, int ntsteps)
	{
		this->tstep = tstep;
		this->ntsteps = ntsteps;
	}

	void setxgrid(double xmax, int xngrids)
	{
		this->xmax = xmax;
		this->xngrids = xngrids;
	}

	void setvmin(double* vmin)
	{
		for (size_t i = 0; i < nspecies; i++)
		{
			this->vmin[i] = vmin[i];
		}
	}

	void setvmax(double* vmax)
	{
		for (size_t i = 0; i < nspecies; i++)
		{
			this->vmax[i] = vmax[i];
		}
	}

	void setvngrids(int* vngrids)
	{
		for (size_t i = 0; i < nspecies; i++)
		{
			this->vngrids[i] = vngrids[i];
		}
	}

	void initdiagvars(int nvars)
	{
		vars = new Diagvar[nvars];
	}

	void setdiagvars(std::string name, int rate, int ind)
	{
		Diagvar var;
		var.name = name; var.rate = rate;

		vars[ind] = var;
	}

	void dispose()
	{
		delete[] vmin;
		delete[] vmax;
		delete[] vngrids;
		delete[] vars;
	}
};

//Structure which storge the distribution function and derivatives of one specie
//It also contains the charge density and velocity in config space
extern struct distr
{
	int dim[2];

	//f->Distribution function
	//fdv->Derivative of f in velocity space
	//fdx->Derivative of f in config space
	double* f;
	double* fdv;
	double* fdx;

	//S->Temporal variable for f
	//Sdv->Temporal variable for fdv
	//Sdx->Temporal variable for fdx
	double* S;
	double* Sdv;
	double* Sdx;

	//v->Velocity in config space
	double* v;

	//rho->Charge density of current specie
	double* rho;
	double* mid;

public:
	void initialize(size_t size, size_t vngrids, size_t xngrids);
	void writecharge(std::string filename);
	void writedistr(std::string filename, char type);
	void dispose();
	void vlinspace(double vmin, double vmax, double vngrids);
};

//Class which contains all simulation code of electrostatic 1-D model
extern class es1d_vsiumlator
{
private:
	int nspecies;
	distr* distr_fcns;
	size_t* distr_size;

	double dx;
	double* dv;

	int ntsteps;
	double tstep;

	double* devPtr_F;
	double* devPtr_Fs;

	double* devPtr_rk;
	double* devPtr_k;

	cufftDoubleComplex* devPtr_rho;
	cufftDoubleComplex* devPtr_phi;

	int xngrids;
	int* vngrids;

	double* vmax;
	double* vmin;
	double xmax;

	double* charge;
	double* cmratio;
	double* density;

	int nvars;
	Diagvar* vars;

	bool* isSave;

public:
	es1d_vsiumlator(Solver solver, Particle particle);
	void initialize();
	void start_simulation();

private:
	void dispose();
};

//Class which used to implement memory IO operation
class Memorystream
{
private:
	int offset;
	int bufferlength;

	char* buffer;

	std::string filename;

	const char* first;

public:
	Memorystream(std::string filename, int bufferlength);
	Memorystream();

public:
	void save(double* data, int length, char isCuda, char isLift);
	void save(int* shape, int ndim);
	void save();

public:
	void moveto(int position);
	void lift(int liftoffset);
	void setoffset(int offset);
	void dispose();
};

extern void trapz(double* f, double dx, int m, int n,
	int dim, double* mid, double* I);

extern void forthcentral(double* f, double d, int m,
	int n, int dim, char type, double* df);

extern void cuPoisson(cufftDoubleComplex * rho, cufftDoubleComplex * phi,
	double* k_recip, int n, cufftHandle plan);

extern void cuGradient(cufftDoubleComplex * phi, double* E,
	double* k, int n, cufftHandle plan);

extern void diagmatrix(double* val, int* pos, double* A, int nlines, int n, int m);

extern void pchip_2d_het(double* A, double* dA, double* F, double step,
	double dt, int n, int m, int dim, double* S, double* dS);

extern void pchip_2d_dev(double* dA, double* F,
	double* dA_i_p, double* dA_i, double step, double dt,
	int n, int m, int dim, double* dA_p);

extern void pointwiseop(double* vec1, double* vec2, int len, char op);

extern void pointwiseop(double* vec, int len, char op);

extern void real2cpx(double* real, cufftDoubleComplex* vec, int len);

extern void doublescl(double* vec, double alpha, int len);

extern void kspacesquare(double Lx, int n, double* k);

extern void kspace(double Lx, int n, double* k);

extern void generate_distr(double xmax, int xngrids, double vmin,
	double vmax, int vngrids, int specie, double* f);

/**/

void printmatrix(double* A, int n, int m);

Particle loadplasma(std::string filename);

Solver loadsolver(std::string filename, int nspecies);

void printplasma(Particle particle);

void printsolver(Solver solver);

void save(std::string filename, double* data, int* shape, int ndim);

void save(std::ofstream& stream, double* data, int length, char isCuda);

void save(std::ofstream& stream, int* shape, int ndim);

/**/