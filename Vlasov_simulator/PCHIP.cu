#include <iostream>
#include <cmath>

#include "vlasov1d.h"

#define IDX2C(i, j, cols) ((i)*cols+j)

__device__ void pchip_coeff(double f1, double f2,
	double df1, double df2, double dx, double* coeff)
{
	coeff[0] = (df1 + df2) / (dx * dx) +
		2.0 * (f1 - f2) / (dx * dx * dx);
	coeff[1] = -(2.0 * df1 + df2) / dx -
		3.0 * (f1 - f2) / (dx * dx);
	coeff[2] = df1;
	coeff[3] = f1;
}

__global__ void transpose(double* A, int m, int n)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	double temp = A[x + y * m];

	A[x + y * m] = A[y + x * n];

	A[y + x * n] = temp;
}

__global__ void pchip_2d_het_x(double* A, double* dA, double* F, double dx,
	double dt, int m, double* S, double* dS)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	int lin_ind = IDX2C(y, x, m);

	double coeff[4];
	double vel = F[y];
	double zeta;

	//IDX2C: ":" in matlab equal to "y" or "x" in this kernel

	if (vel <= 0)
	{
		if (x == m - 1)
		{
			pchip_coeff(A[IDX2C(y, 0, m)], A[IDX2C(y, 1, m)],
				dA[IDX2C(y, 0, m)], dA[IDX2C(y, 1, m)], dx, coeff);
		}
		else
		{
			pchip_coeff(A[lin_ind], A[lin_ind + 1],
				dA[lin_ind], dA[lin_ind + 1], dx, coeff);
		}
		zeta = -vel * dt;
	}
	else
	{
		dx = -dx;
		if (x == 0)
		{
			pchip_coeff(A[IDX2C(y, m - 1, m)], A[IDX2C(y, m - 2, m)],
				dA[IDX2C(y, m - 1, m)], dA[IDX2C(y, m - 2, m)], dx, coeff);
		}
		else
		{
			pchip_coeff(A[lin_ind], A[lin_ind - 1],
				dA[lin_ind], dA[lin_ind - 1], dx, coeff);
		}
		zeta = -vel * dt;
	}

	S[lin_ind] = coeff[0] * (zeta * zeta * zeta) +
		coeff[1] * (zeta * zeta) + coeff[2] * zeta + coeff[3];
	dS[lin_ind] = 3.0 * coeff[0] * (zeta * zeta) +
		2.0 * coeff[1] * zeta + coeff[2];
}

__global__ void pchip_2d_het_y(double* A, double* dA, double* F, double dy,
	double dt, int n, int m, double* S, double* dS)
{
	//n--number of rows
	//m--number of cols
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	int lin_ind = IDX2C(y, x, m);

	double coeff[4];
	double vel = F[x];
	double zeta;

	if (vel <= 0)
	{
		if (y == n - 1)
		{
			pchip_coeff(A[IDX2C(0, x, m)], A[IDX2C(1, x, m)],
				dA[IDX2C(0, x, m)], dA[IDX2C(1, x, m)], dy, coeff);
		}
		else
		{
			pchip_coeff(A[lin_ind], A[lin_ind + m],
				dA[lin_ind], dA[lin_ind + m], dy, coeff);
		}
		zeta = -vel * dt;
	}
	else
	{
		dy = -dy;
		if (y == 0)
		{
			pchip_coeff(A[IDX2C(n - 1, x, m)], A[IDX2C(n - 2, x, m)],
				dA[IDX2C(n - 1, x, m)], dA[IDX2C(n - 2, x, m)], dy, coeff);
		}
		else
		{
			pchip_coeff(A[lin_ind], A[lin_ind - m],
				dA[lin_ind], dA[lin_ind - m], dy, coeff);
		}
		zeta = -vel * dt;
	}

	S[lin_ind] = coeff[0] * (zeta * zeta * zeta) +
		coeff[1] * (zeta * zeta) + coeff[2] * zeta + coeff[3];
	dS[lin_ind] = 3.0 * coeff[0] * (zeta * zeta) +
		2.0 * coeff[1] * zeta + coeff[2];
}

__global__ void pchip_2d_devy(double* dA_y, double* Fx,
	double* dA_x_p, double* dA_x, double dy, double dt,
	int n, int m, double* dA_y_p)
{
	//y--first index, x--second index
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	int lin_ind = IDX2C(y, x, m);

	double courant = dt / (4.0 * dy);
	double vel;
	double temp;

	if (y == n - 1)
	{
		vel = Fx[1];
		temp = -vel * (dA_x_p[IDX2C(1, x, m)] + dA_x[IDX2C(1, x, m)]);
	}
	else
	{
		vel = Fx[y + 1];
		//IDX2C(y+1, x, m)
		temp = -vel * (dA_x_p[lin_ind + m] + dA_x[lin_ind + m]);
	}

	if (y == 0)
	{
		vel = Fx[n - 2];
		temp += vel * (dA_x_p[IDX2C(n - 2, x, m)] + dA_x[IDX2C(n - 2, x, m)]);
	}
	else
	{
		vel = Fx[y - 1];
		//IDX2C(y-1, x, m)
		temp += vel * (dA_x_p[lin_ind - m] + dA_x[lin_ind - m]);
	}

	dA_y_p[lin_ind] = dA_y[lin_ind] + temp * courant;
}

__global__ void pchip_2d_devx(double* dA_x, double* Fy,
	double* dA_y_p, double* dA_y, double dx, double dt,
	int n, int m, double* dA_x_p)
{
	//y--first index, x--second index
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	int lin_ind = IDX2C(y, x, m);

	double courant = dt / (4.0 * dx);
	double vel;
	double temp;

	if (x == m - 1)
	{
		vel = Fy[1];
		temp = -vel * (dA_y[IDX2C(y, 1, m)] + dA_y_p[IDX2C(y, 1, m)]);
	}
	else
	{
		vel = Fy[x + 1];
		//IDX2C(y, x+1, m)
		temp = -vel * (dA_y[lin_ind + 1] + dA_y_p[lin_ind + 1]);
	}

	if (x == 0)
	{
		vel = Fy[m - 2];
		temp += vel * (dA_y[IDX2C(y, m - 2, m)] + dA_y_p[IDX2C(y, m - 2, m)]);
	}
	else
	{
		vel = Fy[x - 1];
		//IDX2C(y, x-1, m)
		temp += vel * (dA_y[lin_ind - 1] + dA_y_p[lin_ind - 1]);
	}

	dA_x_p[lin_ind] = dA_x[lin_ind] + temp * courant;
}

__global__ void pchip_2d_het_x_nonp(
	const double* A, 
	const double* dA, 
	const double* F, 
	double dx,
	double dt, 
	int m, 
	double* S, 
	double* dS)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	int lin_ind = IDX2C(y, x, m);

	double coeff[4];
	double vel = F[y];
	double zeta;

	//IDX2C: ":" in matlab equal to "y" or "x" in this kernel

	if (vel <= 0)
	{	
		if (x != m - 1)
		{
			pchip_coeff(A[lin_ind], A[lin_ind + 1],
				dA[lin_ind], dA[lin_ind + 1], dx, coeff);
		}
		zeta = -vel * dt;
	}
	else
	{
		dx = -dx;

		if (x != 0)
		{
			pchip_coeff(A[lin_ind], A[lin_ind - 1],
				dA[lin_ind], dA[lin_ind - 1], dx, coeff);
		}
		zeta = -vel * dt;
	}

	if (x != 0 && x != m - 1) {
		S[lin_ind] = coeff[0] * (zeta * zeta * zeta) +
			coeff[1] * (zeta * zeta) + coeff[2] * zeta + coeff[3];
		dS[lin_ind] = 3.0 * coeff[0] * (zeta * zeta) +
			2.0 * coeff[1] * zeta + coeff[2];
	}
	else
	{
		S[lin_ind] = A[lin_ind];
		dS[lin_ind] = dA[lin_ind];
	}
}

__global__ void pchip_2d_het_y_nonp(
	const double* A, 
	const double* dA, 
	const double* F, 
	double dy,
	double dt, 
	int n, int m, 
	double* S,
	double* dS)
{
	//n--number of rows
	//m--number of cols
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	int lin_ind = IDX2C(y, x, m);

	double coeff[4];
	double vel = F[x];
	double zeta;

	if (vel <= 0)
	{	
		if (y != n - 1)
		{
			pchip_coeff(A[lin_ind], A[lin_ind + m],
				dA[lin_ind], dA[lin_ind + m], dy, coeff);
		}
		zeta = -vel * dt;
	}
	else
	{
		dy = -dy;

		if (y != 0)
		{
			pchip_coeff(A[lin_ind], A[lin_ind - m],
				dA[lin_ind], dA[lin_ind - m], dy, coeff);
		}
		zeta = -vel * dt;
	}

	if (y != 0 && y != n - 1) {
		S[lin_ind] = coeff[0] * (zeta * zeta * zeta) +
			coeff[1] * (zeta * zeta) + coeff[2] * zeta + coeff[3];
		dS[lin_ind] = 3.0 * coeff[0] * (zeta * zeta) +
			2.0 * coeff[1] * zeta + coeff[2];
	}
	else
	{
		S[lin_ind] = A[lin_ind];
		dS[lin_ind] = dA[lin_ind];
	}
}

__global__ void pchip_2d_devy_nonp(
	const double* dA_y, 
	const double* Fx,
	const double* dA_x_p, 
	const double* dA_x, 
	double dy, 
	double dt,
	int n, int m, 
	double* dA_y_p)
{
	//y--first index, x--second index
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	int lin_ind = IDX2C(y, x, m);

	double temp;

	if (y > 0 && y < n - 1)
	{
		//IDX2C(y+1, x, m)
		temp = -Fx[y + 1] * (dA_x_p[lin_ind + m] + dA_x[lin_ind + m]);

		//IDX2C(y-1, x, m)
		temp += Fx[y - 1] * (dA_x_p[lin_ind - m] + dA_x[lin_ind - m]);

		dA_y_p[lin_ind] = dA_y[lin_ind] + temp * dt / (4.0 * dy);
	}
	else if(y == 0)
	{
		temp = Fx[0] * (3 * dA_y[IDX2C(0, x, m)] + 3 * dA_y_p[IDX2C(0, x, m)]);
		temp += -Fx[1] * (4 * dA_y[IDX2C(1, x, m)] + 4 * dA_y_p[IDX2C(1, x, m)]);
		temp += Fx[2] * (dA_y[IDX2C(2, x, m)] + dA_y_p[IDX2C(2, x, m)]);
	}
	else if (y == n - 1)
	{
		temp = -Fx[n - 1] * (3 * dA_y[IDX2C(n - 1, x, m)] + 3 * dA_y_p[IDX2C(n - 1, x, m)]);
		temp += Fx[n - 1] * (4 * dA_y[IDX2C(n - 1, x, m)] + 4 * dA_y_p[IDX2C(n - 1, x, m)]);
		temp += -Fx[n - 1] * (dA_y[IDX2C(n - 1, x, m)] + dA_y_p[IDX2C(n - 1, x, m)]);
	}

	dA_y_p[lin_ind] = dA_y[lin_ind] + temp * dt / (4.0 * dy);
}

__global__ void pchip_2d_devx_nonp(
	const double* dA_x, 
	const double* Fy,
	const double* dA_y_p, 
	const double* dA_y, 
	double dx, 
	double dt,
	int n, int m, 
	double* dA_x_p)
{
	//y--first index, x--second index
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	int lin_ind = IDX2C(y, x, m);

	double temp;

	if (x > 0 && x < m - 1)
	{
		//IDX2C(y, x+1, m)
		temp = -Fy[x + 1] * (dA_y[lin_ind + 1] + dA_y_p[lin_ind + 1]);

		//IDX2C(y, x-1, m)
		temp += Fy[x - 1] * (dA_y[lin_ind - 1] + dA_y_p[lin_ind - 1]);
	}
	else if (x == 0)
	{
		temp = Fy[0] * (3 * dA_y[IDX2C(y, 0, m)] + 3 * dA_y_p[IDX2C(y, 0, m)]);
		temp += -Fy[1] * (4 * dA_y[IDX2C(y, 1, m)] + 4 * dA_y_p[IDX2C(y, 1, m)]);
		temp += Fy[2] * (dA_y[IDX2C(y, 2, m)] + dA_y_p[IDX2C(y, 2, m)]);
	}
	else if (x == m - 1)
	{
		temp = -Fy[m - 1] * (3 * dA_y[IDX2C(y, m - 1, m)] + 3 * dA_y_p[IDX2C(y, m - 1, m)]);
		temp += Fy[m - 2] * (4 * dA_y[IDX2C(y, m - 2, m)] + 4 * dA_y_p[IDX2C(y, m - 2, m)]);
		temp += -Fy[m - 3] * (dA_y[IDX2C(y, m - 3, m)] + dA_y_p[IDX2C(y, m - 3, m)]);
	}

	dA_x_p[lin_ind] = dA_x[lin_ind] + temp * dt / (4.0 * dx);
}

/// <summary>
/// User API which implements the 1-D PCHI (Peicewise cubic Hermitian
/// interpolation) on a 2-D matrix (function).
/// </summary>
/// <param name="A">: 2-D function</param>
/// <param name="dA">: partial derivative on direction of interpolation</param>
/// <param name="F">: velocity (force) in phase space</param>
/// <param name="step">: grid size</param>
/// <param name="dt">: time step</param>
/// <param name="n">: number of rows</param>
/// <param name="m">: number of colums</param>
/// <param name="dim">: direction of interpolation</param>
/// <param name="S">: interpolated 2-D function</param>
/// <param name="dS">: interpolated partial derivatives</param>
extern void pchip_2d_het(double* A, double* dA, double* F, double step,
	double dt, int n, int m, int dim, double* S, double* dS)
{
	dim3 grid(m, n); dim3 block(1, 1);

	//n->size of first dimension
	//m->size of second dimension
	switch (dim)
	{
	case 1:
		//step->dy
		//F->Fy
		pchip_2d_het_y << <grid, block >> > (A, dA, F, step, dt, n, m, S, dS);
		break;
	case 2:
		//step->dx
		//F->Fx
		pchip_2d_het_x << <grid, block >> > (A, dA, F, step, dt, m, S, dS);
		break;
	default:
		std::cout << "Error: Dimension must larger than 0 and smaller than 3." << "\n";
		break;
	}
}

/// <summary>
/// User API which implements the 1-D PCHI (Peicewise cubic Hermitian
/// interpolation) on a 2-D matrix (function). (non-periodic)
/// </summary>
/// <param name="A">: 2-D function</param>
/// <param name="dA">: partial derivative on direction of interpolation</param>
/// <param name="F">: velocity (force) in phase space</param>
/// <param name="step">: grid size</param>
/// <param name="dt">: time step</param>
/// <param name="n">: number of rows</param>
/// <param name="m">: number of colums</param>
/// <param name="dim">: direction of interpolation</param>
/// <param name="S">: interpolated 2-D function</param>
/// <param name="dS">: interpolated partial derivatives</param>
extern void pchip_2d_het_nonp(
	const double* A, 
	const double* dA, 
	const double* F, 
	double step,
	double dt, 
	int n, int m, int dim,
	double* S,
	double* dS)
{
	dim3 grid(m, n); dim3 block(1, 1);

	//n->size of first dimension
	//m->size of second dimension
	switch (dim)
	{
	case 1:
		//step->dy
		//F->Fy
		pchip_2d_het_y_nonp << <grid, block >> > (A, dA, F, step, dt, n, m, S, dS);
		break;
	case 2:
		//step->dx
		//F->Fx
		pchip_2d_het_x_nonp << <grid, block >> > (A, dA, F, step, dt, m, S, dS);
		break;
	default:
		std::cout << "Error: Dimension must larger than 0 and smaller than 3." << "\n";
		break;
	}
}

/// <summary>
/// User API which solves the advection equation of derivatives
/// within one time step.
/// </summary>
/// <param name="dA">: partial derivative</param>
/// <param name="F">: velocity associates with dA_i (dA_i_p)</param>
/// <param name="dA_i_p">: partial derivative on another direction (+dt)</param>
/// <param name="dA_i">: partial derivative on another direction</param>
/// <param name="step">: grid size associates with dA</param>
/// <param name="dt">: time step</param>
/// <param name="n">: number of rows</param>
/// <param name="m">: number of colums</param>
/// <param name="dim">: direction of advection</param>
/// <param name="dA_p">: output partial derivative</param>
extern void pchip_2d_dev(double* dA, double* F,
	double* dA_i_p, double* dA_i, double step, double dt,
	int n, int m, int dim, double* dA_p)
{
	dim3 grid(m, n); dim3 block(1, 1);

	//n->size of first dimension
	//m->size of second dimension
	switch (dim)
	{
	case 1:
		//dA->dA_y
		//dA_i_p->dA_x_p, dA_i->dA_x
		//step->dy
		//dA_p->dA_y_p
		pchip_2d_devy << <grid, block >> > (dA, F, dA_i_p, dA_i, step, dt, n, m, dA_p);
		break;
	case 2:
		//dA->dA_x
		//dA_i_p->dA_y_p, dA_i->dA_y
		//step->dx
		//dA_p->dA_x_p
		pchip_2d_devx << <grid, block >> > (dA, F, dA_i_p, dA_i, step, dt, n, m, dA_p);
		break;
	default:
		std::cout << "Error: " << "\n";
		break;
	}
}

/// <summary>
/// User API which solves the advection equation of derivatives
/// within one time step. (non-periodic)
/// </summary>
/// <param name="dA">: partial derivative</param>
/// <param name="F">: velocity associates with dA_i (dA_i_p)</param>
/// <param name="dA_i_p">: partial derivative on another direction (+dt)</param>
/// <param name="dA_i">: partial derivative on another direction</param>
/// <param name="step">: grid size associates with dA</param>
/// <param name="dt">: time step</param>
/// <param name="n">: number of rows</param>
/// <param name="m">: number of colums</param>
/// <param name="dim">: direction of advection</param>
/// <param name="dA_p">: output partial derivative</param>
extern void pchip_2d_dev_nonp(
	const double* dA,
	const double* F,
	const double* dA_i_p,
	const double* dA_i, 
	double step, 
	double dt,
	int n, int m, int dim, 
	double* dA_p)
{
	dim3 grid(m, n); dim3 block(1, 1);

	//n->size of first dimension
	//m->size of second dimension
	switch (dim)
	{
	case 1:
		//dA->dA_y
		//dA_i_p->dA_x_p, dA_i->dA_x
		//step->dy
		//dA_p->dA_y_p
		pchip_2d_devy_nonp << <grid, block >> > (dA, F, dA_i_p, dA_i, step, dt, n, m, dA_p);
		break;
	case 2:
		//dA->dA_x
		//dA_i_p->dA_y_p, dA_i->dA_y
		//step->dx
		//dA_p->dA_x_p
		pchip_2d_devx_nonp << <grid, block >> > (dA, F, dA_i_p, dA_i, step, dt, n, m, dA_p);
		break;
	default:
		std::cout << "Error: Dimension must larger than 0 and smaller than 3." << "\n";
		break;
	}
}