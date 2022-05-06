/**/

#include <cmath>

#include "vlasov1d.h"

/*!!!---User-defined initial distribution functions---!!!*/

/*
	DESCRIPTION:

	These functions donot belong to the simulation code.
	User can define their own distribution function but
	the signature must has following form:

	function_name(double , double* )

*/

__device__ void v_distr_1(double v, double *fv)
{
	fv[0] = exp(-v * v / 2) / sqrt(2 * PI);
 }

__device__ void x_distr_1(double x, double *fx)
{
	fx[0] = 1 + 0.01 * cos(0.5*x);
}

__device__ void v_distr_2(double v, double* fv)
{
	fv[0] = exp(-v * v / 2) / sqrt(2 * PI);
}

__device__ void x_distr_2(double x, double* fx)
{
	fx[0] = 1;
}

/*!!!---End of user-defined functions---!!!*/


/// <summary>
/// Kernel function for initialization of distribution function
/// </summary>
__global__ void generatedistr(double xmax, double xngrids, double vmin,
	double vmax, int vngrids, int specie, double *f)
{
	//velocity at "x" direction
	//space coordinate at "y" direction
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	double step_v = (vmax - vmin) / (vngrids - 1);
	double step_x = xmax / (xngrids - 1);
	
	double fx[1];
	double fv[1];

	if (x < vngrids && y < xngrids)
	{
		/*!!!---User-defined code---!!!*/

		/*
		*
		*	DESCRIPTION:
		*
		*	Users can add their own code based on number of
		*	particle species, but each code block of switch
		*	must give following results:
		*
		*	fx[0] (value of density disturbance)
		*	fv[0] (value of velocity distribution)
		*
		*/

		switch (specie)
		{
		case 0:
			v_distr_1(vmin + step_v * x, fv);
			x_distr_1(step_x * y, fx);
			break;
		case 1:
			v_distr_2(vmin + step_v * x, fv);
			x_distr_2(step_x * y, fx);
			break;
		default:
			break;
		}

		/*!!!---End of user-defined code---!!!*/

		f[x + y * vngrids] = fx[0] * fv[0];
	}

}

/// <summary>
/// Kernel function for initialization of derivatives through
/// analytical expressions
/// </summary>
__global__ void generatedistrderivatives(double xmax, double xngrids, double vmin,
	double vmax, int vngrids, int specie, double* df)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;


}

/// <summary>
/// User API which allow generation of distribution function
/// in 2-D phase space with GPU
/// </summary>
extern void generate_distr(double xmax, int xngrids, double vmin,
	double vmax, int vngrids, int specie, double* f)
{
	dim3 grid((vngrids + K - 1) / K, (xngrids + K2 - 1) / K2);
	dim3 block(K, K2);

	generatedistr << <grid, block >> > (xmax, xngrids, vmin, vmax, vngrids, specie, f);
}