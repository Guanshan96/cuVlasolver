/*---User APIs and internal APIs of auxiliary functions---*/

/*
* 
*	DESCRIPTION:
* 
*	This cuda C++ code file contains several basic arithmetic 
*	functions (e.g. complex multiply and point wise arithmetic
*	operations), 1-D periodic poisson equation solver and
*	numerical caculus (i.e. numerical integral and differentiation)
*	functions.
*
*	Inner APIs can only be called in this .cu file, but the
*	user APIs can be called in other .cu or .cpp files.
*	Please refer to the summary of this APIs.
* 
*/

#include <Windows.h>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "vlasov1d.h"

#include "cusolverDn.h"

#pragma region Basic arithmetics
__global__ void complexscale(cufftDoubleComplex* vec, double* alpha)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	vec[x].x = alpha[x] * vec[x].x;
	vec[x].y = alpha[x] * vec[x].y;
}

__global__ void complexscale(cufftDoubleComplex* vec, double alpha)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	vec[x].x = alpha * vec[x].x;
	vec[x].y = alpha * vec[x].y;
}

__global__ void complexscale(cufftDoubleComplex* vec, cufftDoubleComplex z)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	cufftDoubleComplex ele;

	ele.x = vec[x].x * z.x - vec[x].y * z.y;

	ele.y = vec[x].x * z.y + vec[x].y * z.x;

	vec[x].x = ele.x; vec[x].y = ele.y;
}

__global__ void doublesacle(double* vec, double alpha)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	vec[x] = alpha * vec[x];
}

__global__ void doublescale_mat(double *vec, double alpha, int m)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	vec[x + y * m] = alpha * vec[x + y * m];
}

__global__ void pointwiseplus(double* vec, double *vec2)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	vec2[x] += vec[x];
}

__global__ void pointwisesquare(double *vec)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	vec[x] *= vec[x];
}

__global__ void pointwisereciprocal(double *vec)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	vec[x] = 1.0 / vec[x];
}

__global__ void takerealpart(cufftDoubleComplex *vec, double *real)
{

	int x = threadIdx.x + blockIdx.x * blockDim.x;

	real[x] = vec[x].x;

}

__global__ void takerealpart_(cufftDoubleComplex* vec, double* real, int len)
{
	//len is the length of vec
	//length(vec) equal with length(real) - 1
	//max(x) is equal with length(real)
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	if (x < len)
	{
		real[x] = vec[x].x;
	}

	if (x == len)
	{
		real[x] = vec[0].x;
	}
}

__global__ void takeimagpart(cufftDoubleComplex *vec, double *imag)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	imag[x] = vec[x].y;
}

__global__ void real2complex(double* real, cufftDoubleComplex* vec)
{
	//lenth(real) could be larger than length(vec)
	//max(x) is length of vec
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	vec[x].x = real[x];
	vec[x].y = 0;
}




/// <summary>
/// 
/// </summary>
/// <param name="vec"></param>
/// <param name="alpha"></param>
/// <param name="len"></param>
extern void doublescl(double *vec, double alpha, int len)
{
	doublesacle << <len, 1 >> > (vec, alpha);
}

/// <summary>
/// 
/// </summary>
/// <param name="real"></param>
/// <param name="vec"></param>
/// <param name="len"></param>
extern void real2cpx(double *real, cufftDoubleComplex* vec, int len)
{
	real2complex << <len, 1 >> > (real, vec);
}

/// <summary>
/// User API which implements 1-D (vector) point-wise operation
/// between two vectors. The result is stored in the second vector.
/// </summary>
/// <param name="vec1">: fisrt vector</param>
/// <param name="vec2">: second vector</param>
/// <param name="len">: length of vectors</param>
/// <param name="op">: type of operation</param>
extern void pointwiseop(double* vec1, double* vec2, int len, char op)
{
	switch (op)
	{
	case '+':
		pointwiseplus << <len, 1 >> > (vec1, vec2);
		break;
	case '-':
		break;
	default:
		break;
	}
}

/// <summary>
/// User API which implements 1-D (vector) point-wise operation on
/// a single vector. The result is stored in this vector.
/// </summary>
/// <param name="vec">: vector</param>
/// <param name="len">: length of vector</param>
/// <param name="op">: type of operation</param>
extern void pointwiseop(double* vec, int len, char op)
{
	switch (op)
	{
	case '2':
		pointwisesquare << <len, 1 >> > (vec);
		break;
	case 'r':
		pointwisereciprocal << <len, 1 >> > (vec);
		break;
	default:
		break;
	}
}

#pragma endregion

/*---1-D FFT periodic Poisson solver---*/

/*
* 
*	DESCRIPTION:
* 
*	This program use FFT to solve 1-D periodic Poisson 
*	equation. The length of rho (charge density) vector
*	must equal with xngrids-1.
* 
*	The form of Poisson equation here is
*	d^2(phi)/dx^2 = -rho
* 
*/

#pragma region Periodic Poisson equation 1D
/// <summary>
/// User API used to generate wavenumber vector k, which
/// used in FFT gradient. Note the *k is a pointer on
/// V-RAM
/// </summary>
/// <param name="Lx">: physical length of system</param>
/// <param name="n">: length of rho vector</param>
/// <param name="k">: wavenumber vector</param>
extern void kspace(double Lx, int n, double* k)
{
	//n equal with length of length(rho)
	double* k_0 = new double[n];

	double unit = 2 * PI / Lx;

	for (size_t i = 0; i < n / 2; i++)
	{
		k_0[i] = (double)i * unit;
		k_0[i + n / 2] = (-(double)n / 2 + i) * unit;
	}

	if (n % 2 != 0)
	{
		k_0[n - 1] = 0;
	}

	cudaMemcpy(k, k_0, sizeof(double) * n, cudaMemcpyHostToDevice);

	delete[] k_0;
}

/// <summary>
/// User API used to generate inverse square of wavenumber
/// vector k, which used in FFT Poisson solver. Note the *k
/// is a pointer on V-RAM
/// </summary>
/// <param name="Lx">: physical length of system</param>
/// <param name="n">: length of rho vector</param>
/// <param name="k">: inverse square of wavenumber vector</param>
extern void kspacesquare(double Lx, int n, double* k)
{
	//n equal with length of length(rho)
	double* k_0 = new double[n];
	
	double unit = 2 * PI / Lx;

	for (size_t i = 0; i < n / 2; i++)
	{
		k_0[i] = (double)i * unit;
		k_0[i] *= k_0[i];
		k_0[i + n / 2] = (-(double)n / 2 + i) * unit;
		k_0[i + n / 2] *= k_0[i + n / 2];
	}

	if (n % 2 != 0)
	{
		k_0[n - 1] = 1;
	}

	k_0[0] = 1;

	cudaMemcpy(k, k_0, sizeof(double) * n, cudaMemcpyHostToDevice);

	delete[] k_0;
}

/// <summary>
/// User API which uses fast Fourier transform to solve the
/// 1-D periodic Poisson equation
/// </summary>
/// <param name="rho">: charge density</param>
/// <param name="phi">: electrostatic potential</param>
/// <param name="k_recip">: inverse square of k (wavenumber vector)</param>
/// <param name="n">: length of rho vector</param>
/// <param name="plan">: cufft plan</param>
extern void cuPoisson(cufftDoubleComplex* rho, cufftDoubleComplex* phi,
	double* k_recip, int n, cufftHandle plan)
{
	cufftExecZ2Z(plan, rho, phi, CUFFT_FORWARD);

	complexscale << <n, 1 >> > (phi, k_recip);

	cufftExecZ2Z(plan, phi, phi, CUFFT_INVERSE);

	complexscale << <n, 1 >> > (phi, 1.0 / n);
}

/// <summary>
/// User API which uses fast Fourier transform to calculate
/// the 1-D periodic gradient
/// </summary>
/// <param name="phi">: electrostatic potential</param>
/// <param name="E">: electric field</param>
/// <param name="k">: wavenumber vector</param>
/// <param name="n">: length of potential</param>
/// <param name="plan">: cufft plan</param>
extern void cuGradient(cufftDoubleComplex * phi, double* E,
	double* k, int n, cufftHandle plan)
{
	cufftExecZ2Z(plan, phi, phi, CUFFT_FORWARD);

	complexscale << <n, 1 >> > (phi, k);

	cufftDoubleComplex unit;

	unit.x = 0; unit.y = -1.0 / n;

	complexscale << <n, 1 >> > (phi, unit);

	cufftExecZ2Z(plan, phi, phi, CUFFT_INVERSE);

	takerealpart_ << <n + 1, 1 >> > (phi, E, n);
}

#pragma endregion


/*---1-D non-periodic Poisson solver---*/

/*
*
*	DESCRIPTION:
*
*	This program use FEM to solve 1-D non-periodic Poisson
*	equation. The length of rho (charge density) vector
*	must equal with xngrids.
*
*	The form of Poisson equation here is
*	d^2(phi)/dx^2 = -rho
*
*/

#pragma region Non-periodic Poisson equation 1D

/// <summary>
/// Generate diagonal matrix on v-RAM in column-major style (cublas)
/// </summary>
/// <param name="val">: Value on each diagonal line</param>
/// <param name="pos">: Position of diagonal line relative to the central diagonal line</param>
/// <param name="res">: Pointer to the diagonal matrix on v-RAM</param>
/// <param name="nlines">: Number of non-zero diagonal lines</param>
/// <param name="n">: Number of rows</param>
/// <param name="m">: Number of columns</param>
extern void diagmatrix(double *val, int *pos, double *res, int nlines, int n, int m)
{
	double* A = new double[n * m];

	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			A[j + i * n] = 0;
		}
	}

	for (size_t l = 0; l < nlines; l++)
	{
		if (pos[l] == 0)
		{
			for (size_t i = 0; i < n && i < m; i++)
			{
				A[i + i * n] = val[l];
			}
		}
		else if (pos[l] < 0)
		{
			for (size_t i = -pos[l]; i < (m - pos[l]) && i < n; i++)
			{
				A[i + (i + pos[l]) * n] = val[l];
			}
		}
		else
		{
			for (size_t i = 0; (i + pos[l]) < m && i < n; i++)
			{
				A[i + (i + pos[l]) * n] = val[l];
			}
		}
	}

	cublasSetMatrix(n, m, sizeof(double), A, n, res, n);

	delete[] A;
}

/// <summary>
/// User API which is used to generate matrices of 1-D finite-element method
/// </summary>
/// <param name="handle">: cuSolver context</param>
/// <param name="bc">: Boundary conditions</param>
/// <param name="n">: Length of charge density vector</param>
/// <param name="Lx">: Physical length of system</param>
/// <param name="A">: Left matrix</param>
/// <param name="B">: Right matrix</param>
/// <param name="ipiv"></param>
extern void cuPoisson_nonp_matrix(
	cusolverDnHandle_t handle,
	Boundary bc,
	int n,
	double Lx,
	double* A,
	double* B,
	int* ipiv)
{
	double val[3];
	int pos[3];

	double rdx = (n - 1) / Lx;
	double* buffer = NULL;

	int buffersize = 0;
	int sidelength = 0;

	if (strcmp(bc.bc1.c_str(), "Dirichlet") == 0
		&& strcmp(bc.bc2.c_str(), "Dirichlet") == 0)
	{
		val[0] = -rdx * rdx; val[1] = 2 * rdx * rdx; val[2] = -1 * rdx * rdx;
		pos[0] = -1; pos[1] = 0; pos[2] = 1;
		diagmatrix(val, pos, A, 3, n - 2, n - 2);

		val[0] = 1 / 6.0; val[1] = 2 / 3.0; val[2] = 1 / 6.0;
		pos[0] = 0; pos[1] = 1; pos[2] = 2;
		diagmatrix(val, pos, B, 3, n - 2, n);

		sidelength = n - 2;
	}
	else if (strcmp(bc.bc1.c_str(), "Dirichlet") == 0
		&& strcmp(bc.bc2.c_str(), "Neumann") == 0)
	{
		val[0] = -rdx * rdx; val[1] = 2 * rdx * rdx; val[2] = -1 * rdx * rdx;
		pos[0] = -1; pos[1] = 0; pos[2] = 1;
		diagmatrix(val, pos, A, 3, n - 1, n - 1);

		double temp = rdx * rdx;
		cudaMemcpy(A + (n - 1) * (n - 1) - 1, &temp, sizeof(double), cudaMemcpyHostToDevice);

		val[0] = 1 / 6.0; val[1] = 2 / 3.0; val[2] = 1 / 6.0;
		pos[0] = 0; pos[1] = 1; pos[2] = 2;
		diagmatrix(val, pos, B, 3, n - 1, n);

		temp = 1 / 3.0;
		cudaMemcpy(B + (n - 1) * n - 1, &temp, sizeof(double), cudaMemcpyHostToDevice);

		sidelength = n - 1;
	}
	else if (strcmp(bc.bc1.c_str(), "Neumann") == 0
		&& strcmp(bc.bc2.c_str(), "Dirichlet") == 0)
	{
		val[0] = -rdx * rdx; val[1] = 2 * rdx * rdx; val[2] = -1 * rdx * rdx;
		pos[0] = -1; pos[1] = 0; pos[2] = 1;
		diagmatrix(val, pos, A, 3, n - 1, n - 1);

		double temp = rdx * rdx;
		cudaMemcpy(A, &temp, sizeof(double), cudaMemcpyHostToDevice);

		val[0] = 1 / 6.0; val[1] = 2 / 3.0; val[2] = 1 / 6.0;
		pos[0] = -1; pos[1] = 0; pos[2] = 1;
		diagmatrix(val, pos, B, 3, n - 1, n);

		temp = 1 / 3.0;
		cudaMemcpy(B, &temp, sizeof(double), cudaMemcpyHostToDevice);

		sidelength = n - 1;
	}

	cusolverDnDgetrf_bufferSize(handle, sidelength, sidelength, A, sidelength, &buffersize);
	cudaMalloc(&buffer, sizeof(double) * buffersize);

	cusolverDnDgetrf(handle, sidelength, sidelength, A, sidelength, buffer, ipiv, NULL);

	cudaFree(buffer);
}

/// <summary>
/// User API which is used to solve 1-D non-periodic Poisson equation
/// by finite-element method
/// </summary>
/// <param name="handle_m">: cublas context</param>
/// <param name="handle_s">: cusolver context</param>
/// <param name="bc">: Boundart conditions</param>
/// <param name="n">: Lenght of charge density vector</param>
/// <param name="ipiv"></param>
/// <param name="A">: Left matrix</param>
/// <param name="B">: Right matrix</param>
/// <param name="bc_vec">: Right vector of boundary condition</param>
/// <param name="rho">: Charge density vector</param>
/// <param name="phi">: Electrostatic potential</param>
extern void cuPoisson_nonp(
	cublasHandle_t handle_m,
	cusolverDnHandle_t handle_s,
	Boundary bc,
	int n,
	int* ipiv,
	double* A,
	double* B,
	double* bc_vec,
	double* rho,
	double* phi)
{
	double alpha = 1.0;
	double beta = 0.0;

	int length = 0;
	int shift = 0;

	if (strcmp(bc.bc1.c_str(), "Dirichlet") == 0
		&& strcmp(bc.bc2.c_str(), "Dirichlet") == 0)
	{
		length = n - 2;
		shift = 1;
	}
	else if (strcmp(bc.bc1.c_str(), "Dirichlet") == 0
		&& strcmp(bc.bc2.c_str(), "Neumann") == 0)
	{
		length = n - 1;
		shift = 1;
	}
	else if (strcmp(bc.bc1.c_str(), "Neumann") == 0
		&& strcmp(bc.bc2.c_str(), "Dirichlet") == 0)
	{
		length = n - 1;
		shift = 0;
	}

	cublasDgemv_v2(handle_m, CUBLAS_OP_N, length, n, &alpha, B, length, rho, 1, &beta, rho, 1);

	cudaError_t err;
	pointwiseop(bc_vec, rho, length, '+');

	cusolverDnDgetrs(handle_s, CUBLAS_OP_N, length, 1, A, length, ipiv, rho, length, NULL);

	err = cudaMemcpy(phi + shift, rho, sizeof(double) * length, cudaMemcpyDeviceToDevice);
}

#pragma endregion

#pragma region Calculus_integral

__global__ void sum2(double* A, int m, int n, int dim, double *res)
{

	int j;
	double temp;
	switch (dim)
	{
	case 1:
		int x = threadIdx.x + blockIdx.x * blockDim.x;

		if (x < m)
		{
			j = n / 2;

			temp = A[x + j * m];
			for (size_t i = 1; i < j; i++)
			{
				A[x] += A[x + i * m];
				temp += A[x + (i + j) * m];
			}

			A[x] += temp;

			if (2 * j != n)
			{
				A[x] += A[x + (n - 1) * m];
			}

			res[x] = A[x];
		}
		break;
	case 2:
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (y < n)
		{
			j = m / 2;

			temp = A[j + y * m];
			for (size_t i = 1; i < j; i++)
			{
				A[y * m] += A[i + y * m];
				temp += A[i + j + y * m];
			}

			A[y * m] += temp;

			if (2 * j != m)
			{
				A[y * m] += A[m - 1 + y * m];
			}

			res[y] = A[y * m];
		}
		break;
	default:
		break;
	}
}

__global__ void trapz2_y(double* f, double dx, int m, int n, double* res)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	__shared__ double cache[K2][K];

	cache[threadIdx.y][threadIdx.x] = 0;

	if (x < m && y < n)
	{
		cache[threadIdx.y][threadIdx.x] = f[x + y * m];


		if (y == 0 || y == n - 1)
		{
			cache[threadIdx.y][threadIdx.x] *= 0.5 * dx;
		}
		else
		{
			cache[threadIdx.y][threadIdx.x] *= dx;
		}

		//Synchronization within single thread block
		__syncthreads();

		int i = K2 / 2;

		while (i != 0)
		{
			if (threadIdx.y < i)
			{
				cache[threadIdx.y][threadIdx.x] =
					cache[threadIdx.y][threadIdx.x] + cache[i + threadIdx.y][threadIdx.x];
			}
			__syncthreads();
			i = (int)(i * 0.5);
		}

		res[x + blockIdx.y * m] = cache[0][threadIdx.x];
	}
}

__global__ void trapz2_x(double* f, double dx, int m, int n, double* res)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	__shared__ double cache[K2][K];

	cache[threadIdx.y][threadIdx.x] = 0;

	if (x < m && y < n)
	{
		cache[threadIdx.y][threadIdx.x] = f[x + y * m];


		if (x == 0 || x == m - 1)
		{
			cache[threadIdx.y][threadIdx.x] *= 0.5 * dx;
		}
		else
		{
			cache[threadIdx.y][threadIdx.x] *= dx;
		}

		//Synchronization within single thread block
		__syncthreads();

		int i = K / 2;

		while (i != 0)
		{
			if (threadIdx.x < i)
			{
				cache[threadIdx.y][threadIdx.x] =
					cache[threadIdx.y][threadIdx.x] + cache[threadIdx.y][i + threadIdx.x];
			}
			__syncthreads();
			i = (int)(i * 0.5);
		}

		res[blockIdx.x + y * gridDim.x] = cache[threadIdx.y][0];
	}
}

/// <summary>
/// User API which calculates integral by using trapezoidal
/// method
/// </summary>
extern void trapz(double* f, double dx, int m, int n, int dim, double* mid, double* I)
{
	dim3 grid((m + K - 1) / K, (n + K2 - 1) / K2); dim3 block(K, K2);
	switch (dim)
	{
	case 1:
		trapz2_y<<<grid, block>>>(f, dx, m, n, mid);

		grid.y = 1; block.y = 1;
		sum2 << <grid, block >> > (mid, m, (n + K2 - 1) / K2, 1, I);
		break;
	case 2:
		trapz2_x << <grid, block >> > (f, dx, m, n, mid);

		grid.x = 1; block.x = 1;
		sum2 << <grid, block >> > (mid, (m + K - 1)/ K, n, 2, I);
		break;
	default:
		std::cout << "Error: dimension cannot exceed 2 or be zero or negative" << "\n";
		break;
	}
}

#pragma endregion

#pragma region Calculus_differential

__device__ void secondforward(double f1, double f2, double f3,
	double d, double* res)
{
	//Value of index is f1<f2<f3
	res[0] = (-3 * f1 + 4 * f2 - f3) / (2 * d);
}

__device__ void secondbackward(double f1, double f2, double f3,
	double d, double* res)
{
	//Value of index is f1<f2<f3
	res[0] = (f1 - 4 * f2 + 3 * f3) / (2 * d);
}

/// <summary>
/// Kernel function which uses periodic forth-order central difference
/// to calculate first derivative
/// </summary>
__global__ void forthcentral_periodic(double *f, double d, int m,
	int n, int dim, double *df)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	
	int nindex = 0;
	int pindex = 0;

	if (x < m && y < n)
	{
		switch (dim)
		{
		case 2:
			if (x == 0 || x == m - 1)
			{
				nindex = m - 3 + m * y;
				pindex = 1 + m * y;
			}
			else if (x == 1)
			{
				nindex = m - 2 + m * y;
				pindex = 2 + m * y;
			}
			else if (x == m - 2)
			{
				nindex = m - 4 + m * y;
				pindex = 0 + m * y;
			}
			else
			{
				nindex = x - 2 + m * y;
				pindex = x + 1 + m * y;
			}

			df[x + m * y] = (f[nindex] - 8 * f[nindex + 1]
				+ 8 * f[pindex] - f[pindex + 1]) / (12.0 * d);
			break;
		case 1:
			if (y == 0 || y == n - 1)
			{
				nindex = x + m * (n - 3);
				pindex = x + m;
			}
			else if (y == 1)
			{
				nindex = x + m * (n - 2);
				pindex = x + m * 2;
			}
			else if (y == n - 2)
			{
				nindex = x + m * (n - 4);
				pindex = x + m * 0;
			}
			else
			{
				nindex = x + m * (y - 2);
				pindex = x + m * (y + 1);
			}

			df[x + m * y] = (f[nindex] - 8 * f[nindex + m]
				+ 8 * f[pindex] - f[pindex + m]) / (12.0 * d);
			break;
		}
	}


}

/// <summary>
/// Kernel function which uses non-periodic forth-order central difference
/// and second-order forward/backward difference
/// to calculate first derivative
/// </summary>
__global__ void forthcentral_nonp(double* f, double d, int m,
	int n, int dim, double* df)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	int linind = 0;

	double res[1];

	if (x < m && y < n)
	{
		switch (dim)
		{
		case 2:
			linind = x + m * y;
			if (x > 1 && x < m - 2)
			{
				df[linind] = f[linind - 2] - 8 * f[linind - 1]
					+ 8 * f[linind + 1] - f[linind + 2];
				df[linind] = df[linind] / (12.0 * d);
			}

			if (x <= 1)
			{
				secondforward(f[linind], f[linind + 1], f[linind + 2],
					d, res);
				df[linind] = res[0];
			}

			if (x == m - 2 || x == m - 1)
			{
				secondbackward(f[linind - 2], f[linind - 1], f[linind],
					d, res);
				df[linind] = res[0];
			}
			break;
		case 1:
			linind = x + m * y;
			if (y > 1 && y < n - 2)
			{
				df[linind] = f[linind - 2 * m] - 8 * f[linind - m]
					+ 8 * f[linind + m] - f[linind + 2 * m];
				df[linind] = df[linind] / (12.0 * d);
			}

			if (y <= 1)
			{
				secondforward(f[linind], f[linind + m], f[linind + 2 * m],
					d, res);
				df[linind] = res[0];
			}

			if (y == n - 2 || y == n - 1)
			{
				secondbackward(f[linind - 2 * m], f[linind - m], f[linind],
					d, res);
				df[linind] = res[0];
			}
			break;
		default:
			break;
		}
	}	

}

/// <summary>
/// User API which calculates partial derivative on a 2-D
/// matrix by using forth-order central difference
/// </summary>
/// <param name="f">: pointer to 2-D function</param>
/// <param name="df">: pointer to partial derivative</param>
extern void forthcentral(double* f, double d, int m,
	int n, int dim, char type, double* df)
{
	dim3 grid((m + K - 1) / K, (n + K2 - 1) / K2); dim3 block(K, K2);

	switch (type)
	{
	case 'p':
		forthcentral_periodic << <grid, block >> > (f, d, m, n, dim, df);
		break;
	case 'n':
		forthcentral_nonp << <grid, block >> > (f, d, m, n, dim, df);
		break;
	default:
		break;
	}
}

/// <summary>
/// User API which uses fast Fourier transform to 
/// calcualte periodic 1-D derivatives on 2-D matrix
/// </summary>
extern void cuDerivatives(cufftDoubleComplex* f,
	double* k, int n, cufftHandle plan, double *df)
{

}

#pragma endregion

/*!!!---Test functions---!!!*/

extern void forthcentral_test()
{
	int n = 256; int m = 256;
	double* f = new double[n * m];
	double* dfx = new double[n * m];
	double* dfy = new double[n * m];

	double* devPtr_f;
	double* devPtr_dfx;
	double* devPtr_dfy;

	size_t size = sizeof(double) * n * m;

	cudaMalloc((void**)&devPtr_f, size);
	cudaMalloc((void**)&devPtr_dfx, size);
	cudaMalloc((void**)&devPtr_dfy, size);

	double step_v = 8.0 / (m - 1);
	double step_x = 4 * PI / (n - 1);

	for (size_t j = 0; j < n; j++)
	{
		for (size_t i = 0; i < m; i++)
		{
			//f[i + j * m] = exp(-(i * step - PI) * (i * step - PI) / 0.5);
			f[i + j * m] = (1 + 0.1 * cos(j * step_x)) *
				exp(-(-4.0 + i * step_v) * (-4.0 + i * step_v) / 2);
		}
	}

	cudaMemcpy(devPtr_f, f, size, cudaMemcpyHostToDevice);

	dim3 grid(m / K, n / K2); dim3 block(K, K2);

	//forthcentral_nonp << <grid, block >> > (devPtr_f, step_v, m, n, 2, devPtr_dfx);
	forthcentral(devPtr_f, step_v, m, n, 2, 'n', devPtr_dfx);
	cudaMemcpy(dfx, devPtr_dfx, size, cudaMemcpyDeviceToHost);

	cudaFree(devPtr_f);
	cudaFree(devPtr_dfx);
	cudaFree(devPtr_dfy);

	std::ofstream outFile;
	outFile.open("C:\\Users\\59669\\Desktop\\derivative.dat", std::ios::out | std::ios::binary);

	for (size_t i = 0; i < n * m; i++)
	{
		outFile.write((char*)&dfx[i], sizeof(f[i]));
	}

	outFile.close();
}

extern void poisson_test()
{
	int n = 1024;

	cufftDoubleComplex* rho = new cufftDoubleComplex[n];
	
	double* E = new double[n];
	double* k = new double[n];
	double* k_recip = new double[n];

	double w = 2;
	double step = 4 * PI / n;
	for (size_t i = 0; i < n; i++)
	{
		rho[i].x = sin(w * i * step);
		rho[i].y = 0;
	}

	cufftDoubleComplex* devPtr_rho;
	cufftDoubleComplex* devPtr_phi;

	double* devPtr_E;
	double* devPtr_k;
	double* devPtr_k_recip;

	size_t sizedouble = sizeof(double) * n;
	size_t sizecomp = sizeof(cufftDoubleComplex) * n;

	cudaMalloc((void**)&devPtr_k, sizedouble);
	cudaMalloc((void**)&devPtr_k_recip, sizedouble);
	cudaMalloc((void**)&devPtr_E, sizedouble);
	cudaMalloc((void**)&devPtr_phi, sizecomp);
	cudaMalloc((void**)&devPtr_rho, sizecomp);

	cufftHandle plan;
	cufftPlan1d(&plan, n, CUFFT_Z2Z, 1);

	kspace(4 * PI, n, k);
	kspacesquare(4 * PI, n, k_recip);

	cudaMemcpy(devPtr_k, k, sizedouble, cudaMemcpyHostToDevice);
	cudaMemcpy(devPtr_k_recip, k_recip, sizedouble, cudaMemcpyHostToDevice);
	cudaMemcpy(devPtr_rho, rho, sizecomp, cudaMemcpyHostToDevice);

	pointwisereciprocal << <n, 1 >> > (devPtr_k_recip);
	cudaMemcpy(k_recip, devPtr_k_recip, sizedouble, cudaMemcpyDeviceToHost);

	LARGE_INTEGER t1, t2, tc;
	QueryPerformanceFrequency(&tc);
	QueryPerformanceCounter(&t1);

	cuPoisson(devPtr_rho, devPtr_phi, devPtr_k_recip, n, plan);
	cuGradient(devPtr_phi, devPtr_E, devPtr_k, n, plan);

	cudaDeviceSynchronize();

	QueryPerformanceCounter(&t2);

	std::cout << "Time of 8192*256 trapz: " <<
		(double)(t2.QuadPart - t1.QuadPart) / ((double)tc.QuadPart * 1) << "\n";

	cudaFree(devPtr_E);
	cudaFree(devPtr_k);
	cudaFree(devPtr_k_recip);
	cudaFree(devPtr_rho);
	cudaFree(devPtr_phi);

	cufftDestroy(plan);

	std::ofstream outFile;
	outFile.open("C:\\Users\\59669\\Desktop\\efield.dat", std::ios::out | std::ios::binary);

	for (size_t i = 0; i < n; i++)
	{
		outFile.write((char*)&E[i], sizeof(E[i]));
	}

	outFile.close();

	free(E);
	free(k);
	free(k_recip);
}

extern void poisson_nonp_test()
{
	cublasHandle_t handle_m;
	cusolverDnHandle_t handle_s;
	cublasCreate_v2(&handle_m);
	cusolverDnCreate(&handle_s);

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	cusolverDnSetStream(handle_s, stream);
	cublasSetStream_v2(handle_m, stream);

	double* devPtr_A;
	double* devPtr_B;
	double* devPtr_rho;
	double* devPtr_phi;
	double* devPtr_bcvec;

	int* ipiv;
	int n = 1024;

	cudaMalloc((void**)&devPtr_A, sizeof(double) * (n - 1) * (n - 1));
	cudaMalloc((void**)&devPtr_B, sizeof(double) * (n - 1) * n);
	cudaMalloc((void**)&devPtr_rho, sizeof(double) * n);
	cudaMalloc((void**)&devPtr_phi, sizeof(double) * n);
	cudaMalloc((void**)&devPtr_bcvec, sizeof(double) * (n - 1));

	cudaMalloc((void**)&ipiv, sizeof(int) * n);

	double* rho = new double[n];
	double Lx = 3 * PI;
	double dx = Lx / (n - 1);

	for (size_t i = 0; i < n; i++)
	{
		rho[i] = sin(i * dx);
		//std::cout << rho[i] << "\n";
	}

	cublasSetVector(n, sizeof(double), rho, 1, devPtr_rho, 1);

	Boundary boundary;

	boundary.bc1 = "Neumann";
	boundary.bc2 = "Dirichlet";

	boundary.vbc1 = 0;
	boundary.vbc2 = 2;

	cudaError_t err;

	err = cudaMemcpy(devPtr_phi + n - 1, &boundary.vbc2, sizeof(double), cudaMemcpyHostToDevice);

	double temp = boundary.vbc1 / dx;
	err = cudaMemcpy(devPtr_bcvec, &temp, sizeof(double), cudaMemcpyHostToDevice);
	temp = boundary.vbc2 / (dx * dx);
	err = cudaMemcpy(devPtr_bcvec + n - 2, &temp, sizeof(double), cudaMemcpyHostToDevice);

	cuPoisson_nonp_matrix(handle_s, boundary, n, Lx, devPtr_A, devPtr_B, ipiv);

	double* A = new double[(n - 1) * (n - 1)];
	double* B = new double[(n - 1) * n];

	cublasGetMatrix(n - 1, n - 1, sizeof(double), devPtr_A, n - 1, A, n - 1);
	cublasGetMatrix(n - 1, n, sizeof(double), devPtr_B, n - 1, B, n - 1);

	//std::cout << "\n\n";
	//printmatrix(A, n - 1, n - 1);
	//std::cout << "\n\n";
	//printmatrix(B, n - 1, n);

	cuPoisson_nonp(handle_m, handle_s, boundary, n, ipiv, devPtr_A, devPtr_B, devPtr_bcvec, devPtr_rho, devPtr_phi);

	cublasGetVector(n, sizeof(double), devPtr_phi, 1, rho, 1);
	std::cout << "\n\n";
	for (size_t i = 0; i < n; i++)
	{
		std::cout << std::fixed << std::setprecision(15) << rho[i] << "\n";
	}

	delete[] A;
	delete[] B;
	delete[] rho;

	cudaFree(devPtr_A);
	cudaFree(devPtr_B);
	cudaFree(devPtr_rho);
	cudaFree(devPtr_phi);
	cudaFree(ipiv);
}

extern void trapz_test()
{
	int n = 512; int m = 8192;
	double* y = new double[n * m];
	double* r = new double[n * m / K2];
	double* I = new double[m];

	//n--number of integration
	//m--length of integral sequence

	double step = 2*PI / (n - 1);
	for (size_t j = 0; j < n; j++)
	{
		for (size_t i = 0; i < m; i++)
		{
			y[i + j * m] = exp(-(j * step - PI) * (j * step - PI)/0.5);
		}
	}

	size_t size = sizeof(double) * n * m;

	double* devPtr_y = NULL;
	double* devPtr_r = NULL;
	double* devPtr_I = NULL;

	cudaMalloc((void**)&devPtr_y, size);
	cudaMalloc((void**)&devPtr_r, sizeof(double) * n * m / K2);
	cudaMalloc((void**)&devPtr_I, sizeof(double) * m);

	cudaMemcpy(devPtr_y, y, size, cudaMemcpyHostToDevice);

	dim3 grid(m / K, n / K2); dim3 block(K, K2);

	trapz2_y << <grid, block >> > (devPtr_y, step, m, n, devPtr_r);
	cudaDeviceSynchronize();

	grid.y = 1; block.y = 1;
	sum2 << <grid, block >> > (devPtr_r, m, n / K2, 1, devPtr_I);
	cudaDeviceSynchronize();

	cudaMemcpy(I, devPtr_I, sizeof(double) * m, cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < m; i++)
	{
		std::cout << I[i] << "\n";
	}

	cudaFree(devPtr_y);
	cudaFree(devPtr_r);
	cudaFree(devPtr_I);

	delete[] y;
	delete[] r;
	delete[] I;
}

extern void cuMatMulti_test()
{
	int n = 5; int m = 10;

	double* A = new double[n * m];
	double* r = new double[m];
	double* x = new double[n];

	double* devPtr_A;
	double* devPtr_r;
	double* devPtr_x;

	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			A[j + i * n] = j + i * n;
		}
	}

	for (size_t i = 0; i < m; i++)
		r[i] = i;

	cublasHandle_t handle;
	cublasCreate_v2(&handle);

	cudaMalloc((void**)&devPtr_A, sizeof(double) * n * m);
	cudaMalloc((void**)&devPtr_r, sizeof(double) * m);
	cudaMalloc((void**)&devPtr_x, sizeof(double) * n);

	cublasSetVector(m, sizeof(double), r, 1, devPtr_r, 1);
	cublasSetMatrix(n, m, sizeof(double), A, n, devPtr_A, n);

	double alpha = 1; double beta = 0;

	cublasDgemv_v2(handle, CUBLAS_OP_N, n, m, &alpha, devPtr_A, n,
		devPtr_r, 1, &beta, devPtr_x, 1);

	cublasGetVector(n, sizeof(double), devPtr_x, 1, x, 1);

	for (size_t i = 0; i < n; i++)
	{
		std::cout << x[i] << "\n";
	}

	cudaFree(devPtr_A);
	cudaFree(devPtr_r);
	cudaFree(devPtr_x);

	delete[] A;
	delete[] r;
	delete[] x;
}

/*!!!---End of test functions---!!!*/