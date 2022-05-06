// Vlasov_simulator.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <cmath>

#include "cusolverDn.h"
#include "tinyxml2.h"
#include "vlasov1d.h"

#define PI acos(-1.0)

extern void forthcentral_test();
extern void cuMatMulti_test();
extern void poisson_nonp_test();

using namespace tinyxml2;

/*
int linearSolverLU(
	cusolverDnHandle_t handle,
	int n,
	const double* Acopy,
	int lda,
	const double* b,
	double* x)
{
	int bufferSize = 0;
	int* info = NULL;
	double* buffer = NULL;
	double* A = NULL;
	int* ipiv = NULL; // pivoting sequence
	int h_info = 0;
	double start, stop;
	double time_solve;

	//calculate the necessary size of work buffers
	cusolverDnDgetrf_bufferSize(handle, n, n, (double*)Acopy, lda, &bufferSize);

	cudaMalloc(&info, sizeof(int));
	cudaMalloc(&buffer, sizeof(double) * bufferSize);
	cudaMalloc(&A, sizeof(double) * lda * n);
	cudaMalloc(&ipiv, sizeof(int) * n);


	// prepare a copy of A because getrf will overwrite A with L
	cudaMemcpy(A, Acopy, sizeof(double) * lda * n, cudaMemcpyDeviceToDevice);
	cudaMemset(info, 0, sizeof(int));

	cusolverDnDgetrf(handle, n, n, A, lda, buffer, ipiv, info);

	cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost);

	if (0 != h_info) {
		fprintf(stderr, "Error: LU factorization failed\n");
	}

	cudaMemcpy(x, b, sizeof(double) * n, cudaMemcpyDeviceToDevice);
	cusolverDnDgetrs(handle, CUBLAS_OP_N, n, 1, A, lda, ipiv, x, n, info);
	cudaDeviceSynchronize();
	if (info) { (cudaFree(info)); }
	if (buffer) { (cudaFree(buffer)); }
	if (A) { (cudaFree(A)); }
	if (ipiv) { (cudaFree(ipiv)); }

	return 0;
}

*/

int main()
{
	forthcentral_test();
	Particle particle = loadplasma("C:\\Programmer\\Vlasov_ES1D\\Plasma.xml");

	Solver solver = loadsolver("C:\\Programmer\\Vlasov_ES1D\\Solver.xml", particle.nspecies);

	es1d_vsiumlator simulator(solver, particle);

	simulator.initialize();
	simulator.start_simulation();
	
	solver.dispose();
	particle.dispose();

	return 0;
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
