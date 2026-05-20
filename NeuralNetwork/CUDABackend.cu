#include <stdexcept>
#include <cuda_runtime.h>
#include "Backends.h"

using namespace std;

__global__ void matmulKernel(const float* A, const float* B, float* C, int N, int M, int K)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M)
	{
		float sum = 0.0f;

		for (int k = 0; k < K; k++)
		{
			sum += A[i * K + k] * B[k * M + j];
		}

		C[i * M + j] = sum;
	}
}

Matrix CUDABackend::matmul(const Matrix& A, const Matrix& B)
{
	if (A.getCols() != B.getRows())
		throw runtime_error("Invalid shape for matrix multiplication");

	Matrix C(A.getRows(), B.getCols());

	float* d_A = nullptr;
	float* d_B = nullptr;
	float* d_C = nullptr;

	size_t bytesA = A.size() * sizeof(float);
	size_t bytesB = B.size() * sizeof(float);
	size_t bytesC = C.size() * sizeof(float);

	cudaMalloc(&d_A, bytesA);
	cudaMalloc(&d_B, bytesB);
	cudaMalloc(&d_C, bytesC);

	dim3 block(16, 16);
	dim3 grid((B.getCols() + block.x - 1) / block.x, (A.getRows() + block.y - 1) / block.y);

	cudaMemcpy(d_A, A.rawData(), bytesA, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B.rawData(), bytesB, cudaMemcpyHostToDevice);

	matmulKernel << <grid, block >> > (d_A, d_B, d_C, A.getRows(), B.getCols(), A.getCols());

	cudaMemcpy(C.rawData(), d_C, bytesC, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return C;
}

__global__ void mulKernel(const float* A, const float* B, float* C, int N, int M)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M)
	{
		C[i * M + j] = A[i * M + j] * B[i * M + j];
	}
}

Matrix CUDABackend::mul(const Matrix& A, const Matrix& B)
{
	if (A.getRows() != B.getRows() || A.getCols() != B.getCols())
		throw runtime_error("Invalid shape");

	Matrix C(B.getRows(), B.getCols());

	float* d_A = nullptr;
	float* d_B = nullptr;
	float* d_C = nullptr;

	size_t bytesA = A.size() * sizeof(float);
	size_t bytesB = B.size() * sizeof(float);
	size_t bytesC = C.size() * sizeof(float);

	cudaMalloc(&d_A, bytesA);
	cudaMalloc(&d_B, bytesB);
	cudaMalloc(&d_C, bytesC);

	dim3 block(16, 16);
	dim3 grid((A.getCols() + block.x - 1) / block.x, (A.getRows() + block.y - 1) / block.y);

	cudaMemcpy(d_A, A.rawData(), bytesA, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B.rawData(), bytesB, cudaMemcpyHostToDevice);

	mulKernel << <grid, block >> > (d_A, d_B, d_C, A.getRows(), A.getCols());

	cudaMemcpy(C.rawData(), d_C, bytesC, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return C;
}