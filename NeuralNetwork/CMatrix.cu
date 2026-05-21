#include "CMatrix.cuh"
#include "Matrix.h"
#include <stdexcept>

CMatrix::CMatrix(): rows(0), cols(0), data(nullptr) {}

CMatrix::CMatrix(int r, int c) : rows(r), cols(c), data(nullptr)
{
	cudaMalloc(&data, r * c * sizeof(float));
}

CMatrix::~CMatrix()
{
	if(data!=nullptr)
		cudaFree(data);
}

CMatrix::CMatrix(CMatrix&& other) noexcept: rows(other.rows), cols(other.cols), data(other.data)
{
	other.rows = 0;
	other.cols = 0;
	other.data = nullptr;
}

void checkCuda(cudaError_t result, const char* message)
{
	if (result != cudaSuccess)
	{
		throw std::runtime_error(
			std::string(message) + ": " + cudaGetErrorString(result)
		);
	}
}

CMatrix& CMatrix::operator=(CMatrix&& other) noexcept
{
	if (this != &other)
	{
		if (data != nullptr)
			cudaFree(data);

		rows = other.rows;
		cols = other.cols;
		data = other.data;

		other.rows = 0;
		other.cols = 0;
		other.data = nullptr;
	}

	return *this;
}

int CMatrix::getRows() const
{
	return rows;
}

int CMatrix::getCols() const
{
	return cols;
}

int CMatrix::size() const
{
	return rows * cols;
}

float* CMatrix::rawData()
{
	return data;
}

const float* CMatrix::rawData() const
{
	return data;
}

Matrix CMatrix::toCPU() const
{
	Matrix C(rows, cols);

	cudaMemcpy(C.rawData(), data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

	return C;
}

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

CMatrix CMatrix::matmul(const CMatrix& B) const
{
	if (cols != B.getRows())
		throw std::runtime_error("Invalid shape for matrix multiplication");

	CMatrix C(rows, B.getCols());

	dim3 block(16, 16);
	dim3 grid((B.getCols() + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

	matmulKernel<<<grid, block>>>(data, B.rawData(), C.rawData(), rows, B.getCols(), cols);

	checkCuda(cudaGetLastError(), "matmulKernel launch failed");
	checkCuda(cudaDeviceSynchronize(), "matmulKernel execution failed");

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

CMatrix CMatrix::operator*(const CMatrix& B) const
{
	if (rows != B.getRows() || cols != B.getCols())
		throw std::runtime_error("Invalid shape");

	CMatrix C(B.getRows(), B.getCols());

	dim3 block(16, 16);
	dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

	mulKernel << <grid, block >> > (data, B.rawData(), C.rawData(), rows, cols);

	return C;
}

__global__ void scalarMulKernel(const float* A, float x, float* C, int N, int M)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M)
	{
		C[i * M + j] = A[i * M + j] * x;
	}
}

CMatrix CMatrix::operator*(float x) const
{
	CMatrix C(rows, cols);

	dim3 block(16, 16);
	dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

	scalarMulKernel << <grid, block >> > (data, x, C.rawData(), rows, cols);

	return C;
}

__global__ void divKernel(const float* A, const float* B, float* C, int N, int M)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M)
	{
		C[i * M + j] = A[i * M + j] / B[i * M + j];
	}
}

CMatrix CMatrix::operator/(const CMatrix& B) const
{
	if(rows!=B.getRows() || cols!=B.getCols())
		throw std::runtime_error("Invalid shape for element wise division");

	CMatrix C(rows, cols);

	dim3 block(16, 16);
	dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

	divKernel << <grid, block >> > (data, B.rawData(), C.rawData(), rows, cols);

	return C;
}

__global__ void divScalarKernel(const float* A, float x, float* C, int N, int M)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M)
	{
		C[i * M + j] = A[i * M + j] / x;
	}
}

CMatrix CMatrix::operator/(float x) const
{
	if (x==0)
		throw std::runtime_error("Division by zero");

	CMatrix C(rows, cols);

	dim3 block(16, 16);
	dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

	divScalarKernel << <grid, block >> > (data, x, C.rawData(), rows, cols);

	return C;
}

__global__ void addKernel(const float* A, const float* B, float* C, int N, int M)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M)
	{
		C[i * M + j] = A[i * M + j] + B[i * M + j];
	}
}

CMatrix CMatrix::operator+(const CMatrix& B) const
{
	if (rows != B.getRows() || cols != B.getCols())
		throw std::runtime_error("Invalid shape for element wise division");

	CMatrix C(rows, cols);

	dim3 block(16, 16);
	dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

	addKernel << <grid, block >> > (data, B.rawData(), C.rawData(), rows, cols);

	return C;
}

__global__ void scalarAddKernel(const float* A, float x, float* C, int N, int M)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N && j < M)
	{
		C[i * M + j] = A[i * M + j] + x;
	}
}

CMatrix CMatrix::operator+(float x) const
{
	CMatrix C(rows, cols);

	dim3 block(16, 16);
	dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

	scalarAddKernel << <grid, block >> > (data, x, C.rawData(), rows, cols);

	return C;
}