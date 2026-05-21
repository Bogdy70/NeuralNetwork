#include "CMatrix.cuh"

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