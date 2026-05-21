#pragma once

#include <cuda_runtime.h>

class Matrix;

class CMatrix
{
private:
	int rows;
	int cols;
	float* data;

public:
	CMatrix();

	CMatrix(int r, int c);

	~CMatrix();

	CMatrix(const CMatrix&) = delete;
	CMatrix& operator=(const CMatrix&) = delete;

	CMatrix(CMatrix&& other) noexcept;
	CMatrix& operator=(CMatrix&& other) noexcept;

	int getRows() const;

	int getCols() const;

	int size() const;

	float* rawData();

	const float* rawData() const;

	Matrix toCPU() const;

	CMatrix matmul(const CMatrix& B) const;

	CMatrix operator*(const CMatrix& B) const;

	CMatrix operator*(float x) const;

	CMatrix operator/(const CMatrix& B) const;

	CMatrix operator/(float x) const;

	CMatrix operator+(const CMatrix& B) const;

	CMatrix operator+(float x) const;
};