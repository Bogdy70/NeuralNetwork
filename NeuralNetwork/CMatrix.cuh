#include <cuda_runtime.h>

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
};