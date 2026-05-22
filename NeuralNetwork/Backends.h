#pragma once

#include "Matrix.h"

struct SequentialBackend
{
    static Matrix matmul(const Matrix& A, const Matrix& B)
    {
        return A.matmul(B);
    }

    static Matrix mul(const Matrix& A, const Matrix& B)
    {
        return A * B;
    }

    static Matrix scalarMul(const Matrix& A, float x)
    {
        return A * x;
    }

    static Matrix div(const Matrix& A, const Matrix& B)
    {
        return A / B;
    }

    static Matrix divScalar(const Matrix& A, float x)
    {
        return A / x;
    }

    static Matrix scalarDiv(float x, const Matrix& A)
    {
        return x / A;
    }

    static Matrix add(const Matrix& A, const Matrix& B)
    {
        return A + B;
    }

    static Matrix scalarAdd(const Matrix& A, float x)
    {
        return A + x;
    }

    static Matrix sub(const Matrix& A, const Matrix& B)
    {
        return A - B;
    }

    static Matrix subScalar(const Matrix& A, float x)
    {
        return A - x;
    }

    static Matrix scalarSub(float x, const Matrix& A)
    {
        return x - A;
    }

    static Matrix equals(const Matrix& A, const Matrix& B)
    {
        return A == B;
    }

    static Matrix greaterth(const Matrix& A, float x)
    {
        return A > x;
    }

    static Matrix broadcastAdd(const Matrix& A, const Matrix& B)
    {
        return A.broadcastAdd(B);
    }

    static Matrix T(const Matrix& A)
    {
        return A.T();
    }

    static Matrix random(int rows, int cols)
    {
        return Matrix::random(rows, cols);
    }

    static Matrix sum(const Matrix& A, int axis = -1)
    {
        return Matrix::sum(A, axis);
    }

    static Matrix powM(const Matrix& A, float power)
    {
        return Matrix::powM(A, power);
    }

    static Matrix sqrtM(const Matrix& A)
    {
        return Matrix::sqrtM(A);
    }

    static Matrix expM(const Matrix& A)
    {
        return Matrix::expM(A);
    }

    static Matrix logM(const Matrix& A)
    {
        return Matrix::logM(A);
    }

    static Matrix maxM(const Matrix& A, int axis = -1)
    {
        return Matrix::maxM(A, axis);
    }

    static Matrix argmax(const Matrix& A, int axis = 0)
    {
        return Matrix::argmax(A, axis);
    }

    static Matrix clipM(const Matrix& A, float minValue, float maxValue)
    {
        return Matrix::clipM(A, minValue, maxValue);
    }
};

struct OpenMPBackend
{
    static Matrix matmul(const Matrix& A, const Matrix& B);

    static Matrix mul(const Matrix& A, const Matrix& B);

    static Matrix scalarMul(const Matrix& A, float x);

    static Matrix div(const Matrix& A, const Matrix& B);

    static Matrix divScalar(const Matrix& A, float x);

    static Matrix scalarDiv(float x, const Matrix& A);

    static Matrix add(const Matrix& A, const Matrix& B);

    static Matrix scalarAdd(const Matrix& A, float x);

    static Matrix sub(const Matrix& A, const Matrix& B);

    static Matrix subScalar(const Matrix& A, float x);

    static Matrix scalarSub(float x, const Matrix& A);

    static Matrix equals(const Matrix& A, const Matrix& B);

    static Matrix greaterth(const Matrix& A, float x);

    static Matrix broadcastAdd(const Matrix& A, const Matrix& B);

    static Matrix T(const Matrix& A);

    static Matrix random(int rows, int cols);

    static Matrix sum(const Matrix& A, int axis = -1);

    static Matrix powM(const Matrix& A, float power);

    static Matrix sqrtM(const Matrix& A);

    static Matrix expM(const Matrix& A);

    static Matrix logM(const Matrix& A);

    static Matrix maxM(const Matrix& A, int axis = -1);

    static Matrix argmax(const Matrix& A, int axis = 0);

    static Matrix clipM(const Matrix& A, float minValue, float maxValue);
};

struct CUDABackend
{
    static CMatrix matmul(const CMatrix& A, const CMatrix& B)
    {
        return A.matmul(B);
    }

    static CMatrix mul(const CMatrix& A, const CMatrix& B)
    {
        return A * B;
    }

    static CMatrix scalarMul(const CMatrix& A, float x)
    {
        return A * x;
    }

    static CMatrix div(const CMatrix& A, const CMatrix& B)
    {
        return A / B;
    }

    static CMatrix divScalar(const CMatrix& A, float x)
    {
        return A / x;
    }

    static CMatrix add(const CMatrix& A, const CMatrix& B)
    {
        return A + B;
    }

    static CMatrix scalarAdd(const CMatrix& A, float x)
    {
        return A + x;
    }

    static CMatrix sub(const CMatrix& A, const CMatrix& B)
    {
        return A - B;
    }

    static CMatrix subScalar(const CMatrix& A, float x)
    {
        return A - x;
    }

    static CMatrix equals(const CMatrix& A, const CMatrix& B)
    {
        return A == B;
    }

    static CMatrix greaterth(const CMatrix& A, float x)
    {
        return A > x;
    }

    static CMatrix broadcastAdd(const CMatrix& A, const CMatrix& B)
    {
        return A.broadcastAdd(B);
    }

    static CMatrix T(const CMatrix& A)
    {
        return A.T();
    }

    static CMatrix random(int rows, int cols)
    {
        return CMatrix::random(rows, cols);
    }

    static CMatrix sum(const CMatrix& A, int axis = -1)
    {
        return CMatrix::sum(A, axis);
    }

    static CMatrix powM(const CMatrix& A, float power)
    {
        return CMatrix::powM(A, power);
    }

    static CMatrix sqrtM(const CMatrix& A)
    {
        return CMatrix::sqrtM(A);
    }

    static CMatrix expM(const CMatrix& A)
    {
        return CMatrix::expM(A);
    }

    static CMatrix logM(const CMatrix& A)
    {
        return CMatrix::logM(A);
    }

    static CMatrix maxA(const CMatrix& A, int axis = -1)
    {
        return CMatrix::maxA(A, axis);
    }
};