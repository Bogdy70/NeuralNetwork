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
