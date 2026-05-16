// NeuralNetwork.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <string>
#include <sstream>
#include <random>
#include <cstdint>
#include <chrono>
#include <filesystem>
#include <stdexcept>

using namespace std;

random_device rd;
mt19937 gen(rd());

struct Matrix
{
    int rows;
    int cols;
    vector<float> data;

    Matrix() : rows(0), cols(0) {}

    Matrix(int r, int c) : rows(r), cols(c), data(r*c) {}

    float& operator()(int row, int col)
    {
        return data[row * cols + col];
    }

    const float& operator()(int row, int col) const
    {
        return data[row * cols + col];
    }

    Matrix matmul(const Matrix& B) const
    {
        if (cols != B.rows)
        {
            throw runtime_error("Invalid matrix shape for multiplication");
        }

        Matrix C(rows, B.cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < B.cols; j++)
            {
                float sum = 0.0f;

                for (int k = 0; k < cols; k++)
                {
                    sum += data[i*cols+k] * B(k, j);
                }
                C(i, j) = sum;
            }
        }

        return C;
    }

    Matrix operator*(const Matrix& B) const
    {
        if (rows != B.rows || cols != B.cols)
        {
            throw runtime_error("Invalid shape for element wise multiplication");
        }

        Matrix C(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                C(i, j) = data[i * cols + j] * B(i, j);
            }
        }

        return C;
    }

    Matrix operator*(float x) const
    {
        Matrix C(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                C(i, j) = data[i * cols + j] * x;
            }
        }

        return C;
    }

    Matrix operator/(const Matrix& B) const
    {
        if (rows != B.rows || cols != B.cols)
        {
            throw runtime_error("Invalid shape for element wise division");
        }

        Matrix C(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (B(i, j) == 0)
                    throw runtime_error("Division by zero");
                C(i, j) = data[i * cols + j] / B(i, j);
            }
        }

        return C;
    }

    Matrix operator/(float x) const
    {
        if (x == 0)
            throw runtime_error("Division by zero");

        Matrix C(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                C(i, j) = data[i * cols + j] / x;
            }
        }

        return C;
    }

    Matrix operator+(const Matrix& B) const
    {
        if (rows != B.rows || cols != B.cols)
        {
            throw runtime_error("Invalid shape for element wise addition");
        }

        Matrix C(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                C(i, j) = data[i * cols + j] + B(i, j);
            }
        }

        return C;
    }

    Matrix operator+(float x) const
    {
        Matrix C(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                C(i, j) = data[i * cols + j] + x;
            }
        }

        return C;
    }

    Matrix operator-(const Matrix& B) const
    {
        if (rows != B.rows || cols != B.cols)
        {
            throw runtime_error("Invalid shape for substraction");
        }

        Matrix C(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                C(i, j) = data[i * cols + j] - B(i, j);
            }
        }

        return C;
    }

    Matrix operator-(float x) const
    {
        Matrix C(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                C(i, j) = data[i * cols + j] - x;
            }
        }

        return C;
    }

    Matrix operator==(const Matrix& B) const
    {
        if (rows != B.rows || cols != B.cols)
            throw runtime_error("Invalid shape");

        Matrix C(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                C(i, j) = data[i * cols + j] == B(i, j) ? 1.0f : 0.0f;
            }
        }

        return C;
    }

    Matrix operator>(float x) const
    {
        Matrix C(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                C(i, j) = data[i * cols + j] > x ? 1.0f : 0.0f;
            }
        }

        return C;
    }

    Matrix broadcastAdd(const Matrix& B) const
    {
        if (rows != B.rows || B.cols != 1)
        {
            throw runtime_error("Invalid shape for addition with broadcast");
        }

        Matrix C(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                C(i, j) = data[i * cols + j] + B(i, 0);
            }
        }

        return C;
    }

    Matrix T() const
    {
        Matrix Tr(cols, rows);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                Tr(j, i) = data[i * cols + j];
            }
        }

        return Tr;
    }
};

Matrix random(const int rows, const int cols)
{
    normal_distribution<float> dist(0.0f, 1.0f);

    Matrix C(rows, cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            C(i, j) = dist(gen);
        }
    }

    return C;
}

Matrix operator*(float x, const Matrix& A)
{
    return A * x;
}

Matrix operator+(float x, const Matrix& A)
{
    return A + x;
}

Matrix operator/(float x, const Matrix& A)
{
    Matrix C(A.rows, A.cols);

    for (int i = 0; i < A.rows; i++)
    {
        for(int j=0; j < A.cols; j++)
        {
            if (A(i, j) == 0)
                throw runtime_error("Division by zero");
            C(i, j) = x / A(i, j);
        }
    }

    return C;
}

Matrix operator-(float x, const Matrix& A)
{
    Matrix C(A.rows, A.cols);

    for (int i = 0; i < A.rows; i++)
    {
        for(int j = 0; j < A.cols; j++)
        {
            C(i, j) = x - A(i, j);
        }
    }

    return C;
}

Matrix sum(const Matrix& A, int axis=-1)
{
    if (axis == 0)
    {
        Matrix C(1, A.cols);

        for (int i = 0; i < A.cols; i++)
        {
            float sum = 0.0f;

            for (int j = 0; j < A.rows; j++)
            {
                sum += A(j, i);
            }
            C(0, i) = sum;
        }
        return C;
    }
    else if (axis == 1)
    {
        Matrix C(A.rows, 1);

        for (int i = 0; i < A.rows; i++)
        {
            float sum = 0.0f;

            for (int j = 0; j < A.cols; j++)
            {
                sum += A(i, j);
            }
            C(i, 0) = sum;
        }
        return C;
    }
    else if(axis==-1)
    {
        Matrix res(1, 1);
        float sum=0.0f;

        for (int i = 0; i < A.rows; i++)
        {
            for (int j = 0; j < A.cols; j++)
            {
                sum += A(i, j);
            }
        }
        res.data = { sum };
        return res;
    }
    else
    {
        throw runtime_error("Invalid axis input. Please input -1, 0 or 1");
    }
}

Matrix powM(const Matrix& A, float power)
{
    Matrix C(A.rows, A.cols);

    for (int i = 0; i < A.rows; i++)
    {
        for (int j = 0; j < A.cols; j++)
        {
            C(i, j) = pow(A(i, j), power);
        }
    }

    return C;
}

Matrix sqrtM(const Matrix& A)
{
    Matrix C(A.rows, A.cols);

    for (int i = 0; i < A.rows; i++)
    {
        for (int j = 0; j < A.cols; j++)
        {
            if (A(i, j) < 0)
                throw runtime_error("Numbers must not be negative");
            C(i, j) = sqrtf(A(i, j));
        }
    }

    return C;
}

Matrix expM(const Matrix& A)
{
    Matrix C(A.rows, A.cols);

    for (int i = 0; i < A.rows; i++)
    {
        for (int j = 0; j < A.cols; j++)
        {
            C(i, j) = exp(A(i, j));
        }
    }
    return C;
}

Matrix logM(const Matrix& A)
{
    Matrix C(A.rows, A.cols);

    for (int i = 0; i < A.rows; i++)
    {
        for (int j = 0; j < A.cols; j++)
        {
            if (A(i, j) <= 0)
                throw runtime_error("Matrix values must be positive");
            C(i, j) = log(A(i, j));
        }
    }
 
    return C;
}

Matrix maxM(const Matrix& A, int axis = -1)
{
    if (axis == 0)
    {
        Matrix C(1, A.cols);

        for (int i = 0; i < A.cols; i++)
        {
            float maxA = A(0, i);

            for (int j = 1; j < A.rows; j++)
            {
                maxA = max(maxA, A(j, i));
            }
            C(0, i) = maxA;
        }

        return C;
    }
    else if (axis == 1)
    {
        Matrix C(A.rows, 1);

        for (int i = 0; i < A.rows; i++)
        {
            float maxA = A(i, 0);

            for (int j = 1; j < A.cols; j++)
            {
                maxA = max(maxA, A(i, j));
            }
            C(i, 0) = maxA;
        }

        return C;
    }
    else if (axis == -1)
    {
        Matrix C(1, 1);

        float maxA = A(0, 0);

        for (int i = 0; i < A.rows; i++)
        {
            for (int j = 0; j < A.cols; j++)
            {
                maxA = max(maxA, A(i, j));
            }
        }

        C(0, 0) = maxA;
        return C;
    }
    else
        throw runtime_error("Invalid axis value. Please input -1, 0 or 1");
}

Matrix argmax(const Matrix& A, int axis = 0)
{
    if (axis == 0)
    {
        Matrix C(1, A.cols);

        for (int i = 0; i < A.cols; i++)
        {
            int idx = 0;
            float maxA = A(0, i);

            for (int j = 1; j < A.rows; j++)
            {
                if (A(j, i) > maxA)
                {
                    maxA = A(j, i);
                    idx = j;
                }
            }
            C(0, i) = idx;
        }

        return C;
    }
    else if (axis == 1)
    {
        Matrix C(A.rows, 1);

        for (int i = 0; i < A.rows; i++)
        {
            int idx = 0;
            float maxA = A(i, 0);

            for (int j = 0; j < A.cols; j++)
            {
                if (A(i, j) > maxA)
                {
                    maxA = A(i, j);
                    idx = j;
                }
            }
            C(i, 0) = idx;
        }

        return C;
    }
    else
        throw runtime_error("Invalid axis false. Please input 0 or 1");
}

Matrix clipM(const Matrix& A, float minValue, float maxValue)
{
    Matrix C(A.rows, A.cols);

    for (int i = 0; i < A.rows; i++)
    {
        for (int j = 0; j < A.cols; j++)
        {
            C(i, j) = max(minValue, min(maxValue, A(i, j)));
        }
    }

    return C;
}

Matrix sigmoid(const Matrix& A)
{
    return 1.0f / (1.0f + expM((-1.0f) * A));
}

Matrix softmax(const Matrix& A)
{
    Matrix C(A.rows, A.cols);
    Matrix exp_A = expM(A);
    Matrix sum_exp = sum(exp_A, 0);

    for (int i = 0; i < A.cols; i++)
    {
        for (int j = 0; j < A.rows; j++)
        {
            C(j, i) = exp_A(j, i) / sum_exp(0, i);
        }
    }

    return C;
}

Matrix tanhM(const Matrix& A)
{
    Matrix C(A.rows, A.cols);

    for (int i = 0; i < A.rows; i++)
    {
        for (int j = 0; j < A.cols; j++)
        {
            C(i, j) = tanh(A(i, j));
        }
    }

    return C;
}

Matrix der_tanh(const Matrix& A)
{
    return 1 - powM(tanhM(A), 2);
}

Matrix relu(const Matrix& A)
{
    Matrix C(A.rows, A.cols);

    for (int i = 0; i < A.rows; i++)
    {
        for (int j = 0; j < A.cols; j++)
        {
            C(i, j) = max(0.0f, A(i, j));
        }
    }

    return C;
}

Matrix der_relu(const Matrix& A)
{
    Matrix C(A.rows, A.cols);

    for (int i = 0; i < A.rows; i++)
    {
        for (int j = 0; j < A.cols; j++)
        {
            C(i, j) = A(i, j) > 0.0f ? 1.0f : 0.0f;
        }
    }

    return C;
}

float cost(const Matrix& Y, const Matrix& pred)
{
    int m = Y.cols;
    float epsilon = 1e-8f;
    Matrix clippedPred = clipM(pred, epsilon, 1.0f - epsilon);

    if (Y.rows > 1)
        return (-1.0f / static_cast<float>(m)) * sum(Y * logM(clippedPred))(0, 0);
    else
        return (-1.0f / static_cast<float>(m)) * sum(Y * logM(clippedPred) + (1 - Y) * logM(1 - clippedPred))(0, 0);
}

float accuracy(const Matrix& Y, const Matrix& pred)
{
    if (Y.rows > 1)
    {
        Matrix true_labels = argmax(Y);
        Matrix pred_labels = argmax(pred);

        return sum(true_labels == pred_labels)(0, 0) / static_cast<float>(Y.cols);
    }
    else
    {
        Matrix pred_labels = pred > 0.5f;
        return sum(Y==pred_labels)(0, 0) / static_cast<float>(Y.cols);
    }
}

struct Parameters
{
    vector<Matrix> W;
    vector<Matrix> B;
};

struct Forward
{
    vector<Matrix> Z;
    vector<Matrix> A;
};

struct Backward
{
    vector<Matrix> dZ;
    vector<Matrix> dW;
    vector<Matrix> dB;
};

struct Activation
{
    Matrix(*forward)(const Matrix&);
    Matrix(*derivate)(const Matrix&);

    Activation(const string& name)
    {
        if (name == "relu")
        {
            forward = relu;
            derivate = der_relu;
        }
        else if (name == "tanh")
        {
            forward = tanhM;
            derivate = der_tanh;
        }
        else
            throw runtime_error("Invalid activation type. Please choose 'relu' or 'tanh'");
    }
};

struct NetworkConfig
{
    vector<int> dims;
    Activation activation;

    NetworkConfig(const vector<int>& dim_list, const string& activ_name) : dims(dim_list), activation(activ_name) {}
};

Matrix loadMatrixBins(const string& filepath, int rows, int cols)
{
    Matrix mat(rows, cols);

    ifstream file(filepath, ios::binary);

    if (!file)
    {
        throw runtime_error("File " + filepath + " could not be opened");
    }

    file.read(
        reinterpret_cast<char*>(mat.data.data()),
        rows * cols * sizeof(float)
    );

    if (!file)
    {
        throw runtime_error("Error while reading file" + filepath);
    }

    return mat;
}

void printMnistImage(const Matrix& img, int sample_idx)
{
    for (int i = 0; i < 784; i++)
    {
        if (i % 28 == 0)
            cout << "\n";

        float value = img(i, sample_idx);

        if (value > 0.7f)
            cout << "#";
        else if (value > 0.3f)
            cout << ".";
        else
            cout << " ";
    }
}

int main()
{
    try
    {
        Matrix X_train_cat = loadMatrixBins("data/cat/X_train.bin", 12288, 209);
        Matrix y_train_cat = loadMatrixBins("data/cat/Y_train.bin", 1, 209);

        Matrix X_test_cat = loadMatrixBins("data/cat/X_test.bin", 12288, 50);
        Matrix y_test_cat = loadMatrixBins("data/cat/Y_test.bin", 1, 50);

        Matrix X_train_mnist = loadMatrixBins("data/mnist/X_train.bin", 784, 5000);
        Matrix y_train_mnist = loadMatrixBins("data/mnist/Y_train.bin", 10, 5000);

        Matrix X_test_mnist = loadMatrixBins("data/mnist/X_test.bin", 784, 1000);
        Matrix y_test_mnist = loadMatrixBins("data/mnist/Y_test.bin", 10, 1000);

        cout << "Cat dataset loaded successfully\n";

        cout << "X_train_cat: (" << X_train_cat.rows << ", " << X_train_cat.cols << ")\n";
        cout << "y_train_cat: (" << y_train_cat.rows << ", " << y_train_cat.cols << ")\n";
        
        cout << "X_test_cat: (" << X_test_cat.rows << ", " << X_test_cat.cols << ")\n";
        cout << "y_test_cat: (" << y_test_cat.rows << ", " << y_test_cat.cols << ")\n";

        cout << "\nMnist dataset loaded successfully\n";

        cout << "X_train_mnist: (" << X_train_mnist.rows << ", " << X_train_mnist.cols << ")\n";
        cout << "y_train_mnist: (" << y_train_mnist.rows << ", " << y_train_mnist.cols << ")\n";

        cout << "X_test_mnist: (" << X_test_mnist.rows << ", " << X_test_mnist.cols << ")\n";
        cout << "y_test_mnist: (" << y_test_mnist.rows << ", " << y_test_mnist.cols << ")\n";

        cout << "\nFirst X_train_cat examples\n";
        for (int i = 0; i < 10; i++)
        {
            cout << X_train_cat(i, 0) << " ";
        }

        cout << "\n\nFirst y_train_cat value\n";
        cout << y_train_cat(0, 0) << "\n";

        cout << "\nFirst one hot encoded labels vector y_train_mnist\n";
        for (int i = 0; i < 10; i++)
        {
            cout << y_train_mnist(i, 0) << " ";
        }

        cout << "\n\nFirst mnist image number";

        printMnistImage(X_train_mnist, 0);

        Matrix A(3, 2), B(2, 4);

        A.data = { 2, 3, 6, 7, 9, 10 };
        B.data = { 12, 16, 8, 9, 5, 2, 10, 1 };

        cout << "\n\nMatrix struct example\n\n";

        for (int i = 0; i < A.rows; i++)
        {
            for (int j = 0; j < A.cols; j++)
            {
                cout << A(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n";

        for (int i = 0; i < B.rows; i++)
        {
            for (int j = 0; j < B.cols; j++)
            {
                cout << B(i, j) << " ";
            }
            cout << "\n";
        }

        Matrix mul = A.matmul(B);

        cout << "\n\nMatrix multiplication\n\n";

        for (int i = 0; i < mul.rows; i++)
        {
            for (int j = 0; j < mul.cols; j++)
            {
                cout << mul(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n\nMatrix element wise multiplication\n\n";

        Matrix A1(2, 2), B1(2, 2);

        A1.data = { 2, 3, 4, 5 };
        B1.data = A1.data;

        Matrix C1 = A1 * B1;

        for (int i = 0; i < C1.rows; i++)
        {
            for (int j = 0; j < C1.cols; j++)
            {
                cout << C1(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n\nMatrix multiplication with a scalar\n\n";

        A1 = A1 * (-1);

        for (int i = 0; i < A1.rows; i++)
        {
            for (int j = 0; j < A1.cols; j++)
            {
                cout << A1(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n\nMatrix element wise division\n\n";

        Matrix C2 = A1 / B1;

        for (int i = 0; i < C2.rows; i++)
        {
            for (int j = 0; j < C2.cols; j++)
            {
                cout << C2(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n\nMatrix division with scalar\n\n";

        A1 = A1 / 5;

        for (int i = 0; i < A1.rows; i++)
        {
            for (int j = 0; j < A1.cols; j++)
            {
                cout << A1(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n\nMatrix element wise addition\n\n";

        Matrix C3 = A1 + B1;

        for (int i = 0; i < C3.rows; i++)
        {
            for (int j = 0; j < C3.cols; j++)
            {
                cout << C3(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n\nMatrix addition with a scalar\n\n";

        A1 = A1 + 7;

        for (int i = 0; i < A1.rows; i++)
        {
            for (int j = 0; j < A1.cols; j++)
            {
                cout << A1(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n\nMatrix addition with broadcasting\n\n";

        Matrix D(2, 1);
        D.data = {2, 3};

        A1 = A1.broadcastAdd(D);

        for (int i = 0; i < A1.rows; i++)
        {
            for (int j = 0; j < A1.cols; j++)
            {
                cout << A1(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n\nMatrix element wise substraction\n\n";

        Matrix C4 = A1 - B1;

        for (int i = 0; i < C4.rows; i++)
        {
            for (int j = 0; j < C4.cols; j++)
            {
                cout << C4(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n\nMatrix substraction with a scalar\n\n";

        A1 = A1 - 7;

        for (int i = 0; i < A1.rows; i++)
        {
            for (int j = 0; j < A1.cols; j++)
            {
                cout << A1(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n\nMatrix transpose\n\n";

        A1 = A1.T();

        for (int i = 0; i < A1.rows; i++)
        {
            for (int j = 0; j < A1.cols; j++)
            {
                cout << A1(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n\nMatrix operation chaining\n\n";

        Matrix C5 = (A1 * B1) + A1;

        for (int i = 0; i < C5.rows; i++)
        {
            for (int j = 0; j < C5.cols; j++)
            {
                cout << C5(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n\nMatrix axis sumation\n\n";

        Matrix S1 = sum(A, 0); //row sumation (1, 2)
        Matrix S2 = sum(A, 1); //colums sumation (3, 1)
        Matrix S3 = sum(A); //sum all (1, 1)

        for (int i = 0; i < A.rows; i++)
        {
            for (int j = 0; j < A.cols; j++)
            {
                cout << A(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n\n";

        for (int i = 0; i < S1.rows; i++)
        {
            for (int j = 0; j < S1.cols; j++)
            {
                cout << S1(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n\n";

        for (int i = 0; i < S2.rows; i++)
        {
            for (int j = 0; j < S2.cols; j++)
            {
                cout << S2(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n\n";

        for (int i = 0; i < S3.rows; i++)
        {
            for (int j = 0; j < S3.cols; j++)
            {
                cout << S3(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n\nMatrix exponential function\n\n";

    }
    catch (const exception& e)
    {
        cerr << "Error: " << e.what() << "\n";
    }

    return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
