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

    Parameters(int dim): W(dim), B(dim) {}
};

struct Forward
{
    vector<Matrix> Z;
    vector<Matrix> A;

    Forward(int dim) : Z(dim), A(dim) {}
};

struct Backward
{
    vector<Matrix> dZ;
    vector<Matrix> dW;
    vector<Matrix> dB;

    Backward(int dim): dZ(dim), dW(dim), dB(dim) {}
};

struct Activation
{
    Matrix(*forward)(const Matrix&);
    Matrix(*derivate)(const Matrix&);

    Activation(): forward(), derivate() {}

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

    NetworkConfig(const vector<int>& dim_list): dims(dim_list), activation() {}

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

Parameters init_params(const vector<int>& dim_list)
{
    int L = size(dim_list);
    Parameters params(L);

    for (int l = 1; l < L; l++)
    {
        params.W[l] = random(dim_list[l], dim_list[l - 1]) * sqrt(2.0f / static_cast<float>(dim_list[l - 1]));
        params.B[l] = Matrix(dim_list[l], 1);
    }

    return params;
}

Forward forward_pass(const Parameters& params, const Matrix& X, const string& activation)
{
    int L = size(params.W);
    Forward forward_cache(L);
    Activation activ(activation);
    Matrix(*final_activ)(const Matrix&);

    forward_cache.A[0] = X;

    for (int l = 1; l < L-1; l++)
    {
        forward_cache.Z[l] = params.W[l].matmul(forward_cache.A[l - 1]).broadcastAdd(params.B[l]);
        forward_cache.A[l] = activ.forward(forward_cache.Z[l]);
    }
    forward_cache.Z[L-1] = params.W[L-1].matmul(forward_cache.A[L - 2]).broadcastAdd(params.B[L-1]);
    final_activ = forward_cache.Z[L-1].rows > 1 ? softmax : sigmoid;
    forward_cache.A[L-1] = final_activ(forward_cache.Z[L-1]);

    return forward_cache;
}

Backward backpropagation(const Forward& frd_cache, const Parameters& params, const Matrix& Y, const string& activation)
{
    int L = size(frd_cache.A);
    int m = Y.cols;
    Backward grads(L);
    Activation activ(activation);

    grads.dZ[L - 1] = frd_cache.A[L - 1] - Y;
    grads.dW[L - 1] = (1.0f / static_cast<float>(m)) * (grads.dZ[L - 1].matmul(frd_cache.A[L - 2].T()));
    grads.dB[L - 1] = (1.0f / static_cast<float>(m)) * (sum(grads.dZ[L - 1], 1));

    for (int l = L - 2; l > 0; l--)
    {
        grads.dZ[l] = (params.W[l + 1].T().matmul(grads.dZ[l + 1])) * activ.derivate(frd_cache.Z[l]);
        grads.dW[l] = (1.0f / static_cast<float>(m)) * (grads.dZ[l].matmul(frd_cache.A[l - 1].T()));
        grads.dB[l] = (1.0f / static_cast<float>(m)) * (sum(grads.dZ[l], 1));
    }

    return grads;
}

Parameters& optimizer(Parameters& params, const Backward& grads, const float lr)
{
    int L = size(params.W);
    for (int l = 1; l < L; l++)
    {
        params.W[l] = params.W[l] - lr * grads.dW[l];
        params.B[l] = params.B[l] - lr * grads.dB[l];
    }

    return params;
}

Parameters train(const Matrix& X_train,
    const Matrix& X_test,
    const Matrix& y_train,
    const Matrix& y_test,
    const vector<int>& dim_list,
    const string& activation,
    const float lr,
    const int epochs)
{
    Parameters params = init_params(dim_list);

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        Forward frd_cache = forward_pass(params, X_train, activation);
        float train_cost = cost(y_train, frd_cache.A[size(dim_list) - 1]);
        float train_acc = accuracy(y_train, frd_cache.A[size(dim_list) - 1]);
        Backward grads = backpropagation(frd_cache, params, y_train, activation);
        optimizer(params, grads, lr);

        if (epoch % 10 == 0)
        {
            frd_cache = forward_pass(params, X_test, activation);
            float test_cost = cost(y_test, frd_cache.A[size(dim_list) - 1]);
            float test_acc = accuracy(y_test, frd_cache.A[size(dim_list) - 1]);
            cout << "Epoch: " << epoch << " || Train loss: " << train_cost << " || Test loss: " << test_cost << " || Train accuracy: " << train_acc * 100.0f << "% || Test accuracy: " << test_acc * 100.0f << "%\n";
        }
    }

    return params;
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

        cout << "\n\nNetwork config\n\n";

        vector<int> dim_list = { X_train_cat.rows, 100, 100, 200, y_train_cat.rows };

        for (int i = 0; i < size(dim_list); i++)
        {
            cout << dim_list[i] << " ";
        }

        cout <<"\nSize of dimensions list: " << size(dim_list);

        cout << "\n\nParameter initialization test\n\n";

        Parameters params = init_params(dim_list);

        for (int i = 1; i < size(dim_list); i++)
        {
            cout << "W" << i << " shape: " << "(" << params.W[i].rows << ", " << params.W[i].cols << ")\n";
            cout << "B" << i << " shape: " << "(" << params.B[i].rows << ", " << params.B[i].cols << ")\n";
        }

        cout << "\n\nForward pass test\n\n";

        Forward frd_cache = forward_pass(params, X_train_cat, "relu");

        for (int i = 0; i < size(dim_list); i++)
        {
            cout << "A" << i << " shape: " << "(" << frd_cache.A[i].rows << ", " << frd_cache.A[i].cols << ")\n";
        }

        Matrix pred = frd_cache.A[size(params.W) - 1];
        Matrix pred_labels = pred > 0.5f;

        cout << "\n\nSize of pred: " << "(" << pred_labels.rows << ", " << pred_labels.cols << ")";
        cout << "\n\nPred label: " << pred_labels(0, 101);
        cout << "\n\nAccuracy: " << accuracy(y_train_cat, pred)*100<<"%";

        cout << "\n\nBackpropagation test\n\n";

        Backward grads = backpropagation(frd_cache, params, y_train_cat, "relu");

        for (int i = size(dim_list) - 1; i > 0; i--)
        {
            cout << "dZ" << i << " shape: " << "(" << grads.dZ[i].rows << ", " << grads.dZ[i].cols << ")\n";
            cout << "dW" << i << " shape: " << "(" << grads.dW[i].rows << ", " << grads.dW[i].cols << ")\n";
            cout << "dB" << i << " shape: " << "(" << grads.dB[i].rows << ", " << grads.dB[i].cols << ")\n\n";
        }

        cout << "\n\nPutting it all together and training\n\n";

        Parameters params1 = train(X_train_cat, X_test_cat, y_train_cat, y_test_cat, dim_list, "relu", 0.005f, 100);

        cout << "\n\nPrediction test\n\n";

        Matrix one_cat(X_test_cat.rows, 1);

        for (int i = 0; i < X_test_cat.rows; i++)
        {
            one_cat(i, 0) = X_test_cat(i, 0);
        }

        frd_cache = forward_pass(params1, one_cat, "relu");

        pred = frd_cache.A[size(dim_list) - 1];
        float pred_label = pred(0, 7) > 0.5 ? 1 : 0;

        cout << "Truth: " <<y_test_cat(0, 7)<< " || Pred: " <<pred_label;
    }
    catch (const exception& e)
    {
        cerr << "Error: " << e.what() << "\n";
    }

    return 0;
}