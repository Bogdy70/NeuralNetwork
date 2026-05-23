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
#include "Backends.h"
#include <omp.h>

using namespace std;

template <typename Backend>
typename Backend::Mat sigmoid(const typename Backend::Mat& A)
{
    return Backend::scalarDiv(1.0f, Backend::scalarAdd(Backend::expM(Backend::scalarMul(A, -1.0f)), 1.0f));
}

template <typename Backend>
typename Backend::Mat softmax(const typename Backend::Mat& A)
{
    using Mat = typename Backend::Mat;

    Mat exp_A = Backend::expM(A);
    Mat sum_exp = Backend::sum(exp_A, 0);

    return Backend::broadcastDiv(exp_A, sum_exp);
}

template <typename Backend>
typename Backend::Mat der_tanh(const typename Backend::Mat& A)
{
    return Backend::scalarSub(1.0f, Backend::powM(Backend::tanhM(A), 2.0f));
}

template <typename Backend>
float cost(const typename Backend::Mat& Y, const typename Backend::Mat& pred)
{
    int m = Y.getCols();
    float epsilon = 1e-8f;
    typename Backend::Mat clippedPred = Backend::clipM(pred, epsilon, 1.0f - epsilon);

    if (Y.getRows() > 1)
        return (-1.0f / static_cast<float>(m)) * Backend::toScalar(Backend::sum(Backend::mul(Y, Backend::logM(clippedPred))));
    else
        return (-1.0f / static_cast<float>(m)) * Backend::toScalar(Backend::sum(Backend::add(Backend::mul(Y, Backend::logM(clippedPred)), Backend::mul(Backend::scalarSub(1.0f, Y), Backend::logM(Backend::scalarSub(1.0f, clippedPred))))));
}

template <typename Backend>
float accuracy(const typename Backend::Mat& Y, const typename Backend::Mat& pred)
{
    if (Y.getRows() > 1)
    {
        typename Backend::Mat true_labels = Backend::argmax(Y);
        typename Backend::Mat pred_labels = Backend::argmax(pred);

        return Backend::toScalar(Backend::sum(Backend::equals(true_labels, pred_labels))) / static_cast<float>(Y.getCols());
    }
    else
    {
        typename Backend::Mat pred_labels = Backend::greaterth(pred, 0.5f);
        return Backend::toScalar(Backend::sum(Backend::equals(Y, pred_labels))) / static_cast<float>(Y.getCols());
    }
}

template <typename Backend>
struct Parameters
{
    vector<typename Backend::Mat> W;
    vector<typename Backend::Mat> B;

    Parameters(int dim): W(dim), B(dim) {}
};

template <typename Backend>
struct Forward
{
    vector<typename Backend::Mat> Z;
    vector<typename Backend::Mat> A;

    Forward(int dim) : Z(dim), A(dim) {}
};

template <typename Backend>
struct Backward
{
    vector<typename Backend::Mat> dZ;
    vector<typename Backend::Mat> dW;
    vector<typename Backend::Mat> dB;

    Backward(int dim): dZ(dim), dW(dim), dB(dim) {}
};

template<typename Backend>
struct Activation
{
    typename Backend::Mat(*forward)(const typename Backend::Mat&);
    typename Backend::Mat(*derivate)(const typename Backend::Mat&);

    Activation(): forward(nullptr), derivate(nullptr) {}

    Activation(const string& name)
    {
        if (name == "relu")
        {
            forward = Backend::relu;
            derivate = Backend::der_relu;
        }
        else if (name == "tanh")
        {
            forward = Backend::tanhM;
            derivate = der_tanh<Backend>;
        }
        else
            throw runtime_error("Invalid activation type. Please choose 'relu' or 'tanh'");
    }
};

template <typename Backend>
struct NetworkConfig
{
    vector<int> dims;
    Activation<Backend> activation;

    NetworkConfig(const vector<int>& dim_list): dims(dim_list), activation() {}

    NetworkConfig(const vector<int>& dim_list, const string& activ_name) : dims(dim_list), activation(activ_name) {}
};

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

template<typename Backend>
Parameters<Backend> init_params(const vector<int>& dim_list)
{
    int L = size(dim_list);
    Parameters<Backend> params(L);

    for (int l = 1; l < L; l++)
    {
        params.W[l] = Backend::scalarMul(Backend::random(dim_list[l], dim_list[l - 1]), sqrt(2.0f / static_cast<float>(dim_list[l - 1])));
        params.B[l] = typename Backend::Mat(dim_list[l], 1);
    }

    return params;
}

template<typename Backend>
Forward<Backend> forward_pass(const Parameters<Backend>& params, const typename Backend::Mat& X, const string& activation)
{
    int L = size(params.W);
    Forward<Backend> forward_cache(L);
    Activation<Backend> activ(activation);
    typename Backend::Mat(*final_activ)(const typename Backend::Mat&);

    forward_cache.A[0] = Backend::clone(X);

    for (int l = 1; l < L-1; l++)
    {
        forward_cache.Z[l] = Backend::broadcastAdd(Backend::matmul(params.W[l], forward_cache.A[l - 1]), params.B[l]);
        forward_cache.A[l] = activ.forward(forward_cache.Z[l]);
    }
    forward_cache.Z[L-1] = Backend::broadcastAdd(Backend::matmul(params.W[L-1], forward_cache.A[L - 2]), params.B[L-1]);
    final_activ = forward_cache.Z[L-1].getRows() > 1 ? softmax<Backend> : sigmoid<Backend>;
    forward_cache.A[L-1] = final_activ(forward_cache.Z[L-1]);

    return forward_cache;
}

template<typename Backend>
Backward<Backend> backpropagation(const Forward<Backend>& frd_cache, const Parameters<Backend>& params, const typename Backend::Mat& Y, const string& activation)
{
    int L = size(frd_cache.A);
    int m = Y.getCols();
    Backward<Backend> grads(L);
    Activation<Backend> activ(activation);

    grads.dZ[L - 1] = Backend::sub(frd_cache.A[L - 1], Y);
    grads.dW[L - 1] =  Backend::scalarMul(Backend::matmul(grads.dZ[L - 1], Backend::T(frd_cache.A[L - 2])), (1.0f / static_cast<float>(m)));
    grads.dB[L - 1] =  Backend::scalarMul(Backend::sum(grads.dZ[L - 1], 1), (1.0f / static_cast<float>(m)));

    for (int l = L - 2; l > 0; l--)
    {
        grads.dZ[l] = Backend::mul(Backend::matmul(Backend::T(params.W[l + 1]), grads.dZ[l + 1]), activ.derivate(frd_cache.Z[l]));
        grads.dW[l] = Backend::scalarMul(Backend::matmul(grads.dZ[l], Backend::T(frd_cache.A[l - 1])), (1.0f / static_cast<float>(m)));
        grads.dB[l] = Backend::scalarMul(Backend::sum(grads.dZ[l], 1), (1.0f / static_cast<float>(m)));
    }

    return grads;
}

template<typename Backend>
Parameters<Backend>& optimizer(Parameters<Backend>& params, const Backward<Backend>& grads, const float lr)
{
    int L = size(params.W);
    for (int l = 1; l < L; l++)
    {
        params.W[l] = Backend::sub(params.W[l], Backend::scalarMul(grads.dW[l], lr));
        params.B[l] = Backend::sub(params.B[l], Backend::scalarMul(grads.dB[l], lr));
    }

    return params;
}

template<typename Backend>
Parameters<Backend> train(const typename Backend::Mat& X_train,
    const typename Backend::Mat& X_test,
    const typename Backend::Mat& y_train,
    const typename Backend::Mat& y_test,
    const vector<int>& dim_list,
    const string& activation,
    const float lr,
    const int epochs,
    const int viewing_rate)
{
    Parameters<Backend> params = init_params<Backend>(dim_list);

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        Forward<Backend> frd_cache = forward_pass<Backend>(params, X_train, activation);
        float train_cost = cost<Backend>(y_train, frd_cache.A[size(dim_list) - 1]);
        float train_acc = accuracy<Backend>(y_train, frd_cache.A[size(dim_list) - 1]);
        Backward<Backend> grads = backpropagation<Backend>(frd_cache, params, y_train, activation);
        optimizer<Backend>(params, grads, lr);

        if (epoch % viewing_rate == 0)
        {
            frd_cache = forward_pass<Backend>(params, X_test, activation);
            float test_cost = cost<Backend>(y_test, frd_cache.A[size(dim_list) - 1]);
            float test_acc = accuracy<Backend>(y_test, frd_cache.A[size(dim_list) - 1]);
            cout << "Epoch: " << epoch << " || Train loss: " << train_cost << " || Test loss: " << test_cost << " || Train accuracy: " << train_acc * 100.0f << "% || Test accuracy: " << test_acc * 100.0f << "%\n";
        }
    }

    return params;
}

template <typename Backend>
void predict(const Matrix& X_test, const Matrix& y_test, const Parameters<Backend>& params, const string& activation, int imgIdx)
{
    Matrix one_mnist(X_test.getRows(), 1);

    for (int i = 0; i < X_test.getRows(); i++)
    {
        one_mnist(i, 0) = X_test(i, imgIdx);
    }

    Matrix one_mnisty(y_test.getRows(), 1);

    for (int i = 0; i < y_test.getRows(); i++)
    {
        one_mnisty(i, 0) = y_test(i, imgIdx);
    }

    int L = static_cast<int>(params.W.size());

    int pred_label = 0;

    if (y_test.getRows() > 1)
    {
        if constexpr (std::is_same_v<Backend, CUDABackend>)
        {
            Forward<Backend> frd_cache1 = forward_pass<Backend>(params, one_mnist.toCUDA(), activation);

            pred_label = Backend::toScalar(Backend::argmax(frd_cache1.A[L - 1]));
        }
        else
        {
            Forward<Backend> frd_cache1 = forward_pass<Backend>(params, one_mnist, activation);

            pred_label = Backend::toScalar(Backend::argmax(frd_cache1.A[L - 1]));
        }

        int truth_label = Matrix::argmax(one_mnisty)(0, 0);

        cout << "\nTruth: " << truth_label << " || Pred: " << pred_label;

        printMnistImage(X_test, imgIdx);
    }
    else
    {
        if constexpr (std::is_same_v<Backend, CUDABackend>)
        {
            Forward<Backend> frd_cache1 = forward_pass<Backend>(params, one_mnist.toCUDA(), activation);

            pred_label = Backend::greaterth(frd_cache1.A[L - 1], 0.5f).toCPU()(0, imgIdx);
        }
        else
        {
            Forward<Backend> frd_cache1 = forward_pass<Backend>(params, one_mnist, activation);

            pred_label = Backend::greaterth(frd_cache1.A[L - 1], 0.5f)(0, imgIdx);
        }

        cout << "\nTruth: " << y_test(0, imgIdx) << " || Pred: " << pred_label;
    }
}

int main()
{
    try
    {
        Matrix X_train_cat = Matrix::loadMatrixBin("data/cat/X_train.bin", 12288, 209);
        Matrix y_train_cat = Matrix::loadMatrixBin("data/cat/Y_train.bin", 1, 209);

        Matrix X_test_cat = Matrix::loadMatrixBin("data/cat/X_test.bin", 12288, 50);
        Matrix y_test_cat = Matrix::loadMatrixBin("data/cat/Y_test.bin", 1, 50);

        Matrix X_train_mnist = Matrix::loadMatrixBin("data/mnist/X_train.bin", 784, 5000);
        Matrix y_train_mnist = Matrix::loadMatrixBin("data/mnist/Y_train.bin", 10, 5000);

        Matrix X_test_mnist = Matrix::loadMatrixBin("data/mnist/X_test.bin", 784, 1000);
        Matrix y_test_mnist = Matrix::loadMatrixBin("data/mnist/Y_test.bin", 10, 1000);

        cout << "Cat dataset loaded successfully\n";

        cout << "X_train_cat: (" << X_train_cat.getRows() << ", " << X_train_cat.getCols() << ")\n";
        cout << "y_train_cat: (" << y_train_cat.getRows() << ", " << y_train_cat.getCols() << ")\n";
        
        cout << "X_test_cat: (" << X_test_cat.getRows() << ", " << X_test_cat.getCols() << ")\n";
        cout << "y_test_cat: (" << y_test_cat.getRows() << ", " << y_test_cat.getCols() << ")\n";

        cout << "\nMnist dataset loaded successfully\n";

        cout << "X_train_mnist: (" << X_train_mnist.getRows() << ", " << X_train_mnist.getCols() << ")\n";
        cout << "y_train_mnist: (" << y_train_mnist.getRows() << ", " << y_train_mnist.getCols() << ")\n";

        cout << "X_test_mnist: (" << X_test_mnist.getRows() << ", " << X_test_mnist.getCols() << ")\n";
        cout << "y_test_mnist: (" << y_test_mnist.getRows() << ", " << y_test_mnist.getCols() << ")\n\n";



        cout << "Sequential cat dataset test\n\n";

        vector<int> dim_list = { X_train_cat.getRows(), 100, 100, 200, y_train_cat.getRows() };

        auto start = std::chrono::high_resolution_clock::now();

        Parameters<SequentialBackend> params1 = train<SequentialBackend>(X_train_cat, X_test_cat, y_train_cat, y_test_cat, dim_list, "tanh", 0.005f, 100, 10);

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;

        cout << "\nSequential cat training time: " << elapsed.count() << " seconds\n";

        predict<SequentialBackend>(X_test_cat, y_test_cat, params1, "tanh", 7);

        

        cout << "\n\nOpenMP cat dataset test\n\n";

        start = std::chrono::high_resolution_clock::now();

        Parameters<OpenMPBackend> params2 = train<OpenMPBackend>(X_train_cat, X_test_cat, y_train_cat, y_test_cat, dim_list, "tanh", 0.005f, 300, 50);

        end = std::chrono::high_resolution_clock::now();

        elapsed = end - start;

        cout << "\nOpenMP cat training time: " << elapsed.count() << " seconds\n";

        predict<OpenMPBackend>(X_test_cat, y_test_cat, params2, "tanh", 7);



        cout << "\n\nCUDA cat dataset test\n\n";

        start = std::chrono::high_resolution_clock::now();

        Parameters<CUDABackend> params3 = train<CUDABackend>(X_train_cat.toCUDA(), X_test_cat.toCUDA(), y_train_cat.toCUDA(), y_test_cat.toCUDA(), dim_list, "tanh", 0.005f, 700, 100);

        cudaDeviceSynchronize();

        end = std::chrono::high_resolution_clock::now();

        elapsed = end - start;

        cout << "\nCUDA cat training time: " << elapsed.count() << " seconds\n";

        predict<CUDABackend>(X_test_cat, y_test_cat, params3, "tanh", 7);



        cout << "\n\nSequential mnist dataset test\n\n";

        dim_list = { X_train_mnist.getRows(), 100, 100, 200, y_train_mnist.getRows() };

        start = std::chrono::high_resolution_clock::now();

        Parameters<SequentialBackend> params4 = train<SequentialBackend>(X_train_mnist, X_test_mnist, y_train_mnist, y_test_mnist, dim_list, "relu", 0.01f, 100, 10);

        end = std::chrono::high_resolution_clock::now();

        elapsed = end - start;

        cout << "\nSequential mnist training time: " << elapsed.count() << " seconds\n";

        predict<SequentialBackend>(X_test_mnist, y_test_mnist, params4, "relu", 7);



        cout << "\n\nOpenMP mnist dataset test\n\n";

        start = std::chrono::high_resolution_clock::now();

        Parameters<OpenMPBackend> params5 = train<OpenMPBackend>(X_train_mnist, X_test_mnist, y_train_mnist, y_test_mnist, dim_list, "relu", 0.01f, 300, 50);

        end = std::chrono::high_resolution_clock::now();

        elapsed = end - start;

        cout << "\nOpenMP mnist training time: " << elapsed.count() << " seconds\n";
        
        predict<OpenMPBackend>(X_test_mnist, y_test_mnist, params5, "relu", 7);



        cout << "\n\nCUDA mnist dataset test\n\n";

        start = std::chrono::high_resolution_clock::now();

        Parameters<CUDABackend> params6 = train<CUDABackend>(X_train_mnist.toCUDA(), X_test_mnist.toCUDA(), y_train_mnist.toCUDA(), y_test_mnist.toCUDA(), dim_list, "relu", 0.01f, 700, 100);

        cudaDeviceSynchronize();

        end = std::chrono::high_resolution_clock::now();

        elapsed = end - start;

        cout << "\nCUDA mnist training time: " << elapsed.count() << " seconds\n";
        
        predict<CUDABackend>(X_test_mnist, y_test_mnist, params6, "relu", 7);
    }
    catch (const exception& e)
    {
        cerr << "Error: " << e.what() << "\n";
    }

    return 0;
}