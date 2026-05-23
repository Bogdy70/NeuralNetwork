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

template<typename Backend>
void debug_accuracy_parts(const typename Backend::Mat& Y,
    const typename Backend::Mat& pred)
{
    using Mat = typename Backend::Mat;

    Mat pred_labels = Backend::greaterth(pred, 0.5f);
    Mat correct = Backend::equals(Y, pred_labels);
    Mat correct_sum = Backend::sum(correct, -1);

    Matrix pred_cpu = pred.toCPU();
    Matrix labels_cpu = pred_labels.toCPU();
    Matrix correct_cpu = correct.toCPU();
    Matrix sum_cpu = correct_sum.toCPU();

    cout << "pred shape: (" << pred.getRows() << ", " << pred.getCols() << ")\n";
    cout << "correct shape: (" << correct.getRows() << ", " << correct.getCols() << ")\n";
    cout << "sum shape: (" << correct_sum.getRows() << ", " << correct_sum.getCols() << ")\n";
    cout << "correct sum: " << sum_cpu(0, 0) << "\n";

    cout << "first 10 pred labels: ";
    for (int i = 0; i < min(10, labels_cpu.getCols()); i++)
        cout << labels_cpu(0, i) << " ";
    cout << "\n";

    cout << "first 10 correct: ";
    for (int i = 0; i < min(10, correct_cpu.getCols()); i++)
        cout << correct_cpu(0, i) << " ";
    cout << "\n";
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

        cout << "Truth: " << truth_label << " || Pred: " << pred_label;
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

        cout << "Truth: " << y_test(0, imgIdx) << " || Pred: " << pred_label;
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
        cout << "y_test_mnist: (" << y_test_mnist.getRows() << ", " << y_test_mnist.getCols() << ")\n";

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

        A.setData({ 2, 3, 6, 7, 9, 10 });
        B.setData({ 12, 16, 8, 9, 5, 2, 10, 1 });

        cout << "\n\nMatrix struct example\n\n";

        for (int i = 0; i < A.getRows(); i++)
        {
            for (int j = 0; j < A.getCols(); j++)
            {
                cout << A(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n";

        for (int i = 0; i < B.getRows(); i++)
        {
            for (int j = 0; j < B.getCols(); j++)
            {
                cout << B(i, j) << " ";
            }
            cout << "\n";
        }

        CMatrix d_A = A.toCUDA();
        CMatrix d_B = B.toCUDA();

        Matrix mul = CUDABackend::matmul(d_A, d_B).toCPU();

        cout << "\n\nMatrix multiplication\n\n";

        for (int i = 0; i < mul.getRows(); i++)
        {
            for (int j = 0; j < mul.getCols(); j++)
            {
                cout << mul(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n\nMatrix element wise multiplication\n\n";

        Matrix A1(2, 2), B1(2, 2);

        A1.setData({ 2, 3, 4, 5 });
        B1.setData({ 2, 3, 4, 5 });

        Matrix C1 = CUDABackend::mul(A1.toCUDA(), B1.toCUDA()).toCPU();

        for (int i = 0; i < C1.getRows(); i++)
        {
            for (int j = 0; j < C1.getCols(); j++)
            {
                cout << C1(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n\nMatrix multiplication with a scalar\n\n";

        A1 = CUDABackend::scalarMul(A1.toCUDA(), -1.0f).toCPU();

        for (int i = 0; i < A1.getRows(); i++)
        {
            for (int j = 0; j < A1.getCols(); j++)
            {
                cout << A1(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n\nMatrix element wise division\n\n";

        Matrix C2 = CUDABackend::div(A1.toCUDA(), B1.toCUDA()).toCPU();

        for (int i = 0; i < C2.getRows(); i++)
        {
            for (int j = 0; j < C2.getCols(); j++)
            {
                cout << C2(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n\nMatrix division with scalar\n\n";

        A1 = CUDABackend::divScalar(A1.toCUDA(), 5.0f).toCPU();

        for (int i = 0; i < A1.getRows(); i++)
        {
            for (int j = 0; j < A1.getCols(); j++)
            {
                cout << A1(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n\nMatrix element wise addition\n\n";

        Matrix C3 = CUDABackend::add(A1.toCUDA(), B1.toCUDA()).toCPU();

        for (int i = 0; i < C3.getRows(); i++)
        {
            for (int j = 0; j < C3.getCols(); j++)
            {
                cout << C3(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n\nMatrix addition with a scalar\n\n";

        A1 = CUDABackend::scalarAdd(A1.toCUDA(), 7.0f).toCPU();

        for (int i = 0; i < A1.getRows(); i++)
        {
            for (int j = 0; j < A1.getCols(); j++)
            {
                cout << A1(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n\nMatrix addition with broadcasting\n\n";

        Matrix D(2, 1);
        D.setData({ 2, 3 });

        A1 = CUDABackend::broadcastAdd(A1.toCUDA(), D.toCUDA()).toCPU();

        for (int i = 0; i < A1.getRows(); i++)
        {
            for (int j = 0; j < A1.getCols(); j++)
            {
                cout << A1(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n\nMatrix element wise substraction\n\n";

        Matrix C4 = CUDABackend::sub(A1.toCUDA(), B1.toCUDA()).toCPU();

        for (int i = 0; i < C4.getRows(); i++)
        {
            for (int j = 0; j < C4.getCols(); j++)
            {
                cout << C4(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n\nMatrix substraction with a scalar\n\n";

        A1 = CUDABackend::subScalar(A1.toCUDA(), 7.0f).toCPU();

        for (int i = 0; i < A1.getRows(); i++)
        {
            for (int j = 0; j < A1.getCols(); j++)
            {
                cout << A1(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n\nMatrix transpose\n\n";

        A1 = CUDABackend::T(A1.toCUDA()).toCPU();

        for (int i = 0; i < A1.getRows(); i++)
        {
            for (int j = 0; j < A1.getCols(); j++)
            {
                cout << A1(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n\nMatrix operation chaining\n\n";

        Matrix C5 = (A1 * B1) + A1;

        for (int i = 0; i < C5.getRows(); i++)
        {
            for (int j = 0; j < C5.getCols(); j++)
            {
                cout << C5(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n\nRandom matrix generation\n\n";

        Matrix C6 = CUDABackend::random(2, 3).toCPU();

        for (int i = 0; i < C6.getRows(); i++)
        {
            for (int j = 0; j < C6.getCols(); j++)
            {
                cout << C6(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n\nMatrix axis sumation\n\n";

        Matrix S1 = CMatrix::sum(A.toCUDA(), 0).toCPU(); //row sumation (1, 2)
        Matrix S2 = CMatrix::sum(A.toCUDA(), 1).toCPU(); //colums sumation (3, 1)
        Matrix S3 = CMatrix::sum(A.toCUDA()).toCPU(); //sum all (1, 1)

        for (int i = 0; i < A.getRows(); i++)
        {
            for (int j = 0; j < A.getCols(); j++)
            {
                cout << A(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n\n";

        for (int i = 0; i < S1.getRows(); i++)
        {
            for (int j = 0; j < S1.getCols(); j++)
            {
                cout << S1(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n\n";

        for (int i = 0; i < S2.getRows(); i++)
        {
            for (int j = 0; j < S2.getCols(); j++)
            {
                cout << S2(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n\n";

        for (int i = 0; i < S3.getRows(); i++)
        {
            for (int j = 0; j < S3.getCols(); j++)
            {
                cout << S3(i, j) << " ";
            }
            cout << "\n";
        }

        cout << "\n\nMatrix total max value\n\n";

        float maxt = CMatrix::maxA(A1.toCUDA()).toCPU()(0, 0);
        cout << maxt;

        cout << "\n\nNetwork config\n\n";

        vector<int> dim_list = { X_train_cat.getRows(), 100, 100, 200, y_train_cat.getRows() };

        for (int i = 0; i < size(dim_list); i++)
        {
            cout << dim_list[i] << " ";
        }

        cout <<"\nSize of dimensions list: " << size(dim_list);

        cout << "\n\nParameter initialization test\n\n";

        Parameters<CUDABackend> params = init_params<CUDABackend>(dim_list);

        for (int i = 1; i < size(dim_list); i++)
        {
            cout << "W" << i << " shape: " << "(" << params.W[i].getRows() << ", " << params.W[i].getCols() << ")\n";
            cout << "B" << i << " shape: " << "(" << params.B[i].getRows() << ", " << params.B[i].getCols() << ")\n";
        }

        cout << "\n\nForward pass test\n\n";

        Forward<CUDABackend> frd_cache2 = forward_pass<CUDABackend>(params, X_train_cat.toCUDA(), "relu");

        for (int i = 0; i < size(dim_list); i++)
        {
            cout << "A" << i << " shape: " << "(" << frd_cache2.A[i].getRows() << ", " << frd_cache2.A[i].getCols() << ")\n";
        }

        Matrix pred = frd_cache2.A[size(params.W) - 1].toCPU();
        Matrix pred_labels = pred > 0.5f;

        cout << "\n\nSize of pred: " << "(" << pred_labels.getRows() << ", " << pred_labels.getCols() << ")";
        cout << "\n\nPred label: " << pred_labels(0, 101);
        cout << "\n\nAccuracy: " << accuracy<SequentialBackend>(y_train_cat, pred)*100<<"%";

        cout << "\n\nBackpropagation test\n\n";

        Backward<CUDABackend> grads = backpropagation<CUDABackend>(frd_cache2, params, y_train_cat.toCUDA(), "relu");

        for (int i = size(dim_list) - 1; i > 0; i--)
        {
            cout << "dZ" << i << " shape: " << "(" << grads.dZ[i].getRows() << ", " << grads.dZ[i].getCols() << ")\n";
            cout << "dW" << i << " shape: " << "(" << grads.dW[i].getRows() << ", " << grads.dW[i].getCols() << ")\n";
            cout << "dB" << i << " shape: " << "(" << grads.dB[i].getRows() << ", " << grads.dB[i].getCols() << ")\n\n";
        }

        cout << "\n\nTests\n\n";

        Matrix Pclip(1, 4);
        Pclip.setData({ -1.0f, 0.0f, 0.5f, 2.0f });

        Matrix clipped = CUDABackend::clipM(Pclip.toCUDA(), 1e-8f, 1.0f - 1e-8f).toCPU();

        for (int i = 0; i < 4; i++)
            cout << clipped(0, i) << " ";

        cout << "\n";

        Matrix logTest(1, 2);
        logTest.setData({ 0.5f, 1.0f });

        Matrix logged = CUDABackend::logM(logTest.toCUDA()).toCPU();

        cout << logged(0, 0) << " " << logged(0, 1) << "\n";

        dim_list = { X_train_cat.getRows(), 100, 100, 200, y_train_cat.getRows() };

        Parameters<CUDABackend> params1 = train<CUDABackend>(X_train_cat.toCUDA(), X_test_cat.toCUDA(), y_train_cat.toCUDA(), y_test_cat.toCUDA(), dim_list, "relu", 0.005f, 100, 10);

        cout << "\n\nPrediction test\n\n";

        predict<CUDABackend>(X_test_cat, y_test_cat, params1, "relu", 7);

        int threads = omp_get_max_threads();
        cout << "\n\nMax OpenMP threads: " << threads << "\n";

        Matrix T(2, 3);

        T.setData({ 1, 2, 2, 3, 7, 9 });

        CMatrix d_T = T.toCUDA();
        Matrix T2 = d_T.toCPU();

        cout << "\n\nCUDA copy test: "<<Matrix::sum((T==T2))(0, 0)<<"\n\n";

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        std::cout << "GPU: " << prop.name << "\n";
        std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "Max block dimensions: "
            << prop.maxThreadsDim[0] << " "
            << prop.maxThreadsDim[1] << " "
            << prop.maxThreadsDim[2] << "\n";
        std::cout << "Max grid dimensions: "
            << prop.maxGridSize[0] << " "
            << prop.maxGridSize[1] << " "
            << prop.maxGridSize[2] << "\n";
        std::cout << "Shared memory per block: "
            << prop.sharedMemPerBlock << "\n";
        std::cout << "SM count: "
            << prop.multiProcessorCount << "\n";
    }
    catch (const exception& e)
    {
        cerr << "Error: " << e.what() << "\n";
    }

    return 0;
}