#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <mpi.h>

// Класс Matrix представляет матрицу и методы для работы с ней

class Matrix
{
public:
    // Конструктор класса, инициализирующий матрицу с заданным количеством строк и столбцов
    Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols), data_(rows * cols) {}

    // Методы доступа к размерам матрицы и ее данным
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    double *data()
    {
        return data_.data();
    }

    const double *data() const
    {
        return data_.data();
    }

    // Перегруженные операторы () для удобства доступа к элементам матрицы
    double &operator()(size_t row, size_t col) { return data_[row * cols_ + col]; }
    const double &operator()(size_t row, size_t col) const { return data_[row * cols_ + col]; }

    // Метод для заполнения матрицы случайными значениями в указанном диапазоне
    void randomize(double lower_bound = -1, double upper_bound = 1)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(lower_bound, upper_bound);

        for (size_t i = 0; i < rows_; ++i)
        {
            for (size_t j = 0; j < cols_; ++j)
            {
                (*this)(i, j) = dis(gen);
            }
        }
    }

    // Метод для вывода матрицы
    void print() const
    {
        for (size_t i = 0; i < rows(); ++i)
        {
            for (size_t j = 0; j < cols(); ++j)
            {
                std::cout << std::setw(10) << std::setprecision(4) << (*this)(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }

    // Метод для создания подматрицы (среза) из текущей матрицы.
    Matrix slice(size_t row_start, size_t col_start, size_t row_end, size_t col_end) const
    {
        if (row_start >= row_end || col_start >= col_end || row_end > rows() || col_end > cols())
        {
            throw std::out_of_range("Invalid slice range");
        }

        size_t new_rows = row_end - row_start;
        size_t new_cols = col_end - col_start;
        Matrix new_matrix(new_rows, new_cols);

        for (size_t i = 0; i < new_rows; ++i)
        {
            for (size_t j = 0; j < new_cols; ++j)
            {
                new_matrix(i, j) = (*this)(row_start + i, col_start + j);
            }
        }

        return new_matrix;
    }

    // Метод для получения общего размера матрицы (количество элементов)
    size_t size() const
    {
        return cols_ * rows_;
    }

    // Перегруженный оператор вывода для вывода матрицы в стандартный поток вывода
    friend std::ostream &operator<<(std::ostream &os, const Matrix &matrix)
    {
        os << "[" << std::endl;
        for (int i = 0; i < matrix.rows(); ++i)
        {
            for (int j = 0; j < matrix.cols(); ++j)
            {
                os << std::fixed << std::setw(5) << matrix(i, j) << " ";
            }
            os << std::endl;
        }
        os << "]" << std::endl;
        return os;
    }

private:
    size_t rows_;
    size_t cols_;
    std::vector<double> data_;
};

// Функция для умножения матриц с использованием алгоритма разбиения по блокам
Matrix matrix_multiply_cannon(const Matrix &A, const Matrix &B, int rank, int size)
{
    const int N = A.rows();
    const int block_size = N / size;
    const int remainder = N % size;

    // Буфер для локального блока матрицы A.
    std::vector<double> A_local((block_size + (rank < remainder ? 1 : 0)) * N);

    // Рассчитать смещения и количества элементов для операций scatterv и gatherv.
    std::vector<int> displacements(size, 0);
    std::vector<int> counts(size, 0);

    for (int i = 0; i < size; ++i)
    {
        displacements[i] = i * block_size * N + std::min(i, remainder) * N;
        counts[i] = block_size * N + (i < remainder ? N : 0);
    }

    // Разослать матрицу A по всем процессам.
    MPI_Scatterv(A.data(), counts.data(), displacements.data(), MPI_DOUBLE, A_local.data(), A_local.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Передать матрицу B всем процессам.
    MPI_Bcast(const_cast<double *>(B.data()), B.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Вычислить локальный блок матрицы C.
    Matrix Result_local = Matrix(block_size + (rank < remainder ? 1 : 0), B.cols());
    for (int i = 0; i < Result_local.rows(); ++i)
    {
        for (int j = 0; j < B.cols(); ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < N; ++k)
            {
                sum += A_local[i * N + k] * B(k, j);
            }
            Result_local(i, j) = sum;
        }
    }

    // Собрать все локальные блоки матрицы C на главном процессе.
    Matrix Result = Matrix(N, B.cols());

    // Рассчитать смещения и количества элементов для операции gatherv.
    for (int i = 0; i < size; ++i)
    {
        displacements[i] = i * block_size * B.cols() + std::min(i, remainder) * B.cols();
        counts[i] = block_size * B.cols() + (i < remainder ? B.cols() : 0);
    }

    MPI_Gatherv(Result_local.data(), Result_local.size(), MPI_DOUBLE, Result.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return Result;
}

std::string check_correctness(const Matrix &A, const Matrix &B, const Matrix &Result)
{
    const double EPSILON = 1e-6;

    if (A.cols() != B.rows() || A.rows() != Result.rows() || B.cols() != Result.cols())
    {
        return "Result incorrect";
    }

    Matrix ref_Result(A.rows(), B.cols());
    for (size_t i = 0; i < A.rows(); ++i)
    {
        for (size_t j = 0; j < B.cols(); ++j)
        {
            double sum = 0;
            for (size_t k = 0; k < A.cols(); ++k)
            {
                sum += A(i, k) * B(k, j);
            }
            ref_Result(i, j) = sum;
        }
    }

    for (size_t i = 0; i < Result.rows(); ++i)
    {
        for (size_t j = 0; j < Result.cols(); ++j)
        {
            if (std::abs(Result(i, j) - ref_Result(i, j)) > EPSILON)
            {
                return "Result incorrect";
            }
        }
    }

    return "Result correct";
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: -n <num threads> " << argv[0] << "<matrix size>" << std::endl;
        return 1;
    }

    size_t matrix_size = std::stoi(argv[1]);

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    Matrix A(matrix_size, matrix_size);
    Matrix B(matrix_size, matrix_size);
    Matrix Result(matrix_size, matrix_size);

    A.randomize();
    B.randomize();

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::steady_clock::now();

    Result = matrix_multiply_cannon(A, B, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    long long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_seconds).count();
    if (rank == 0)
    {
        // std::cout << "Matrix A\n" << A;
        // std::cout << "Matrix B\n" << B;
        // std::cout << "Result\n" << Result;
        std::cout << check_correctness(A, B, Result) << " " << elapsed_ms << "ms" << std::endl;
    }

    if (elapsed_ms != -1)
    {
        std::ofstream file("task2/time.txt");
        file << elapsed_ms;
        file.close();
    }
    MPI_Finalize();
    return 0;
}