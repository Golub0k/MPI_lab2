#include <iostream>
#include <random>
#include <mpi.h>
#include <string>
#include <fstream>

#define EPS 1e-2

void print_matrix(const float *matrix, int rows, int cols)
{
    std::cout << "[\n";
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "]\n";
}

void clear(float *vector, int size)
{
    for (int i = 0; i < size; ++i)
    {
        vector[i] = 0;
    }
}

void generate(float *matrix, float *vector, int matrix_size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1, 1);

    for (int i = 0; i < matrix_size * matrix_size; ++i)
    {
        matrix[i] = dist(gen);
    }
    for (int i = 0; i < matrix_size; ++i)
    {
        vector[i] = dist(gen);
    }
}

void max_duration(int rank, double &duration)
{
    double maxTime = 0;
    MPI_Reduce(&duration, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        std::cout << "Duration: " << maxTime * 1000 << " ms\n";
        duration = maxTime;
    }
}

void check_correctness(const float *matrix, const float *vector, const float *result, int rows, int cols)
{
    auto *resultRef = new float[rows];
    for (int i = 0; i < rows; ++i)
    {
        resultRef[i] = 0;
        for (int j = 0; j < cols; ++j)
        {
            resultRef[i] += matrix[i * cols + j] * vector[j];
        }
    }
    int fail = 0;
    for (int i = 0; i < rows; ++i)
    {
        if (std::abs(result[i] - resultRef[i]) >= EPS)
        {
            std::cout << result[i] << " != " << resultRef[i] << "\n";
            fail++;
        }
    }

    if (fail == 0)
    {
        std::cout << "Result correct\n";
    }
    else
    {
        std::cout << "Result incorrect :(\n";
    }

    delete[] resultRef;
}

void multiply_by_column(const float *matrix, const float *vector, float *result, size_t numRows, size_t numCols, size_t start_col, size_t end_col)
{
    for (size_t i = start_col; i < end_col; ++i)
    {
        for (size_t j = 0; j < numRows; ++j)
        {
            result[j] += matrix[j * numCols + i] * vector[i];
        }
    }
}
void multiply_by_row(const float *localMatrix, const float *localVector, float *result, size_t local_rows, size_t cols)
{
    for (size_t i = 0; i < local_rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            result[i] += localMatrix[i * cols + j] * localVector[j];
        }
    }
}
void multiply_by_block(const float *matrix, const float *vector, float *result, size_t numCols, size_t startRow, size_t endRow, size_t startCol, size_t endCol)
{
    for (size_t i = startRow; i < endRow; ++i)
    {
        for (size_t j = startCol; j < endCol; ++j)
        {
            result[i] += matrix[i * numCols + j] * vector[j];
        }
    }
}

void print_result(float *matrix, float *vector, float *global_result, int matrix_size, double &duration)
{
    {
        std::cout << "Matrix" << "\n";
        print_matrix(matrix, matrix_size, matrix_size);
        std::cout << "Vector"<< "\n";
        print_matrix(vector, 1, matrix_size);
        std::cout << "Result" << "\n";
        print_matrix(global_result, 1, matrix_size);
    }
}

int main(int argc, char **argv)
{

    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << "-n <num_threads> <matrix_size> <algo_name> " << std::endl;
        return 1;
    }
    int matrix_size = std::stoi(argv[1]);
    std::string algo = std::string(argv[2]);
    MPI_Init(&argc, &argv);

    int num_proc, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    double start_time;
    double end_time;
    double duration = 0;

    if (rank == 0)
    {
        std::cout << "Number of processes: " << num_proc << "\n";
        std::cout << "Matrix size: " << matrix_size << "x" << matrix_size << "\n";
        std::cout << "Algorithm: " << algo << "\n";
    }

    if (algo == "row")
    {

        int local_rows = matrix_size / num_proc;
        int loca_elements = local_rows * matrix_size;

        float *matrix = nullptr;
        auto *vector = new float[matrix_size];
        auto *localMatrix = new float[local_rows * matrix_size];
        auto *local_result = new float[local_rows];
        float *global_result = nullptr;

        for (int i = 0; i < local_rows; ++i)
        {
            local_result[i] = 0;
        }

        if (rank == 0)
        {
            matrix = new float[matrix_size * matrix_size];
            global_result = new float[matrix_size];
            std::fill(global_result, global_result + matrix_size, 0.0f);
            generate(matrix, vector, matrix_size);
        }

        MPI_Scatter(matrix, local_rows * matrix_size, MPI_FLOAT, localMatrix, local_rows * matrix_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(vector, matrix_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

        size_t iters = 100;
        for (size_t i = 0; i < iters; ++i)
        {
            start_time = MPI_Wtime();
            multiply_by_row(localMatrix, vector, local_result, local_rows, matrix_size);
            end_time = MPI_Wtime();
            duration += end_time - start_time;
            //std::cout << "iter: " << i << " duration "<< duration*1000<< " ms\n";
            clear(local_result, local_rows);
        }
        duration /= static_cast<double>(iters);
        //std::cout << " midl duration "<< duration*1000<< " ms\n";

        multiply_by_row(localMatrix, vector, local_result, local_rows, matrix_size);

        MPI_Gather(local_result, local_rows, MPI_FLOAT, global_result, local_rows, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
        if (rank == 0)
        {
            size_t remainRows = matrix_size % num_proc;
            size_t offset = matrix_size - remainRows;
            start_time = MPI_Wtime();
            multiply_by_row(matrix + matrix_size * offset, vector, global_result + offset, remainRows, matrix_size);
            end_time = MPI_Wtime();
            duration += end_time - start_time;
        }
        if (rank == 0)
        {
            check_correctness(matrix, vector, global_result, matrix_size, matrix_size);
            // print_result(matrix, vector, global_result, matrix_size, duration);
        };

        delete[] matrix;
        delete[] vector;
        delete[] local_result;
        delete[] global_result;
    }
    else if (algo == "column")
    {
        auto *matrix = new float[matrix_size * matrix_size];
        auto *vector = new float[matrix_size];
        auto *local_result = new float[matrix_size];
        float *global_result = nullptr;

        for (size_t i = 0; i < matrix_size; ++i)
        {
            local_result[i] = 0;
        }

        if (rank == 0)
        {
            global_result = new float[matrix_size];
            std::fill(global_result, global_result + matrix_size, 0.0f);
            generate(matrix, vector, matrix_size);
        }

        MPI_Bcast(vector, matrix_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(matrix, matrix_size * matrix_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

        size_t cols_per_process = 1;
        size_t start_col = 0;
        size_t end_col = 0;
        if (rank < matrix_size)
        {
            if (num_proc <= matrix_size)
            {
                cols_per_process = matrix_size / num_proc;
            }
            start_col = rank * cols_per_process;
            end_col = start_col + cols_per_process;
            if (rank == num_proc - 1)
            {
                end_col = matrix_size;
            }
        }

        size_t iters = 100;
        for (size_t i = 0; i < iters; ++i)
        {
            if (rank < matrix_size)
            {
                start_time = MPI_Wtime();
                multiply_by_column(matrix, vector, local_result, matrix_size, matrix_size, start_col, end_col);
                end_time = MPI_Wtime();
                duration += end_time - start_time;
                clear(local_result, matrix_size);
            }
        }
        duration /= static_cast<double>(iters);

        if (rank < matrix_size)
        {
            multiply_by_column(matrix, vector, local_result, matrix_size, matrix_size, start_col, end_col);
        }

        MPI_Barrier(MPI_COMM_WORLD);
     
        MPI_Reduce(local_result, global_result, matrix_size, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    
        if (rank == 0)
        {
            check_correctness(matrix, vector, global_result, matrix_size, matrix_size);
            // print_result(matrix, vector, global_result, matrix_size, duration);
        };
        delete[] matrix;
        delete[] vector;
        delete[] local_result;
        delete[] global_result;
    }
    else if (algo == "block")
    {
        auto blocks_per_dimension = static_cast<int>(std::sqrt(num_proc));
        int totalBlocks = blocks_per_dimension * blocks_per_dimension;
        int blockRows = matrix_size / blocks_per_dimension;
        int blockCols = matrix_size / blocks_per_dimension;

        auto *matrix = new float[matrix_size * matrix_size];
        auto *vector = new float[matrix_size];
        auto *local_result = new float[matrix_size];
        float *global_result = nullptr;

        for (int i = 0; i < matrix_size; ++i)
        {
            local_result[i] = 0;
        }

        if (rank == 0)
        {
            global_result = new float[matrix_size];
            std::fill(global_result, global_result + matrix_size, 0.0f);
            generate(matrix, vector, matrix_size);
        }

        MPI_Bcast(vector, matrix_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(matrix, matrix_size * matrix_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

        size_t startRow = (rank / blocks_per_dimension) * blockRows;
        size_t endRow = startRow + blockRows;
        size_t startCol = (rank % blocks_per_dimension) * blockCols;
        size_t endCol = startCol + blockCols;

        if (rank < totalBlocks)
        {
            if (rank == totalBlocks - 1)
            {
                endRow = matrix_size;
                endCol = matrix_size;
            }
            else if ((rank + 1) % blocks_per_dimension == 0)
            {
                endCol = matrix_size;
            }
            else if ((rank + 1) > totalBlocks - blocks_per_dimension)
            {
                endRow = matrix_size;
            }
        }

        size_t iters = 100;
        for (size_t i = 0; i < iters; ++i)
        {
            start_time = MPI_Wtime();
            if (rank < totalBlocks)
            {
                multiply_by_block(matrix, vector, local_result, matrix_size, startRow, endRow, startCol, endCol);
            }
            end_time = MPI_Wtime();
            duration += end_time - start_time;
            clear(local_result, matrix_size);
        }
        duration /= static_cast<double>(iters);
        
        if (rank < totalBlocks)
        {
            multiply_by_block(matrix, vector, local_result, matrix_size, startRow, endRow, startCol, endCol);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Reduce(local_result, global_result, matrix_size, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            check_correctness(matrix, vector, global_result, matrix_size, matrix_size);
            // print_result(matrix, vector, global_result, matrix_size, duration);
        }

        delete[] matrix;
        delete[] vector;
        delete[] local_result;
    }
    else if (rank == 0)
    {
        std::cerr << "Algo name could be either row, column or block" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    max_duration(rank, duration);
    std::ofstream file("time.txt");
    file << duration * 1000;
    file.close();
    MPI_Finalize();

    return 0;
}