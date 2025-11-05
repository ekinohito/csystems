#include <stdio.h>
#include <mpi.h>
#include "fun.h"

int main(int argc, char **argv)
{
    int rank;
    float A[N][N], B[N][N], C[N][N], D[N][N];
    float T0[N][N], T1[N][N], T2[N][N];
    MPI_Status status;

    inits(A); // Инициализация матриц
    inits(B); // Инициализация матриц
    inits(C); // Инициализация матриц
    inits(D); // Инициализация матриц

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        mul(A, B, T0);
        MPI_Send(T0, N * N, MPI_FLOAT, 1, 1, MPI_COMM_WORLD);
        add(T0, C, T1);
        mul(T1, T1, T2);
        // Принимаем данные в неиспользуемый массив D
        MPI_Recv(D, N * N, MPI_FLOAT, 2, 1, MPI_COMM_WORLD, &status);
        sub(T2, D, T1); // Пусть T1 будет за Y1
        // Принимаем в T2 и T0 значения Y2 и Y3
        MPI_Recv(T2, N * N, MPI_FLOAT, 1, 2, MPI_COMM_WORLD, &status);
        MPI_Recv(T0, N * N, MPI_FLOAT, 2, 2, MPI_COMM_WORLD, &status);
    }

    if (rank == 1)
    {
        add(C, D, T0);
        sub(T0, B, T1);
        // Прием данных в уже «ненужный» массив C
        MPI_Recv(C, N * N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        mul(T1, C, T2);
        // Прием данных в уже «ненужный» массив D
        MPI_Recv(D, N * N, MPI_FLOAT, 2, 1, MPI_COMM_WORLD, &status);
        divv(T2, D, T0); // Пусть T0 будет за Y2
        // Посылаем Y2 нулевому процессу
        MPI_Send(T0, N * N, MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
    }

    if (rank == 2)
    {
        add(C, D, T0);
        add(T0, A, T1);
        MPI_Send(T1, N * N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(T1, N * N, MPI_FLOAT, 1, 1, MPI_COMM_WORLD);
        divv(T1, D, T2); // Пусть T2 будет за Y3
        // Посылаем Y3 нулевому процессу
        MPI_Send(T2, N * N, MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
