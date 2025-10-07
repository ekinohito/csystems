#include <stdio.h>
#include <mpi.h>

#define N 256  // Размер массивов

int main(int argc, char** argv)
{
    static float A[N], B[N], C[N], Y[N];
    float S1 = 2.0; // Скаляр
    int rank, size, cycle, i;
    MPI_Status status;
    double time1, time2, time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //printf("Hello from process %d of %d\n", rank, size);

    // --- Если процесс нулевой ---
    if (rank == 0)
    {
        // Инициализация массивов
        for (i = 0; i < N; i++) {
            A[i] = i + 1;
            B[i] = i + 2;
            C[i] = i + 3;
        }

        time1 = MPI_Wtime(); // начало замера времени

        for (cycle = 0; cycle < 10000; cycle++)  // Цикл кратности
        {
            // Рассылка частей массива B другим процессам
            for (int p = 1; p < size; p++) {
                //MPI_Send(&A[p * N / size], N / size, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
                MPI_Send(&B[p * N / size], N / size, MPI_FLOAT, p, 1, MPI_COMM_WORLD);
                //MPI_Send(&C[p * N / size], N / size, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
            }

            // Вычисления для нулевого процесса
            for (i = 0; i < N / size; i++)
                Y[i] = A[i] * S1 + C[i] / (A[i] + B[i]);

            // Приём результатов от остальных процессов
            for (int p = 1; p < size; p++)
                MPI_Recv(&Y[p * N / size], N / size, MPI_FLOAT, p, 3, MPI_COMM_WORLD, &status);
        }

        time2 = MPI_Wtime(); // конец замера
        time = time2 - time1;

        // Вывод результатов
        /*for (i = 0; i < N; i++)
            printf("Y[%d] = %f\n", i, Y[i]);*/
        printf("TIME = %f sec\n", time);
    }
    // --- Если процесс не нулевой ---
    else
    {
        // Инициализация массивов
        for (i = 0; i < N; i++) {
            A[i] = i + 1;
            //B[i] = i + 2;
            C[i] = i + 3;
        }

        for (cycle = 0; cycle < 10000; cycle++)
        {
            // Получение частей массива
            //MPI_Recv(&A[rank * N / size], N / size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&B[rank * N / size], N / size, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
            //MPI_Recv(&C[rank * N / size], N / size, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, &status);

            // Вычисление своей части
            for (i = rank * N / size; i < (rank + 1) * N / size; i++)
                Y[i] = A[i] * S1 + C[i] / (A[i] + B[i]);

            // Отправка результата обратно нулевому процессу
            MPI_Send(&Y[rank * N / size], N / size, MPI_FLOAT, 0, 3, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
