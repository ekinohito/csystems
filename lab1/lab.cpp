#include <stdio.h>
#include <mpi.h>
#include "fun.h"

#define TAG_B_MINUS_E 1
#define TAG_E_MINUS_C 2
#define TAG_Y_1 3
#define TAG_Y_2 4
#define TAG_Y_3 5
#define TAG_Y_4 6

#define RANK_0 0 // Вычисляет Y2
#define RANK_1 1 // Вычисляет Y1
#define RANK_2 2 // Вычисляет Y3
#define RANK_3 3 // Вычисляет Y4

// #define DEBUG

#ifdef DEBUG
#include <stdio.h>
#define dbg(fmt, ...)               \
    do                              \
    {                               \
        printf(fmt, ##__VA_ARGS__); \
        printf("\n");               \
        fflush(stdout);             \
    } while (0)
#else
#define dbg(fmt, ...) \
    do                \
    {                 \
    } while (0)
#endif

int main(int argc, char **argv)
{
    int rank;
    static float A[N][N], B[N][N], C[N][N], D[N][N], E[N][N];
    // static float T0[N][N], T1[N][N], T2[N][N];
    MPI_Status status;

    inits(A); // Инициализация матриц
    inits(B); // Инициализация матриц
    inits(C); // Инициализация матриц
    inits(D); // Инициализация матриц
    inits(E); // Инициализация матриц

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double start = MPI_Wtime();

    if (rank == RANK_0)
    {

        sub(B, E, A);
        MPI_Send(A, N * N, MPI_FLOAT, RANK_3, TAG_B_MINUS_E, MPI_COMM_WORLD);
        dbg("r0: B - E -> r3");
        MPI_Recv(B, N * N, MPI_FLOAT, RANK_2, TAG_E_MINUS_C, MPI_COMM_WORLD, &status); // B = E - C (Receive)
        dbg("r0: r2 -> E - C");
        mul(A, B, C);
        MPI_Recv(A, N * N, MPI_FLOAT, RANK_3, TAG_Y_4, MPI_COMM_WORLD, &status); // A = Y4 (Receive)
        dbg("r0: r3 -> Y4");
        add(C, A, B);                                                            // B = Y2
        MPI_Recv(C, N * N, MPI_FLOAT, RANK_2, TAG_Y_3, MPI_COMM_WORLD, &status); // C = Y3 (Receive)
        dbg("r0: r2 -> Y3");
        MPI_Recv(D, N * N, MPI_FLOAT, RANK_1, TAG_Y_1, MPI_COMM_WORLD, &status); // D = Y1 (Receive)
        dbg("r0: r1 -> Y1");
        dbg("r0: DONE");
    }
    else if (rank == RANK_1)
    {
        mul(B, D, C);
        MPI_Recv(B, N * N, MPI_FLOAT, RANK_2, TAG_E_MINUS_C, MPI_COMM_WORLD, &status); // B = E - C (Receive)
        dbg("r1: r2 -> E - C");
        divv(C, B, D);
        sub(D, A, B);
        MPI_Recv(C, N * N, MPI_FLOAT, RANK_3, TAG_Y_4, MPI_COMM_WORLD, &status); // C = Y4 (Receive)
        dbg("r1: r3 -> Y4");
        add(B, C, A);
        MPI_Send(A, N * N, MPI_FLOAT, RANK_0, TAG_Y_1, MPI_COMM_WORLD);
        dbg("r1: Y1 -> r0");
    }
    else if (rank == RANK_2)
    {
        sub(E, C, D);
        MPI_Send(D, N * N, MPI_FLOAT, RANK_0, TAG_E_MINUS_C, MPI_COMM_WORLD);
        dbg("r2: E - C -> r0");
        MPI_Send(D, N * N, MPI_FLOAT, RANK_1, TAG_E_MINUS_C, MPI_COMM_WORLD);
        dbg("r2: E - C -> r1");
        sub(B, C, E);
        mul(A, E, B);
        MPI_Send(B, N * N, MPI_FLOAT, RANK_0, TAG_Y_3, MPI_COMM_WORLD);
        dbg("r2: Y3 -> r0");
    }
    else if (rank == RANK_3)
    {
        mul(C, C, A);
        MPI_Recv(B, N * N, MPI_FLOAT, RANK_0, TAG_B_MINUS_E, MPI_COMM_WORLD, &status); // B = B - E (Receive)
        dbg("r3: r0 -> B - E");
        divv(B, A, D);
        MPI_Send(D, N * N, MPI_FLOAT, RANK_0, TAG_Y_4, MPI_COMM_WORLD);
        dbg("r3: Y4 -> r0");
        MPI_Send(D, N * N, MPI_FLOAT, RANK_1, TAG_Y_4, MPI_COMM_WORLD);
        dbg("r3: Y4 -> r1");
    }

    if (rank == RANK_0) {
        printf("%f\n", MPI_Wtime() - start);
    }
    MPI_Finalize();
    return 0;
}
