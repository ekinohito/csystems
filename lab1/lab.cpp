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
        MPI_Request send_b_minus_e, recv_e_minus_c, recv_y4, recv_y3, recv_y1;
        MPI_Send_init(A, N * N, MPI_FLOAT, RANK_3, TAG_B_MINUS_E, MPI_COMM_WORLD, &send_b_minus_e);
        MPI_Recv_init(C, N * N, MPI_FLOAT, RANK_2, TAG_E_MINUS_C, MPI_COMM_WORLD, &recv_e_minus_c);
        MPI_Recv_init(A, N * N, MPI_FLOAT, RANK_3, TAG_Y_4, MPI_COMM_WORLD, &recv_y4);
        MPI_Recv_init(D, N * N, MPI_FLOAT, RANK_2, TAG_Y_3, MPI_COMM_WORLD, &recv_y3);
        MPI_Recv_init(E, N * N, MPI_FLOAT, RANK_1, TAG_Y_1, MPI_COMM_WORLD, &recv_y1);

        MPI_Start(&recv_e_minus_c);
        MPI_Start(&recv_y4);
        MPI_Start(&recv_y3);

        sub(B, E, A);
        MPI_Start(&recv_y1);
        MPI_Start(&send_b_minus_e);
        dbg("r0: B - E -> r3");
        MPI_Wait(&recv_e_minus_c, &status);
        dbg("r0: r2 -> E - C");
        mul(A, C, B);
        MPI_Wait(&recv_y4, &status);
        dbg("r0: r3 -> Y4");
        add(B, A, C);
        MPI_Wait(&recv_y3, &status);
        dbg("r0: r2 -> Y3");
        MPI_Wait(&recv_y1, &status);
        dbg("r0: r1 -> Y1");
        dbg("r0: DONE");
    }
    else if (rank == RANK_1)
    {
        MPI_Request recv_e_minus_c, recv_y4, send_y1;
        MPI_Recv_init(E, N * N, MPI_FLOAT, RANK_2, TAG_E_MINUS_C, MPI_COMM_WORLD, &recv_e_minus_c); // T0 = E - C (Receive)
        MPI_Recv_init(B, N * N, MPI_FLOAT, RANK_3, TAG_Y_4, MPI_COMM_WORLD, &recv_y4);               // E = Y4 (Receive)
        MPI_Send_init(A, N * N, MPI_FLOAT, RANK_0, TAG_Y_1, MPI_COMM_WORLD, &send_y1);

        MPI_Start(&recv_e_minus_c);

        mul(B, D, C);
        MPI_Start(&recv_y4);
        MPI_Wait(&recv_e_minus_c, &status); // MPI_Recv(T0, N * N, MPI_FLOAT, RANK_2, TAG_E_MINUS_C, MPI_COMM_WORLD, &status); // T0 = E - C (Receive)
        dbg("r1: r2 -> E - C");
        divv(C, E, D);
        sub(D, A, C);
        MPI_Wait(&recv_y4, &status); // MPI_Recv(E, N * N, MPI_FLOAT, RANK_3, TAG_Y_4, MPI_COMM_WORLD, &status); // E = Y4 (Receive)
        dbg("r1: r3 -> Y4");
        add(C, B, A);
        MPI_Start(&send_y1); // MPI_Send(A, N * N, MPI_FLOAT, RANK_0, TAG_Y_1, MPI_COMM_WORLD);
        dbg("r1: Y1 -> r0");
    }
    else if (rank == RANK_2)
    {
        MPI_Request send_e_minus_c_1, send_e_minus_c_2, send_y3;
        MPI_Send_init(D, N * N, MPI_FLOAT, RANK_0, TAG_E_MINUS_C, MPI_COMM_WORLD, &send_e_minus_c_1);
        MPI_Send_init(D, N * N, MPI_FLOAT, RANK_1, TAG_E_MINUS_C, MPI_COMM_WORLD, &send_e_minus_c_2);
        MPI_Send_init(B, N * N, MPI_FLOAT, RANK_0, TAG_Y_3, MPI_COMM_WORLD, &send_y3);

        sub(E, C, D);
        MPI_Start(&send_e_minus_c_1); // MPI_Send(D, N * N, MPI_FLOAT, RANK_0, TAG_E_MINUS_C, MPI_COMM_WORLD);
        dbg("r2: E - C -> r0");
        MPI_Start(&send_e_minus_c_2); // MPI_Send(D, N * N, MPI_FLOAT, RANK_1, TAG_E_MINUS_C, MPI_COMM_WORLD);
        dbg("r2: E - C -> r1");
        sub(B, C, E);
        mul(A, E, B);
        MPI_Start(&send_y3); // MPI_Send(B, N * N, MPI_FLOAT, RANK_0, TAG_Y_3, MPI_COMM_WORLD);
        dbg("r2: Y3 -> r0");
    }
    else if (rank == RANK_3)
    {
        MPI_Request recv_b_minus_e, send_y_4_1, send_y_4_2;
        MPI_Recv_init(B, N * N, MPI_FLOAT, RANK_0, TAG_B_MINUS_E, MPI_COMM_WORLD, &recv_b_minus_e); // B = B - E (Receive)
        MPI_Send_init(D, N * N, MPI_FLOAT, RANK_0, TAG_Y_4, MPI_COMM_WORLD, &send_y_4_1);
        MPI_Send_init(D, N * N, MPI_FLOAT, RANK_1, TAG_Y_4, MPI_COMM_WORLD, &send_y_4_2);

        MPI_Start(&recv_b_minus_e);

        mul(C, C, A);
        MPI_Wait(&recv_b_minus_e, &status); // MPI_Recv(B, N * N, MPI_FLOAT, RANK_0, TAG_B_MINUS_E, MPI_COMM_WORLD, &status); // B = B - E (Receive)
        dbg("r3: r0 -> B - E");
        divv(B, A, D);
        MPI_Start(&send_y_4_1); // MPI_Send(D, N * N, MPI_FLOAT, RANK_0, TAG_Y_4, MPI_COMM_WORLD);
        dbg("r3: Y4 -> r0");
        MPI_Start(&send_y_4_2); // MPI_Send(D, N * N, MPI_FLOAT, RANK_1, TAG_Y_4, MPI_COMM_WORLD);
        dbg("r3: Y4 -> r1");
    }

    if (rank == RANK_0)
    {
        printf("%f\n", MPI_Wtime() - start);
    }
    MPI_Finalize();
    return 0;
}
