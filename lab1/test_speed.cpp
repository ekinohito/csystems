#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "fun.h"

// измерение времени в секундах с плавающей точкой
double seconds()
{
    return (double)clock() / CLOCKS_PER_SEC;
}

// вспомогательная функция для заполнения матрицы случайными числами
void fill_random(float a[N][N])
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            a[i][j] = (float)rand() / RAND_MAX;
}

int main(void)
{
    srand(42); // фиксируем seed для воспроизводимости
    const int repeats = 100000; // число повторов для усреднения

    // массивы для тестов
    static float A[N][N], B[N][N], R[N][N];
    fill_random(A);
    fill_random(B);

    printf("Matrix size N = %d\n", N);
    printf("Averaging over %d runs\n\n", repeats);

    double start, end, total;

    // измеряем add()
    start = seconds();
    for (int i = 0; i < repeats; i++) {
        add(A, B, R);
    }
    total = seconds() - start;
    printf("add: %.6f sec (avg)\n", total / repeats);

    // измеряем sub()
    start = seconds();
    for (int i = 0; i < repeats; i++) {
        sub(A, B, R);
    }
    total = seconds() - start;
    printf("sub: %.6f sec (avg)\n", total / repeats);

    // измеряем mul()
    start = seconds();
    for (int i = 0; i < repeats; i++) {
        mul(A, B, R);
    }
    total = seconds() - start;
    printf("mul: %.6f sec (avg)\n", total / repeats);

    // измеряем divv()
    start = seconds();
    for (int i = 0; i < repeats; i++) {
        divv(A, B, R);
    }
    total = seconds() - start;
    printf("divv: %.6f sec (avg)\n", total / repeats);

    return 0;
}
