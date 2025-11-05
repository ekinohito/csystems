#include <stdio.h>
#include "fun.h"

// Простая инициализация матриц значениями (например, 1.0)
void inits(float a[N][N])
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            a[i][j] = i + j;
}

// Покомпонентное сложение матриц
void add(const float a[N][N], const float b[N][N], float res[N][N])
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            res[i][j] = a[i][j] + b[i][j];
}

// Покомпонентное вычитание матриц
void sub(const float a[N][N], const float b[N][N], float res[N][N])
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            res[i][j] = a[i][j] - b[i][j];
}

// Покомпонентное "умножение" матриц (элемент на элемент)
void mul(const float a[N][N], const float b[N][N], float res[N][N])
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            res[i][j] = a[i][j] * b[i][j];
}

// Покомпонентное деление матриц (a / b)
void divv(const float a[N][N], const float b[N][N], float res[N][N])
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            res[i][j] = (b[i][j] != 0.0f) ? (a[i][j] / b[i][j]) : 0.0f;
}
