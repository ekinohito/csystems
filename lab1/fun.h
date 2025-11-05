#ifndef FUN_H
#define FUN_H

#define N 64

void inits(float a[N][N]);
void add(float a[N][N], float b[N][N], float res[N][N]);
void sub(float a[N][N], float b[N][N], float res[N][N]);
void mul(float a[N][N], float b[N][N], float res[N][N]);
void divv(float a[N][N], float b[N][N], float res[N][N]);

#endif
