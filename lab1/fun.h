#ifndef FUN_H
#define FUN_H

#define N 64

void inits(float a[N][N]);
void add(const float a[N][N], const float b[N][N], float res[N][N]);
void sub(const float a[N][N], const float b[N][N], float res[N][N]);
void mul(const float a[N][N], const float b[N][N], float res[N][N]);
void divv(const float a[N][N], const float b[N][N], float res[N][N]);

#endif
