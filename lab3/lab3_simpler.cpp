#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cmath>

// Global size
int N;

// Allocate NxN matrix
double **alloc_matrix(int n)
{
    double **m = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++)
        m[i] = (double *)malloc(n * sizeof(double));
    return m;
}

void free_matrix(double **m, int n)
{
    for (int i = 0; i < n; i++)
        free(m[i]);
    free(m);
}

// Random matrix generation
void generate_random_matrices(double **A, double **B, double **C, int n, unsigned int seed)
{
    srand(seed);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A[i][j] = (double)rand() / RAND_MAX * 20.0 - 10.0;
            B[i][j] = (double)rand() / RAND_MAX * 20.0 - 10.0;
            C[i][j] = (double)rand() / RAND_MAX * 20.0 - 10.0;
        }
    }
}

// Check if two matrices are equal within tolerance
int matrices_equal(double** A, double** B, int n, double tol) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(A[i][j] - B[i][j]) > tol) {
                printf("Difference at [%d][%d]: %.12e vs %.12e (diff = %.12e)\n",
                       i, j, A[i][j], B[i][j], fabs(A[i][j] - B[i][j]));
                return 0;
            }
        }
    }
    return 1;
}

// Print top-left 5x5 block (or full matrix if N < 5)
void print_matrix_5x5(double** M, const char* name) {
    int size = (N < 5) ? N : 5;
    printf("\n--- %s (top-left %dx%d) ---\n", name, size, size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%12.6f ", M[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

// ===========================================================================
// VERSION 1: Manual threading with #pragma omp parallel (no for, no sections)
// ===========================================================================

void mat_add_parallel(double **A, double **B, double **C, int n)
{
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();
        int rows_per_thread = n / nt;
        int start = tid * rows_per_thread;
        int end = (tid == nt - 1) ? n : start + rows_per_thread;

        for (int i = start; i < end; i++)
            for (int j = 0; j < n; j++)
                C[i][j] = A[i][j] + B[i][j];
    }
}

void mat_sub_parallel(double **A, double **B, double **C, int n)
{
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();
        int rows_per_thread = n / nt;
        int start = tid * rows_per_thread;
        int end = (tid == nt - 1) ? n : start + rows_per_thread;

        for (int i = start; i < end; i++)
            for (int j = 0; j < n; j++)
                C[i][j] = A[i][j] - B[i][j];
    }
}

void mat_div_elem_parallel(double **A, double **B, double **C, int n)
{
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();
        int rows_per_thread = n / nt;
        int start = tid * rows_per_thread;
        int end = (tid == nt - 1) ? n : start + rows_per_thread;

        for (int i = start; i < end; i++)
            for (int j = 0; j < n; j++)
                C[i][j] = A[i][j] / B[i][j];
    }
}

void mat_mul_elem_parallel(double **A, double **B, double **C, int n)
{
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();
        int rows_per_thread = n / nt;
        int start = tid * rows_per_thread;
        int end = (tid == nt - 1) ? n : start + rows_per_thread;

        for (int i = start; i < end; i++)
            for (int j = 0; j < n; j++)
                C[i][j] = A[i][j] * B[i][j];
    }
}

void mat_mul_parallel(double **A, double **B, double **C, int n)
{
    // Zero C first
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = 0.0;

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();
        int rows_per_thread = n / nt;
        int start = tid * rows_per_thread;
        int end = (tid == nt - 1) ? n : start + rows_per_thread;

        for (int i = start; i < end; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double sum = 0.0;
                for (int k = 0; k < n; k++)
                    sum += A[i][k] * B[k][j];
                C[i][j] = sum;
            }
        }
    }
}

// ===========================================================================
// VERSION 2: #pragma omp parallel sections with exactly 8 sections
// ===========================================================================

#define SECTION_START(sec) ((sec) * (N / 8) + ((sec) < (N % 8) ? (sec) : (N % 8)))
#define SECTION_END(sec) (((sec) + 1) * (N / 8) + (((sec) + 1) < (N % 8) ? ((sec) + 1) : (N % 8)))

void section_add(int sec, double **A, double **B, double **C)
{
    int start = SECTION_START(sec);
    int end = SECTION_END(sec);
    for (int i = start; i < end; i++)
        for (int j = 0; j < N; j++)
            C[i][j] = A[i][j] + B[i][j];
}

void section_sub(int sec, double **A, double **B, double **C)
{
    int start = SECTION_START(sec);
    int end = SECTION_END(sec);
    for (int i = start; i < end; i++)
        for (int j = 0; j < N; j++)
            C[i][j] = A[i][j] - B[i][j];
}

void section_div_elem(int sec, double **A, double **B, double **C)
{
    int start = SECTION_START(sec);
    int end = SECTION_END(sec);
    for (int i = start; i < end; i++)
        for (int j = 0; j < N; j++)
            C[i][j] = A[i][j] / B[i][j];
}

void section_mul_elem(int sec, double **A, double **B, double **C)
{
    int start = SECTION_START(sec);
    int end = SECTION_END(sec);
    for (int i = start; i < end; i++)
        for (int j = 0; j < N; j++)
            C[i][j] = A[i][j] * B[i][j];
}

void section_mul_row_block(int sec, double **A, double **B, double **C)
{
    int start = SECTION_START(sec);
    int end = SECTION_END(sec);
    for (int i = start; i < end; i++)
        for (int j = 0; j < N; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < N; k++)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
}

void mat_add_sections(double **A, double **B, double **C, int n)
{
#pragma omp parallel sections
    {
#pragma omp section
        {
            section_add(0, A, B, C);
        }
#pragma omp section
        {
            section_add(1, A, B, C);
        }
#pragma omp section
        {
            section_add(2, A, B, C);
        }
#pragma omp section
        {
            section_add(3, A, B, C);
        }
#pragma omp section
        {
            section_add(4, A, B, C);
        }
#pragma omp section
        {
            section_add(5, A, B, C);
        }
#pragma omp section
        {
            section_add(6, A, B, C);
        }
#pragma omp section
        {
            section_add(7, A, B, C);
        }
    }
}

void mat_sub_sections(double **A, double **B, double **C, int n)
{
#pragma omp parallel sections
    {
#pragma omp section
        {
            section_sub(0, A, B, C);
        }
#pragma omp section
        {
            section_sub(1, A, B, C);
        }
#pragma omp section
        {
            section_sub(2, A, B, C);
        }
#pragma omp section
        {
            section_sub(3, A, B, C);
        }
#pragma omp section
        {
            section_sub(4, A, B, C);
        }
#pragma omp section
        {
            section_sub(5, A, B, C);
        }
#pragma omp section
        {
            section_sub(6, A, B, C);
        }
#pragma omp section
        {
            section_sub(7, A, B, C);
        }
    }
}

void mat_div_elem_sections(double **A, double **B, double **C, int n)
{
#pragma omp parallel sections
    {
#pragma omp section
        {
            section_div_elem(0, A, B, C);
        }
#pragma omp section
        {
            section_div_elem(1, A, B, C);
        }
#pragma omp section
        {
            section_div_elem(2, A, B, C);
        }
#pragma omp section
        {
            section_div_elem(3, A, B, C);
        }
#pragma omp section
        {
            section_div_elem(4, A, B, C);
        }
#pragma omp section
        {
            section_div_elem(5, A, B, C);
        }
#pragma omp section
        {
            section_div_elem(6, A, B, C);
        }
#pragma omp section
        {
            section_div_elem(7, A, B, C);
        }
    }
}

void mat_mul_elem_sections(double **A, double **B, double **C, int n)
{
#pragma omp parallel sections
    {
#pragma omp section
        {
            section_mul_elem(0, A, B, C);
        }
#pragma omp section
        {
            section_mul_elem(1, A, B, C);
        }
#pragma omp section
        {
            section_mul_elem(2, A, B, C);
        }
#pragma omp section
        {
            section_mul_elem(3, A, B, C);
        }
#pragma omp section
        {
            section_mul_elem(4, A, B, C);
        }
#pragma omp section
        {
            section_mul_elem(5, A, B, C);
        }
#pragma omp section
        {
            section_mul_elem(6, A, B, C);
        }
#pragma omp section
        {
            section_mul_elem(7, A, B, C);
        }
    }
}

void mat_mul_sections(double **A, double **B, double **C, int n)
{
    // Zero matrix first (can be parallelized separately)
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = 0.0;

#pragma omp parallel sections
    {
#pragma omp section
        {
            section_mul_row_block(0, A, B, C);
        }
#pragma omp section
        {
            section_mul_row_block(1, A, B, C);
        }
#pragma omp section
        {
            section_mul_row_block(2, A, B, C);
        }
#pragma omp section
        {
            section_mul_row_block(3, A, B, C);
        }
#pragma omp section
        {
            section_mul_row_block(4, A, B, C);
        }
#pragma omp section
        {
            section_mul_row_block(5, A, B, C);
        }
#pragma omp section
        {
            section_mul_row_block(6, A, B, C);
        }
#pragma omp section
        {
            section_mul_row_block(7, A, B, C);
        }
    }
}

// ===========================================================================
// VERSION 3: #pragma omp parallel for
// ===========================================================================

void mat_add_for(double **A, double **B, double **C, int n)
{
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] + B[i][j];
}

void mat_sub_for(double **A, double **B, double **C, int n)
{
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] - B[i][j];
}

void mat_div_elem_for(double **A, double **B, double **C, int n)
{
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] / B[i][j];
}

void mat_mul_elem_for(double **A, double **B, double **C, int n)
{
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] * B[i][j];
}

void mat_mul_for(double **A, double **B, double **C, int n)
{
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = 0.0;

#pragma omp parallel for collapse(3)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];
}

// ===========================================================================
// Combined computation: Y = A*B + C / (A + B)
// ===========================================================================

void compute_y_parallel(double **A, double **B, double **C, double **Y, int n)
{
    double **temp1 = alloc_matrix(n); // A + B
    double **temp2 = alloc_matrix(n); // C / (A+B)
    double **temp3 = alloc_matrix(n); // A * B

    mat_add_parallel(A, B, temp1, n);
    mat_div_elem_parallel(C, temp1, temp2, n);
    mat_mul_elem_parallel(A, B, temp3, n);
    mat_add_parallel(temp3, temp2, Y, n);

    free_matrix(temp1, n);
    free_matrix(temp2, n);
    free_matrix(temp3, n);
}

void compute_y_sections(double **A, double **B, double **C, double **Y, int n)
{
    double **temp1 = alloc_matrix(n); // A + B
    double **temp2 = alloc_matrix(n); // C / (A+B)
    double **temp3 = alloc_matrix(n); // A * B

    mat_add_sections(A, B, temp1, n);
    mat_div_elem_sections(C, temp1, temp2, n);
    mat_mul_elem_sections(A, B, temp3, n);
    mat_add_sections(temp3, temp2, Y, n);

    free_matrix(temp1, n);
    free_matrix(temp2, n);
    free_matrix(temp3, n);
}

void compute_y_for(double **A, double **B, double **C, double **Y, int n)
{
    double **temp1 = alloc_matrix(n); // A + B
    double **temp2 = alloc_matrix(n); // C / (A+B)
    double **temp3 = alloc_matrix(n); // A * B

    mat_add_for(A, B, temp1, n);
    mat_div_elem_for(C, temp1, temp2, n);
    mat_mul_elem_for(A, B, temp3, n);
    mat_add_for(temp3, temp2, Y, n);

    free_matrix(temp1, n);
    free_matrix(temp2, n);
    free_matrix(temp3, n);
}

// ===========================================================================
// Main (for testing)
// ===========================================================================

int main_part()
{
    unsigned int seed = 42;

    double **A = alloc_matrix(N);
    double **B = alloc_matrix(N);
    double **C = alloc_matrix(N);
    double **Y1 = alloc_matrix(N);
    double **Y2 = alloc_matrix(N);
    double **Y3 = alloc_matrix(N);

    generate_random_matrices(A, B, C, N, seed);

    double t1 = omp_get_wtime();
    compute_y_parallel(A, B, C, Y1, N);
    t1 = omp_get_wtime() - t1;

    double t2 = omp_get_wtime();
    compute_y_sections(A, B, C, Y2, N);
    t2 = omp_get_wtime() - t2;

    double t3 = omp_get_wtime();
    compute_y_for(A, B, C, Y3, N);
    t3 = omp_get_wtime() - t3;

    printf("Manual parallel:  %.6f s\n", t1);
    printf("Sections (8):     %.6f s\n", t2);
    printf("omp for:          %.6f s\n", t3);

    // --------------------- Correctness check ---------------------
    double tol = 1e-10;

    int ok12 = matrices_equal(Y1, Y2, N, tol);
    int ok13 = matrices_equal(Y1, Y3, N, tol);
    int ok23 = matrices_equal(Y2, Y3, N, tol);

    if (ok12 && ok13 && ok23) {
        printf("All three implementations produce IDENTICAL results (within %.0e)\n", tol);
    } else {
        printf("ERROR: Results differ between implementations!\n");
    }

    // --------------------- Print small part ---------------------
    // print_matrix_5x5(Y3, "Y = A*B + C/(A+B)  [from omp for version]");

    // Cleanup
    free_matrix(A, N);
    free_matrix(B, N);
    free_matrix(C, N);
    free_matrix(Y1, N);
    free_matrix(Y2, N);
    free_matrix(Y3, N);

    return 0;
}

int main() {
    N = 256;
    for (int i = 1; i <= 8; ++i) {
        omp_set_num_threads(i);
        printf("N = %d; Threads = %d\n", N, i);
        main_part();
    }
    N = 1024;
    for (int i = 1; i <= 8; ++i) {
        omp_set_num_threads(i);
        printf("N = %d; Threads = %d\n", N, i);
        main_part();
    }
    N = 4096;
    for (int i = 1; i <= 8; ++i) {
        omp_set_num_threads(i);
        printf("N = %d; Threads = %d\n", N, i);
        main_part();
    }
}
