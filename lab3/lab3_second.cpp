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
// VERSION 3: #pragma omp parallel for
// ===========================================================================

void mat_add_for(double **A, double **B, double **C, int n)
{
#pragma omp parallel for schedule(dynamic) collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] + B[i][j];
}

void mat_sub_for(double **A, double **B, double **C, int n)
{
#pragma omp parallel for schedule(dynamic) collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] - B[i][j];
}

void mat_div_elem_for(double **A, double **B, double **C, int n)
{
#pragma omp parallel for schedule(dynamic) collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] / B[i][j];
}

void mat_mul_for(double **A, double **B, double **C, int n)
{
#pragma omp parallel for schedule(dynamic) collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = 0.0;

#pragma omp parallel for schedule(dynamic) collapse(3)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];
}

void compute_y_for(double **A, double **B, double **C, double **Y, int n)
{
    double **temp1 = alloc_matrix(n); // A + B
    double **temp2 = alloc_matrix(n); // C / (A+B)
    double **temp3 = alloc_matrix(n); // A * B

    mat_add_for(A, B, temp1, n);
    mat_div_elem_for(C, temp1, temp2, n);
    mat_mul_for(A, B, temp3, n);
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
    double **Y3 = alloc_matrix(N);

    generate_random_matrices(A, B, C, N, seed);

    double t3 = omp_get_wtime();
    compute_y_for(A, B, C, Y3, N);
    t3 = omp_get_wtime() - t3;

    printf("omp for:          %.4f s\n", t3);

    // --------------------- Print small part ---------------------
    // print_matrix_5x5(Y3, "Y = A*B + C/(A+B)  [from omp for version]");

    // Cleanup
    free_matrix(A, N);
    free_matrix(B, N);
    free_matrix(C, N);
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
}
