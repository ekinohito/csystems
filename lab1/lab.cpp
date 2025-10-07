#include <stdio.h>
#define N 32

int lab_main(void)
{
    float A[N], B[N], C[N], Y[N];
    float S1 = 2.0; // пример значения скаляра
    int i;

    // Инициализация исходных данных
    for (i = 0; i < N; i++) {
        A[i] = i + 1;
        B[i] = i + 2;
        C[i] = i + 3;
    }

    // Вычисления: Y[i] = A[i]*S1 + C[i]/(A[i] + B[i])
    for (i = 0; i < N; i++) {
        Y[i] = A[i] * S1 + C[i] / (A[i] + B[i]);
    }

    // Вывод результатов
    for (i = 0; i < N; i++) {
        printf("Y[%d] = %f\n", i, Y[i]);
    }

    return 0;
}
