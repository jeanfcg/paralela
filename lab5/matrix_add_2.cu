#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* KERNEL (Paso C) Un hilo calcula una fila completa de C */
__global__ void matrixAddRow(float *A, float *B, float *C, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // 1D: un thread = una fila

    if (row < N) {
        int base = row * N;
        for (int col = 0; col < N; col++) {
            int idx = base + col;
            C[idx] = A[idx] + B[idx];
        }
    }
}

int main() {
    int N = 32;  // matriz NxN
    int size = N * N * sizeof(float);

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    /* Reservar memoria en host */
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    /* Inicializar matrices */
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    /* Reservar memoria en device */
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    /* Copiar datos al device */
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    /* Configuración de ejecución (Paso C) Un thread = 1 fila => necesitamos N threads en total */
    int threadsPerBlock = 256; // típico se puedes usar 128 o 256
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Lanzar kernel */
    matrixAddRow<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    /* Copiar resultado al host */
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    /* Mostrar resultado */
    printf("Resultado (C = A + B) por filas:\n");
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            printf("%.1f ", h_C[row * N + col]);
        }
        printf("\n");
    }

    /* Liberar memoria */
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
