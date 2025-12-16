#include <stdio.h>
#include <cuda_runtime.h>

/* KERNEL (Paso B) Un hilo calcula un elemento C[row][col]  */

__global__ void matrixAdd(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int idx = row * N + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int N = 32;  // matriz 32x32
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

    /* Configuración de ejecución */
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);

    /* Lanzar kernel */
    matrixAdd<<<grid, block>>>(d_A, d_B, d_C, N);

    /* Copiar resultado al host */
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    /* Mostrar resultado */
    printf("Resultado (vector A+B):\n");
    for (int i = 0; i < N * N; i++) {
        printf("C[%d] = %.1f\n", i, h_C[i]);
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
