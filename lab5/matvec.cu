#include <stdio.h>
#include <cuda_runtime.h>

/* Kernel: un hilo calcula un elemento del vector A */
__global__ void matVecKernel(float *A, float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += B[i * N + j] * C[j];
        }
        A[i] = sum;
    }
}

int main()
{
    int N = 1024;
    int sizeMat = N * N * sizeof(float);
    int sizeVec = N * sizeof(float);

    /* Memoria en host */
    float *h_A = (float *)malloc(sizeVec);
    float *h_B = (float *)malloc(sizeMat);
    float *h_C = (float *)malloc(sizeVec);

    /* Inicialización */
    for (int i = 0; i < N; i++) {
        h_C[i] = 1.0f;
        for (int j = 0; j < N; j++) {
            h_B[i * N + j] = 1.0f;
        }
    }

    /* Memoria en device */
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, sizeVec);
    cudaMalloc((void **)&d_B, sizeMat);
    cudaMalloc((void **)&d_C, sizeVec);

    /* Copiar datos al device */
    cudaMemcpy(d_B, h_B, sizeMat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, sizeVec, cudaMemcpyHostToDevice);

    /* Configuración de ejecución */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    matVecKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    /* Copiar resultado al host */
    cudaMemcpy(h_A, d_A, sizeVec, cudaMemcpyDeviceToHost);


    /* Mostrar todos los valores del vector A */
    printf("Vector A resultante:\n");
    for (int i = 0; i < N; i++) {
        printf("A[%d] = %f\n", i, h_A[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;

}
