#include <iostream>
#include <cuda_runtime.h>

// Kernel en GPU: cada hilo suma 1 a un elemento
__global__ void addOne(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1;
    }
}

int main() {
    int n = 10;
    int size = n * sizeof(int);

    // Crear arreglo en CPU
    int h_data[10];
    for (int i = 0; i < n; i++) h_data[i] = i;

    // Reservar memoria en GPU
    int *d_data;
    cudaMalloc((void**)&d_data, size);

    // Copiar datos CPU → GPU
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // Lanzar kernel: 1 bloque de 10 hilos
    addOne<<<1, 10>>>(d_data, n);

    // Copiar resultado GPU → CPU
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    // Mostrar resultado
    for (int i = 0; i < n; i++)
        std::cout << h_data[i] << " ";
    std::cout << std::endl;

    // Liberar memoria
    cudaFree(d_data);

    return 0;
}