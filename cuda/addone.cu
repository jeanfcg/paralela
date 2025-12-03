#include <iostream>
#include <cuda_runtime.h>

//kernel en GPU: cada hilo suma 1 a un elemento
__global__ void addOne(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1;
    }
}

int main() {
    int n = 10;
    int size = n * sizeof(int);

    //crear arreglo en CPU
    int h_data[10];
    for (int i = 0; i < n; i++) h_data[i] = i;

    //reservar memoria en GPU
    int *d_data;
    cudaMalloc((void**)&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice); //copiar datos del cpu al gpu

    // Lanzar kernel: 1 bloque de 10 hilos
    addOne<<<1, 10>>>(d_data, n);
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost); //copiar el resultado del gpu al cpu

    for (int i = 0; i < n; i++)
        std::cout << h_data[i] << " ";
    std::cout << std::endl;

    cudaFree(d_data);

    return 0;

}
