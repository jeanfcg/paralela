#include <iostream>
#include <vector>
#include <chrono>

using namespace std;
using namespace std::chrono;

void multiplicacionPorBloques(const vector<vector<double>>& A,
                              const vector<vector<double>>& B,
                              vector<vector<double>>& C,
                              int n, int blockSize) {
    for (int ii = 0; ii < n; ii += blockSize) {
        for (int jj = 0; jj < n; jj += blockSize) {
            for (int kk = 0; kk < n; kk += blockSize) {
                for (int i = ii; i < min(ii + blockSize, n); i++) {
                    for (int j = jj; j < min(jj + blockSize, n); j++) {
                        double sum = C[i][j];
                        for (int k = kk; k < min(kk + blockSize, n); k++) {
                            sum += A[i][k] * B[k][j];
                        }
                        C[i][j] = sum;
                    }
                }
            }
        }
    }
}

int main() {
    int n, blockSize;
    cout << "Ingrese el tamaño de la matriz: ";
    cin >> n;
    cout << "Ingrese el tamaño del bloque: ";
    cin >> blockSize;

    vector<vector<double>> A(n, vector<double>(n, 1.0));
    vector<vector<double>> B(n, vector<double>(n, 1.0));
    vector<vector<double>> C(n, vector<double>(n, 0.0));

    auto start = high_resolution_clock::now();
    multiplicacionPorBloques(A, B, C, n, blockSize);
    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "Tiempo de ejecucion (bloques): " << duration.count() << " ms" << endl;

    return 0;
}
