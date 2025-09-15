#include <iostream>
#include <vector>
#include <chrono>

using namespace std;
using namespace std::chrono;

void multiplicacionClasica(const vector<vector<double>>& A,
                           const vector<vector<double>>& B,
                           vector<vector<double>>& C,
                           int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    int n;
    cout << "Ingrese el tamaño de la matriz: ";
    cin >> n;

    vector<vector<double>> A(n, vector<double>(n, 1.0));
    vector<vector<double>> B(n, vector<double>(n, 1.0));
    vector<vector<double>> C(n, vector<double>(n, 0.0));

    auto start = high_resolution_clock::now();
    multiplicacionClasica(A, B, C, n);
    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "Tiempo de ejecucion (clasica): " << duration.count() << " ms" << endl;

    return 0;
}
