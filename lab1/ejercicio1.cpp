#include <iostream>
#include <vector>
#include <chrono>  // Para medir tiempos

using namespace std;
using namespace std::chrono;

const int MAX = 2000;

// Función para inicializar matriz y vectores
void initialize(vector<vector<double>>& A, vector<double>& x, vector<double>& y) {
    for (int i = 0; i < MAX; i++) {
        x[i] = 1.0;   // vector x lleno de 1
        y[i] = 0.0;   // vector y inicializado en 0
        for (int j = 0; j < MAX; j++) {
            A[i][j] = 1.0;  // matriz llena de 1
        }
    }
}

int main() {
    vector<vector<double>> A(MAX, vector<double>(MAX));
    vector<double> x(MAX), y(MAX);

    initialize(A, x, y);

    //Primer par de bucles (recorrido por filas)
    auto start1 = high_resolution_clock::now();

    for (int i = 0; i < MAX; i++) {
        for (int j = 0; j < MAX; j++) {
            y[i] += A[i][j] * x[j];
        }
    }

    auto end1 = high_resolution_clock::now();
    auto duration1 = duration_cast<milliseconds>(end1 - start1).count();
    cout << "Tiempo bucles por FILAS: " << duration1 << " ms" << endl;

    fill(y.begin(), y.end(), 0.0);

    //Segundo par de bucles (recorrido por columnas)
    auto start2 = high_resolution_clock::now();

    for (int j = 0; j < MAX; j++) {
        for (int i = 0; i < MAX; i++) {
            y[i] += A[i][j] * x[j];
        }
    }

    auto end2 = high_resolution_clock::now();
    auto duration2 = duration_cast<milliseconds>(end2 - start2).count();
    cout << "Tiempo bucles por COLUMNAS: " << duration2 << " ms" << endl;

    return 0;
}
