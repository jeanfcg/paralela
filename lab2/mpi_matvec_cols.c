#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static void pack_block_cols(const double* A_full, int n, int c0, int ccols, double* A_pack) {
// Copia columnas [c0, c0+ccols) de A (n×n, fila mayor) a un buffer compacto (n × ccols)
for (int i = 0; i < n; i++) {
const double* src = A_full + i*(size_t)n + c0;
double*       dst = A_pack + i*(size_t)ccols;
for (int j = 0; j < ccols; j++) dst[j] = src[j];
}
}

int main(int argc, char** argv){
MPI_Init(&argc,&argv);
int rank,p; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&p);


if (rank==0 && argc<2){ fprintf(stderr,"Uso: %s <input.txt>\n", argv[0]); MPI_Abort(MPI_COMM_WORLD,1); }

int n = 0;
double* x_full = NULL;
double* A_full = NULL;

if (rank==0){
    FILE* f = fopen(argv[1], "r");
    if (!f){ perror("fopen"); MPI_Abort(MPI_COMM_WORLD,1); }
    if (fscanf(f, "%d", &n) != 1){ fprintf(stderr,"Formato: n, luego n doubles de x, luego n*n de A\n"); MPI_Abort(MPI_COMM_WORLD,1); }
    if (n<=0){ fprintf(stderr,"n invalido\n"); MPI_Abort(MPI_COMM_WORLD,1); }
    x_full = (double*)malloc(sizeof(double)*n);
    for (int i=0;i<n;i++) if (fscanf(f, "%lf", &x_full[i])!=1){ fprintf(stderr,"x incompleto\n"); MPI_Abort(MPI_COMM_WORLD,1); }
    A_full = (double*)malloc(sizeof(double)*n*(size_t)n);
    for (int i=0;i<n;i++)
        for (int j=0;j<n;j++)
            if (fscanf(f, "%lf", &A_full[i*(size_t)n + j])!=1){ fprintf(stderr,"A incompleta\n"); MPI_Abort(MPI_COMM_WORLD,1); }
    fclose(f);
}

// Broadcast de n
MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
if (n % p != 0){
    if (rank==0) fprintf(stderr,"Se requiere que n sea divisible por comm_sz (= %d)\n", p);
    MPI_Abort(MPI_COMM_WORLD,1);
}
int ccols = n / p;      // columnas locales por proceso
int rrows = n / p;      // filas por proceso para el reduce_scatter

// Buffers locales
double* x_loc = (double*)malloc(sizeof(double)*ccols);
double* A_loc = (double*)malloc(sizeof(double)*n*(size_t)ccols);   // n × ccols (compacto por columnas)
double* y_loc = (double*)calloc(n, sizeof(double));                 // contribución local (n)

// Distribución desde el rank 0: columnas por bloques
if (rank==0){
    // Para q=0, empacar a A_loc/x_loc; para q>0, empaquetar a tmp y enviar
    for (int q=0; q<p; q++){
        int c0 = q*ccols;
        if (q==0){
            for (int j=0;j<ccols;j++) x_loc[j] = x_full[c0+j];
            pack_block_cols(A_full, n, c0, ccols, A_loc);
        } else {
            double* tmpA = (double*)malloc(sizeof(double)*n*(size_t)ccols);
            double* tmpx = (double*)malloc(sizeof(double)*ccols);
            for (int j=0;j<ccols;j++) tmpx[j] = x_full[c0+j];
            pack_block_cols(A_full, n, c0, ccols, tmpA);
            MPI_Send(tmpx, ccols, MPI_DOUBLE, q, 10, MPI_COMM_WORLD);
            MPI_Send(tmpA, n*ccols, MPI_DOUBLE, q, 11, MPI_COMM_WORLD);
            free(tmpA); free(tmpx);
        }
    }
} else {
    MPI_Recv(x_loc, ccols,   MPI_DOUBLE, 0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(A_loc, n*ccols, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

// Cálculo local: y_loc = A_loc · x_loc  (A_loc es n×ccols, almacenado por columnas contiguas)
for (int i=0;i<n;i++){
    const double* Arow = A_loc + i*(size_t)ccols;
    double acc = 0.0;
    for (int j=0;j<ccols;j++) acc += Arow[j] * x_loc[j];
    y_loc[i] = acc; // contribución de mis columnas a la fila i
}

// Reduce+Scatter: suma entre procesos y entrega un bloque de filas a cada uno
double* y_block = (double*)malloc(sizeof(double)*rrows);
int* recvcounts = (int*)malloc(sizeof(int)*p);
for (int q=0;q<p;q++) recvcounts[q] = rrows;
MPI_Reduce_scatter(y_loc, y_block, recvcounts, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

// (Opcional) Recolectar en 0 para imprimir el y completo
double* y_full = NULL;
if (rank==0) y_full = (double*)malloc(sizeof(double)*n);
MPI_Gather(y_block, rrows, MPI_DOUBLE, y_full, rrows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

if (rank==0){
    printf("y = [");
    for (int i=0;i<n;i++) printf("%s%.6f", (i?", ":""), y_full[i]);
    printf("]\n");
}

free(y_full); free(recvcounts);
free(y_block); free(y_loc); free(A_loc); free(x_loc);
if (rank==0){ free(A_full); free(x_full); }

MPI_Finalize();
return 0;


}
