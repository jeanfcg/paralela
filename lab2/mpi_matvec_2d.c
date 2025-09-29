#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static void pack_block(const double* A, int n, int r0, int c0, int b, double* B){
    // Copia el bloque A[r0:r0+b, c0:c0+b] (fila mayor) a B (b×b, fila mayor)
    for (int i = 0; i < b; i++){
        const double* src = A + (size_t)(r0+i)*n + c0;
        double* dst = B + (size_t)i*b;
        for (int j = 0; j < b; j++) dst[j] = src[j];
    }
}

int main(int argc, char** argv){
    MPI_Init(&argc,&argv);
    int rank,p; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&p);

    if (rank==0 && argc<2){
        fprintf(stderr,"Uso: %s <input.txt>\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    // Requerido: p debe ser cuadrado perfecto
    int q = (int)(sqrt((double)p) + 0.5);
    if (q*q != p){
        if (rank==0) fprintf(stderr,"Error: comm_sz=%d no es cuadrado perfecto\n", p);
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    int n = 0;
    double *A_full=NULL, *x_full=NULL;
    if (rank==0){
        FILE* f = fopen(argv[1], "r");
        if (!f){ perror("fopen"); MPI_Abort(MPI_COMM_WORLD,1); }
        if (fscanf(f, "%d", &n) != 1){ fprintf(stderr,"Formato: n, luego n doubles de x, luego n*n de A\n"); MPI_Abort(MPI_COMM_WORLD,1); }
        x_full = (double*)malloc(sizeof(double)*n);
        for (int i=0;i<n;i++) if (fscanf(f, "%lf",&x_full[i])!=1){ fprintf(stderr,"x incompleto\n"); MPI_Abort(MPI_COMM_WORLD,1); }
        A_full = (double*)malloc(sizeof(double)*n*(size_t)n);
        for (int i=0;i<n;i++)
            for (int j=0;j<n;j++)
                if (fscanf(f, "%lf", &A_full[i*(size_t)n + j])!=1){ fprintf(stderr,"A incompleta\n"); MPI_Abort(MPI_COMM_WORLD,1); }
        fclose(f);
    }

    MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
    if (n % q != 0){
        if (rank==0) fprintf(stderr,"Error: n=%d no es divisible por sqrt(p)=%d\n", n, q);
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    int b = n / q;                 // tamaño de bloque

    // Coordenadas en la grilla q×q
    int row = rank / q;
    int col = rank % q;

    // Comunicadores por fila y por columna (ranks locales = índice ordenado)
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row, col, &row_comm);  // color=row, key=col
    MPI_Comm_split(MPI_COMM_WORLD, col, row, &col_comm);  // color=col, key=row

    // Buffers locales
    double* A_loc = (double*)malloc(sizeof(double)*b*(size_t)b);
    double* x_blk = (row==col)? (double*)malloc(sizeof(double)*b) : NULL; // solo diagonal al inicio
    double* x_need = (double*)malloc(sizeof(double)*b);                   // recibido por broadcast
    double* y_part = (double*)malloc(sizeof(double)*b);                   // contribución local
    double* y_blk  = (col==row)? (double*)malloc(sizeof(double)*b) : NULL;// resultado en diagonal

    // Distribución desde rank 0
    if (rank==0){
        for (int r=0;r<p;r++){
            int ri = r / q, rj = r % q;
            int r0 = ri*b, c0 = rj*b;

            if (r == 0){
                pack_block(A_full, n, r0, c0, b, A_loc);
                // x_block para (0,0)
                for (int j=0;j<b;j++) x_blk[j] = x_full[0*b + j];
            }else{
                double* tmpA = (double*)malloc(sizeof(double)*b*(size_t)b);
                pack_block(A_full, n, r0, c0, b, tmpA);
                MPI_Send(tmpA, b*b, MPI_DOUBLE, r, 10, MPI_COMM_WORLD);
                free(tmpA);

                if (ri == rj){
                    MPI_Send(&x_full[rj*b], b, MPI_DOUBLE, r, 20, MPI_COMM_WORLD);
                }
            }
        }
        free(A_full); free(x_full);
    }else{
        MPI_Recv(A_loc, b*b, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (row==col){
            x_blk = (double*)malloc(sizeof(double)*b);
            MPI_Recv(x_blk, b, MPI_DOUBLE, 0, 20, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Broadcast del bloque de x por cada columna (root local = col)
    if (row==col) {
        for (int j=0;j<b;j++) x_need[j] = x_blk[j];
    }
    MPI_Bcast(x_need, b, MPI_DOUBLE, /*root=*/col, col_comm);

    // y_part = A_loc (b×b) * x_need (b)
    for (int i=0;i<b;i++){
        double acc = 0.0;
        const double* Ai = A_loc + (size_t)i*b;
        for (int j=0;j<b;j++) acc += Ai[j]*x_need[j];
        y_part[i] = acc;
    }

    // Reduce por filas hacia el diagonal (root local en row_comm = row)
    MPI_Reduce(y_part, (col==row? y_blk : NULL), b, MPI_DOUBLE, MPI_SUM, /*root=*/row, row_comm);

    // Recolectar bloques y en la diagonal y juntar en 0
    MPI_Comm diag_comm;
    if (row==col) {
        MPI_Comm_split(MPI_COMM_WORLD, /*color=*/0, /*key=*/row, &diag_comm);
    } else {
        MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, 0, &diag_comm);
    }

    if (row==col){
        double* y_full = NULL;
        if (rank == 0) y_full = (double*)malloc(sizeof(double)*n);
        MPI_Gather(y_blk, b, MPI_DOUBLE, y_full, b, MPI_DOUBLE, /*root=*/0, diag_comm);

        if (rank == 0){
            printf("y = [");
            for (int i=0;i<n;i++) printf("%s%.6f", (i?", ":""), y_full[i]);
            printf("]\n");
            free(y_full);
        }
        MPI_Comm_free(&diag_comm);
    }

    // Limpieza
    if (y_blk)  free(y_blk);
    free(y_part);
    free(x_need);
    if (x_blk)  free(x_blk);
    free(A_loc);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Finalize();
    return 0;
}
