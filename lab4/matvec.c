#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    const double* A;   // m x n (row-major)
    const double* x;   // n
    double* y;         // m (salida)
    int m, n, t;       // filas, columnas, hilos
} MatVecCtx;

void* worker(void* arg) {
    long tid = (long)arg;
    MatVecCtx* c = ((MatVecCtx**)0)[0]; // never used: silence -Wstrict-aliasing
    c = (MatVecCtx*)(((void**)arg)[0]); // trick to pass both tid+ctx
    tid = (long)(((void**)arg)[1]);

    for (int i = (int)tid; i < c->m; i += c->t) {
        const double* Ai = c->A + (size_t)i * c->n;
        double sum = 0.0;
        for (int j = 0; j < c->n; ++j) sum += Ai[j] * c->x[j];
        c->y[i] = sum; // cada fila es de un solo hilo -> sin carreras
    }
    return NULL;
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr,"uso: %s m n t <A y x por stdin>\n", argv[0]); return 1; }
    int m = atoi(argv[1]), n = atoi(argv[2]), t = atoi(argv[3]);
    double* A = (double*)malloc((size_t)m*n*sizeof(double));
    double* x = (double*)malloc((size_t)n*sizeof(double));
    double* y = (double*)calloc((size_t)m, sizeof(double));
    if (!A || !x || !y) { perror("malloc"); return 1; }

    // Entrada simple: primero A por filas, luego x
    for (int i=0;i<m*n;i++) if (scanf("%lf",&A[i])!=1) return 1;
    for (int j=0;j<n;j++)   if (scanf("%lf",&x[j])!=1) return 1;

    pthread_t* th = (pthread_t*)malloc((size_t)t*sizeof(pthread_t));
    // Pasamos (ctx, tid) empaquetados
    void** args = (void**)malloc((size_t)t*2*sizeof(void*));
    MatVecCtx ctx = {A,x,y,m,n,t};

    for (long k=0;k<t;k++) {
        args[2*k]   = &ctx;
        args[2*k+1] = (void*)k;
        if (pthread_create(&th[k], NULL, worker, &args[2*k])) { perror("pthread_create"); return 1; }
    }
    for (int k=0;k<t;k++) pthread_join(th[k], NULL);

    // Imprime y
    for (int i=0;i<m;i++) printf("%.10f\n", y[i]);

    free(args); free(th); free(y); free(x); free(A);
    return 0;
}