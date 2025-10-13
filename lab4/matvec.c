#define _POSIX_C_SOURCE 200809L
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

typedef struct {
    const double* A; const double* x; double* y;
    int m, n, T, tid;
} Ctx;

/* reloj simple (monotónico en segundos) */
static double now_sec(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}

/* PRNG reproducible muy simple */
static inline double rnd(uint64_t* s){
    *s = *s*6364136223846793005ULL + 1ULL;
    return (double)((*s>>11) & 0xFFFFFFFFFFFFULL) / (double)(1ULL<<52);
}

/* serial: referencia */
static void matvec_serial(const double* A,const double* x,double* y,int m,int n){
    for(int i=0;i<m;i++){
        const double* Ai = A + (size_t)i*n;
        double s=0.0; for(int j=0;j<n;j++) s += Ai[j]*x[j];
        y[i]=s;
    }
}

/* hilo: reparto cíclico (striped) i = tid, tid+T, ... */
static void* worker(void* arg){
    Ctx* c = (Ctx*)arg;
    for(int i=c->tid;i<c->m;i+=c->T){
        const double* Ai = c->A + (size_t)i*c->n;
        double s=0.0; for(int j=0;j<c->n;j++) s += Ai[j]*c->x[j];
        c->y[i]=s;
    }
    return NULL;
}

int main(int argc, char** argv){
    if(argc<4){ fprintf(stderr,"uso: %s m n hilos [seed]\n", argv[0]); return 1; }
    int m=atoi(argv[1]), n=atoi(argv[2]), T=atoi(argv[3]);
    if(m<=0||n<=0||T<=0){ fprintf(stderr,"parametros invalidos\n"); return 1; }
    uint64_t seed = (argc>=5)? (uint64_t)strtoull(argv[4],NULL,10) : 1ULL;

    size_t mn=(size_t)m*(size_t)n;
    double *A=malloc(mn*sizeof*A), *x=malloc((size_t)n*sizeof*x);
    double *y=malloc((size_t)m*sizeof*y), *yref=malloc((size_t)m*sizeof*yref);
    if(!A||!x||!y||!yref){ fprintf(stderr,"memoria insuficiente\n"); return 1; }

    /* datos reproducibles */
    uint64_t s=seed;
    for(size_t i=0;i<mn;i++) A[i]=rnd(&s);
    for(int j=0;j<n;j++)     x[j]=rnd(&s);

    /* tiempo serial */
    double t0=now_sec(); matvec_serial(A,x,yref,m,n); double Tserial=now_sec()-t0;

    /* tiempo paralelo (T hilos) */
    pthread_t* th=malloc((size_t)T*sizeof*pthread_t);
    Ctx* ctx=malloc((size_t)T*sizeof*ctx);
    t0=now_sec();
    for(int k=0;k<T;k++){
        ctx[k]=(Ctx){.A=A,.x=x,.y=y,.m=m,.n=n,.T=T,.tid=k};
        pthread_create(&th[k],NULL,worker,&ctx[k]);
    }
    for(int k=0;k<T;k++) pthread_join(th[k],NULL);
    double Tpar=now_sec()-t0;

    /* eficiencia E = Tserial / (T * Tpar) */
    double E = Tserial / ( (double)T * Tpar );

    printf("# m=%d n=%d T=%d seed=%llu\n", m,n,T,(unsigned long long)seed);
    printf("T_serial=%.6f s   T_parallel=%.6f s   Efficiency=%.6f\n", Tserial, Tpar, E);

    free(ctx); free(th); free(yref); free(y); free(x); free(A);
    return 0;
}
