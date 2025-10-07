#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdatomic.h>

static long long n = 0;
static int thread_count = 0;
static double sum = 0.0;

// turno global 0..thread_count-1
static atomic_int flag = 0;

void* Thread_sum(void* rank){
    long my_rank = (long)rank;

    long long my_n       = n / thread_count;
    long long my_first_i = my_n * my_rank;
    long long my_last_i  = my_first_i + my_n;
    // último hilo absorbe residuo
    if (my_rank == thread_count - 1) my_last_i += n % thread_count;

    double factor = (my_first_i % 2 == 0) ? 1.0 : -1.0;

    for (long long i = my_first_i; i < my_last_i; i++, factor = -factor) {
        // Busy-wait (espera activa) por TURNO en CADA iteración
        while (atomic_load_explicit(&flag, memory_order_acquire) != my_rank) { /* spin */ }
        sum += factor / (2.0*i + 1.0);
        atomic_store_explicit(&flag, (my_rank + 1) % thread_count, memory_order_release);
    }
    return NULL;
}

int main(int argc, char **argv){
    if (argc < 3){ fprintf(stderr,"Uso: %s N T\n", argv[0]); return 1; }
    n = strtoll(argv[1], NULL, 10);
    thread_count = atoi(argv[2]);
    if (thread_count <= 0){ fprintf(stderr,"T>0\n"); return 1; }

    pthread_t *th = calloc(thread_count, sizeof(*th));
    sum = 0.0; atomic_store(&flag, 0);

    for (long t = 0; t < thread_count; t++)
        pthread_create(&th[t], NULL, Thread_sum, (void*)t);
    for (int t = 0; t < thread_count; t++)
        pthread_join(th[t], NULL);

    printf("Busy-wait DENTRO: N=%lld T=%d  pi=%.12f\n", n, thread_count, 4.0*sum);
    free(th);
    return 0;
}