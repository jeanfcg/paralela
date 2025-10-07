#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int main(int argc, char **argv){
    if (argc < 2){ fprintf(stderr,"Uso: %s N\n", argv[0]); return 1; }
    long long n = strtoll(argv[1], NULL, 10);

    double sum = 0.0;
    for (long long i = 0; i < n; i++){
        double term = ((i & 1) ? -1.0 : 1.0) / (double)(2*i + 1);
        sum += term;
    }
    printf("Secuencial: N=%lld  pi=%.12f\n", n, 4.0*sum);
    return 0;
}