#define _POSIX_C_SOURCE 200809L
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { char** lines; int n; int T; int tid; } Args;

void* worker(void* a_) {
    Args* a = (Args*)a_;
    for (int idx = a->tid; idx < a->n; idx += a->T) {
        char* line = a->lines[idx];
        if (!line) continue;
        char* save = NULL;
        char* tok = strtok_r(line, " \t\n", &save);   // versión segura
        while (tok) {
            printf("[T%02d L%03d] %s\n", a->tid, idx, tok);
            tok = strtok_r(NULL, " \t\n", &save);
        }
    }
    return NULL;
}

int main(int argc, char** argv){
    if (argc<2){fprintf(stderr,"uso: %s hilos < texto\n",argv[0]);return 1;}
    int T = atoi(argv[1]);
    size_t cap = 0, n = 0, len = 0;
    char* buf = NULL; char** lines = NULL;

    while (getline(&buf,&len,stdin) != -1) {
        if (n%256==0) lines = (char**)realloc(lines,(n+256)*sizeof(char*));
        lines[n++] = strdup(buf); // cada hilo tokeniza su copia
    }
    free(buf);

    pthread_t* th=(pthread_t*)malloc((size_t)T*sizeof(pthread_t));
    Args* a=(Args*)malloc((size_t)T*sizeof(Args));
    for (int i=0;i<T;i++){ a[i]=(Args){.lines=lines,.n=(int)n,.T=T,.tid=i};
        pthread_create(&th[i],NULL,worker,&a[i]); }
    for (int i=0;i<T;i++) pthread_join(th[i],NULL);

    for (size_t i=0;i<n;i++) free(lines[i]); free(lines); free(a); free(th);
    return 0;
}
