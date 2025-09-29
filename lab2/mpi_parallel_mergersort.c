#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static int cmp_int(const void* a, const void* b){
int x = *(const int*)a, y = *(const int*)b;
return (x>y) - (x<y);
}
static void merge_int(const int* A, int nA, const int* B, int nB, int* C){
int i=0,j=0,k=0;
while(i<nA && j<nB) C[k++] = (A[i]<=B[j]) ? A[i++] : B[j++];
while(i<nA) C[k++] = A[i++];
while(j<nB) C[k++] = B[j++];
}

int main(int argc, char** argv){
MPI_Init(&argc,&argv);
int rank,p; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&p);


if (rank==0 && argc<2){ fprintf(stderr,"Uso: %s <n_total>\n", argv[0]); MPI_Abort(MPI_COMM_WORLD,1); }

int n = 0;
if (rank==0) n = atoi(argv[1]);
MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);

if (n<=0 || n%p!=0){
    if(rank==0) fprintf(stderr,"Requisito: n>0 y divisible por comm_sz=%d\n", p);
    MPI_Abort(MPI_COMM_WORLD,1);
}

int local_n = n/p;
// Genera datos locales pseudoaleatorios, reproducibles por rank
unsigned int seed = 123456u ^ (rank*2654435761u) ^ (unsigned int)n;
int* local = (int*)malloc(sizeof(int)*local_n);
for(int i=0;i<local_n;i++){ seed = seed*1664525u + 1013904223u; local[i] = (int)(seed & 0x7fffffff); }

// Ordena localmente
qsort(local, local_n, sizeof(int), cmp_int);

// (Opcional, pedido por el enunciado) Reunir e imprimir las listas locales antes de fusionar
int* gathered = NULL;
if (rank==0) gathered = (int*)malloc(sizeof(int)*n);
MPI_Gather(local, local_n, MPI_INT, gathered, local_n, MPI_INT, 0, MPI_COMM_WORLD);
if (rank==0){
    printf("Listas locales (ordenadas por cada rank) antes del merge:\n");
    for(int r=0;r<p;r++){
        printf("  rank %d:", r);
        for(int i=0;i<local_n;i++) printf(" %d", gathered[r*local_n+i]);
        printf("\n");
    }
    free(gathered);
}

// Reducción en árbol con fusión (funciona también si p no es potencia de dos)
int curr_n = local_n;
for (int step=1; step<p; step<<=1){
    if ((rank % (2*step)) == 0){
        int partner = rank + step;
        if (partner < p){
            int recv_n=0;
            MPI_Recv(&recv_n,1,MPI_INT,partner,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            int* buf = (int*)malloc(sizeof(int)*recv_n);
            MPI_Recv(buf,recv_n,MPI_INT,partner,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

            int* merged = (int*)malloc(sizeof(int)*(curr_n + recv_n));
            merge_int(local, curr_n, buf, recv_n, merged);
            free(local); free(buf);
            local = merged; curr_n += recv_n;
        }
    }else if ((rank % (2*step)) == step){
        int partner = rank - step;
        MPI_Send(&curr_n,1,MPI_INT,partner,0,MPI_COMM_WORLD);
        MPI_Send(local,curr_n,MPI_INT,partner,0,MPI_COMM_WORLD);
        break; // este rank ya terminó
    }
}

if (rank==0){
    // 'local' contiene los n elementos globales en orden
    printf("Lista global ordenada (n=%d):\n", n);
    for(int i=0;i<n;i++){
        printf("%d%s", local[i], (i+1==n)?"\n":" ");
    }
    free(local);
}else{
    free(local);
}

MPI_Finalize();
return 0;


}
