#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static int is_power_of_two(int x){ return x && ((x & (x-1)) == 0); }

int main(int argc, char** argv){
MPI_Init(&argc,&argv);
int rank, p; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&p);


// Valor local a sumar: por defecto (rank+1). Si pasas un entero por argv[1], usa ese.
long long local = (argc >= 2) ? atoll(argv[1]) : (long long)(rank + 1);

// 1) Si p no es potencia de dos, "apartar" los ranks extra hacia la potencia de dos inferior
int active = 1;
while ((active << 1) <= p) active <<= 1;  // mayor potencia de 2 <= p
// Fase de "compactación" hacia ranks [0..active-1]
if (rank >= active) {
    int partner = rank - active;               // envía al partner "debajo"
    MPI_Send(&local, 1, MPI_LONG_LONG, partner, 0, MPI_COMM_WORLD);
    // Este rank ya no participa en el árbol:
    MPI_Finalize();
    return 0;
} else {
    int partner = rank + active;
    if (partner < p) {
        long long recv_val = 0;
        MPI_Recv(&recv_val, 1, MPI_LONG_LONG, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local += recv_val;
    }
}

// 2) Reducción en árbol binario para 'active' procesos (active es potencia de 2)
for (int step = 1; step < active; step <<= 1) {
    if ((rank % (2*step)) == 0) {
        // Nodo "receptor" en este nivel
        long long recv_val = 0;
        MPI_Recv(&recv_val, 1, MPI_LONG_LONG, rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local += recv_val;
    } else if ((rank % (2*step)) == step) {
        // Nodo "emisor" en este nivel
        MPI_Send(&local, 1, MPI_LONG_LONG, rank - step, 0, MPI_COMM_WORLD);
        break; // este rank ya no participa en niveles superiores
    }
}

if (rank == 0) {
    printf("Suma global = %lld (p=%d)\n", local, p);
}

// (Opcional) Comprobación con MPI_Reduce para verificar corrección:
/*
long long check=0;
MPI_Reduce(&((long long){(argc>=2)?atoll(argv[1]):(rank+1)}), &check, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
if (rank==0) printf("[Reduce]    = %lld\n", check);
*/

MPI_Finalize();
return 0;


}
