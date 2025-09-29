#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static int is_pow2(int x){ return x && ((x & (x-1)) == 0); }
static int highest_pow2_le(int x){ int p=1; while ((p<<1) <= x) p <<= 1; return p; }

int main(int argc, char** argv){
MPI_Init(&argc,&argv);
int rank,p; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&p);


// Valor local a sumar: por defecto rank+1 (fácil de verificar); si pasas un entero, usa ese.
long long local = (argc >= 2) ? atoll(argv[1]) : (long long)(rank + 1);

// 1) “Compactar”: si p NO es potencia de dos, plegar los ranks extra sobre [0..active-1]
int active = highest_pow2_le(p);                // mayor potencia de 2 <= p
int twin = -1;

if (rank >= active) {
    // Enviar mi contribución al "gemelo" rank-active y salir
    twin = rank - active;
    MPI_Send(&local, 1, MPI_LONG_LONG, twin, 100, MPI_COMM_WORLD);
    // Esperar el resultado final para tener también la suma global (estilo all-reduce)
    MPI_Recv(&local, 1, MPI_LONG_LONG, twin, 200, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // Imprimir solo si quieres ver que TODOS lo tienen; aquí deja que imprima rank 0
    MPI_Finalize();
    return 0;
} else {
    // Si tengo un gemelo plegado, acumulo su contribución
    twin = rank + active;
    if (twin < p) {
        long long add=0;
        MPI_Recv(&add, 1, MPI_LONG_LONG, twin, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local += add;
    }
}

// 2) Butterfly all-reduce en el grupo activo (tamaño potencia de dos)
//    En cada fase d, intercambio con partner = rank ^ (1<<d) y acumulo.
for (int d = 0; (1<<d) < active; d++){
    int partner = rank ^ (1<<d);
    long long recv_val = 0;
    MPI_Sendrecv(&local, 1, MPI_LONG_LONG, partner, 300+d,
                 &recv_val, 1, MPI_LONG_LONG, partner, 300+d,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    local += recv_val;
}

// Ahora TODOS los ranks en [0..active-1] tienen la suma global.
// 3) “Descompactar”: devolver el resultado a los gemelos plegados (si existían)
if (twin < p) {
    MPI_Send(&local, 1, MPI_LONG_LONG, twin, 200, MPI_COMM_WORLD);
}

// Imprime una sola vez
if (rank == 0) {
    printf("Suma global = %lld (p=%d)\n", local, p);
}

MPI_Finalize();
return 0;


}
