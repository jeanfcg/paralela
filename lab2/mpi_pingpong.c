#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

static void barrier2(MPI_Comm comm){
int rank; MPI_Comm_rank(comm,&rank);
int x=1, y=0;
if(rank==0){ MPI_Send(&x,1,MPI_INT,1,99,comm); MPI_Recv(&y,1,MPI_INT,1,99,comm,MPI_STATUS_IGNORE); }
else       { MPI_Recv(&y,1,MPI_INT,0,99,comm,MPI_STATUS_IGNORE); MPI_Send(&x,1,MPI_INT,0,99,comm); }
}

int main(int argc, char** argv){
MPI_Init(&argc,&argv);
int rank, p; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&p);


if(p < 2){ if(rank==0) fprintf(stderr,"Se requieren 2 procesos.\n"); MPI_Abort(MPI_COMM_WORLD,1); }
if(rank >= 2){ MPI_Finalize(); return 0; }  // solo 0 y 1 participan

size_t msg_bytes = (argc>=2)? strtoull(argv[1],NULL,10) : 0ULL;     // por defecto: 0 bytes
long long iters  = (argc>=3)? atoll(argv[2]) : 100000LL;            // por defecto: 100000
int warmup = 1000;
if (warmup > iters/10) warmup = (int)(iters/10);

char* buf = NULL;
if (msg_bytes > 0){
    buf = (char*)malloc(msg_bytes);
    for (size_t i=0;i<msg_bytes;i++) buf[i] = (char)(i & 0xFF);
}

// Sincroniza y calienta
barrier2(MPI_COMM_WORLD);
for(int i=0;i<warmup;i++){
    if(rank==0){ MPI_Send(buf, (int)msg_bytes, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                 MPI_Recv(buf, (int)msg_bytes, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); }
    else       { MPI_Recv(buf, (int)msg_bytes, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                 MPI_Send(buf, (int)msg_bytes, MPI_CHAR, 0, 0, MPI_COMM_WORLD); }
}
barrier2(MPI_COMM_WORLD);

// Tiempo de pared
double t0 = MPI_Wtime();
// Tiempo de CPU
clock_t c0 = clock();

for(long long i=0;i<iters;i++){
    if(rank==0){ MPI_Send(buf, (int)msg_bytes, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                 MPI_Recv(buf, (int)msg_bytes, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); }
    else       { MPI_Recv(buf, (int)msg_bytes, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                 MPI_Send(buf, (int)msg_bytes, MPI_CHAR, 0, 0, MPI_COMM_WORLD); }
}

clock_t c1 = clock();
double t1 = MPI_Wtime();

if(rank==0){
    double wall = t1 - t0;                          // seg (tiempo de pared)
    double cpu  = (double)(c1 - c0) / CLOCKS_PER_SEC; // seg (tiempo de CPU)
    double rtt  = wall / (double)iters;             // round-trip time
    double one_way_lat = rtt / 2.0;                 // latencia aprox. unidireccional
    double bytes_xfer = (double)msg_bytes * 2.0 * (double)iters; // ida+vuelta
    double bandwidth  = (bytes_xfer / wall) / 1e6;  // MB/s

    printf("pingpong: size=%zu bytes  iters=%lld\n", msg_bytes, iters);
    printf("MPI_Wtime: wall=%.6f s   RTT=%.3fus   one-way≈%.3fus   BW≈%.1f MB/s\n",
           wall, 1e6*rtt, 1e6*one_way_lat, bandwidth);
    printf("clock()  : cpu =%.6f s   (ticks=%ld, CLOCKS_PER_SEC=%ld)\n",
           cpu, (long)(c1-c0), (long)CLOCKS_PER_SEC);
}

free(buf);
MPI_Finalize();
return 0;


}
