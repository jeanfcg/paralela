#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/* PRNG simple y portable (xorshift64*) */
static inline uint64_t xs64(uint64_t *s){ *s ^= *s>>12; *s ^= *s<<25; *s ^= *s>>27; return *s*2685821657736338717ULL; }
static inline double urand(uint64_t *s){ return (xs64(s) >> 11) * (1.0/9007199254740992.0); } // [0,1)

int main(int argc, char** argv){
  MPI_Init(&argc,&argv);
  int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);


  if (rank==0 && argc<2){ 
      fprintf(stderr,"Uso: %s <n_tosses>\n", argv[0]); 
      MPI_Abort(MPI_COMM_WORLD,1); 
  }

  long long n_total = 0;
  if (rank==0) n_total = atoll(argv[1]);
  MPI_Bcast(&n_total,1,MPI_LONG_LONG,0,MPI_COMM_WORLD);

  long long base = n_total/size, rem = n_total%size;
  long long n_local = base + (rank<rem?1:0);

  uint64_t seed = 88172645463393265ULL ^ ((uint64_t)rank<<32) ^ (uint64_t)n_total;

  long long hits = 0;
  for(long long i=0;i<n_local;i++){
      double x = urand(&seed), y = urand(&seed);
      if (x*x + y*y <= 1.0) hits++;
  }

  long long hits_tot = 0;
  MPI_Reduce(&hits,&hits_tot,1,MPI_LONG_LONG,MPI_SUM,0,MPI_COMM_WORLD);

  if(rank==0){
      double pi = 4.0*(double)hits_tot/(double)n_total;
      printf("Aproximación de Pi con %lld lanzamientos: %.12f\n", n_total, pi);
  }

  MPI_Finalize();
  return 0;


}
