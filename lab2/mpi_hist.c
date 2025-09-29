#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static int find_bin(double x, const double* bin_maxes, int bin_count, double min_meas) {
if (x == bin_maxes[bin_count-1]) return bin_count - 1; // incluir max en el último bin
int lo = 0, hi = bin_count - 1;
while (lo < hi) {
int mid = (lo + hi) / 2;
if (x < bin_maxes[mid]) hi = mid;
else lo = mid + 1;
}
return lo;
}

int main(int argc, char** argv) {
MPI_Init(&argc, &argv);
int rank, comm_sz;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);


if (rank == 0 && argc < 2) {
    fprintf(stderr, "Uso: %s <input.txt>\n", argv[0]);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

long long data_count = 0;
int bin_count = 0;
double min_meas = 0.0, max_meas = 0.0;
double* data = NULL;

if (rank == 0) {
    FILE* f = fopen(argv[1], "r");
    if (!f) { perror("fopen"); MPI_Abort(MPI_COMM_WORLD, 1); }
    if (fscanf(f, "%lld %d %lf %lf", &data_count, &bin_count, &min_meas, &max_meas) != 4) {
        fprintf(stderr, "Formato: data_count bin_count min_meas max_meas \\n <data...>\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (bin_count <= 0 || max_meas <= min_meas) {
        fprintf(stderr, "Parametros invalidos.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    data = (double*)malloc(sizeof(double)*data_count);
    for (long long i = 0; i < data_count; i++) {
        if (fscanf(f, "%lf", &data[i]) != 1) {
            fprintf(stderr, "Faltan datos en el input.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    fclose(f);
}

// Broadcast de metadatos
MPI_Bcast(&data_count, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
MPI_Bcast(&bin_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(&min_meas, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Bcast(&max_meas, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

// Construir bin_maxes en todos
double bin_width = (max_meas - min_meas) / bin_count;
double* bin_maxes = (double*)malloc(sizeof(double)*bin_count);
for (int b = 0; b < bin_count; b++)
    bin_maxes[b] = min_meas + bin_width*(b+1); // [min, ..) … (b+1) // filecite marker in text

// Scatterv de datos
int* sendcounts = (int*)malloc(sizeof(int)*comm_sz);
int* displs = (int*)malloc(sizeof(int)*comm_sz);
long long base = data_count / comm_sz, rem = data_count % comm_sz;
long long offset = 0;
for (int p = 0; p < comm_sz; p++) {
    long long chunk = base + (p < rem ? 1 : 0);
    sendcounts[p] = (int)chunk;
    displs[p] = (int)offset;
    offset += chunk;
}

double* local_data = (double*)malloc(sizeof(double)*sendcounts[rank]);
MPI_Scatterv(data, sendcounts, displs, MPI_DOUBLE,
             local_data, sendcounts[rank], MPI_DOUBLE,
             0, MPI_COMM_WORLD);

// Conteo local
int* local_counts = (int*)calloc(bin_count, sizeof(int));
for (int i = 0; i < sendcounts[rank]; i++) {
    double x = local_data[i];
    int b = find_bin(x, bin_maxes, bin_count, min_meas);
    if (b >= 0 && b < bin_count) local_counts[b]++;
}

// Reducción a proceso 0
int* global_counts = NULL;
if (rank == 0) global_counts = (int*)calloc(bin_count, sizeof(int));
MPI_Reduce(local_counts, global_counts, bin_count, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

// Proceso 0 imprime histograma
if (rank == 0) {
    printf("Histograma (%d bins) en rango [%.6f, %.6f):\n", bin_count, min_meas, max_meas);
    for (int b = 0; b < bin_count; b++) {
        double lo = (b == 0 ? min_meas : bin_maxes[b-1]);
        double hi = bin_maxes[b];
        printf("[%g, %g)%s: %d ", lo, hi, (b==bin_count-1 ? " (incluye max)" : ""), global_counts[b]);
        int stars = global_counts[b] > 60 ? 60 : global_counts[b];
        printf(" |");
        for (int s = 0; s < stars; s++) putchar('*');
        printf("|\n");
    }
}

free(global_counts);
free(local_counts);
free(local_data);
free(displs);
free(sendcounts);
free(bin_maxes);
free(data);

MPI_Finalize();
return 0;


}
