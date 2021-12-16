#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include <mpi.h>

#define N_RUNS 10
#define N_MATRIX_LENS 10
#define SEED 42
#define MAX_DET_VALUE 10000.0
#define MASTER_RANK 0

void
mult_row(double *row, double num, size_t row_len)
{
    for (int i = 0; i < row_len; ++i) {
        row[i] *= num;
    }
}

void
add_row(double *row_dest, double *row_to_add, size_t row_len)
{
    for (int i = 0; i < row_len; ++i) {
        row_dest[i] += row_to_add[i];
    }
}

double
det(double *matrix, size_t len)
{
    double det = 1.0;
    for (int diag_idx = 0; diag_idx < len; ++diag_idx) {
        int diag_row_offset = len * diag_idx;
        int curr_row_len = len - diag_idx;
        double *non_zero_diag_row = matrix + diag_row_offset + diag_idx;

        // reset diagonal element to 1.0
        double diag_elem = *non_zero_diag_row;
        mult_row(non_zero_diag_row, 1.0 / diag_elem, curr_row_len);
        det *= diag_elem;

        // reset elements under diagonal to 0
        int col_idx = diag_idx;
        for (int row_idx = diag_idx + 1; row_idx < len; ++row_idx) {
            int row_offset = row_idx * len;
            double *non_zero_row = matrix + row_offset + col_idx;

            double elem = *non_zero_row;
            mult_row(non_zero_row, -1.0 / elem, curr_row_len);
            det *= -1.0 * elem;

            add_row(non_zero_row, non_zero_diag_row, curr_row_len);
        }
    }
    return det;
}


double
mpi__det(double *matrix, size_t len, size_t threads, int rank)
{
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    double res = 1.0;

    int working_threads = ((threads > len)? len : threads);
    int *working_ranks = malloc(sizeof(int) * working_threads);

    for (int worker_idx = 0; worker_idx < working_threads; ++worker_idx) {
        working_ranks[worker_idx] = worker_idx;
    }

    MPI_Group working_group;
    MPI_Group_incl(world_group, working_threads, working_ranks, &working_group);

    MPI_Comm working_comm;
    MPI_Comm_create_group(MPI_COMM_WORLD, working_group, 0, &working_comm);

    if (working_comm != MPI_COMM_NULL) {
        int working_rank;
        MPI_Comm_rank(working_comm, &working_rank);

        // distribute rows among working processes
        int *displacements = malloc(sizeof(int) * len);
        int *send_counts = malloc(sizeof(int) * len);

        for (int row_idx = 0; row_idx < len; ++row_idx) {
            send_counts[row_idx] = len;
            displacements[row_idx] = row_idx % working_threads;
        }

        int assigned_rows = len / working_threads + (rank < len % working_threads);

        double *compute_rows = malloc(sizeof(double) * len * assigned_rows);
        double *diag_row = malloc(sizeof(double) * len);

        free(displacements);
        free(send_counts);

        MPI_Scatterv(matrix, send_counts, displacements,
                     MPI_DOUBLE, compute_rows, assigned_rows,
                     MPI_DOUBLE, MASTER_RANK, working_comm);

        // everything is ready for determinant computation
        double det = 1.0;
        double *non_zero_diag_row;
        int diag_assigned_working_rank, curr_row_len, non_zeros;

        for (int diag_idx = 0; diag_idx < len; ++diag_idx) {
            diag_assigned_working_rank = diag_idx % working_threads;
            curr_row_len = len - diag_idx;
            non_zeros = diag_idx;
            non_zero_diag_row = diag_row + non_zeros;

            if (working_rank == diag_assigned_working_rank) {
                int diag_idx_in_compute_rows = diag_idx / working_threads;
                int diag_row_offset = diag_idx_in_compute_rows * len;
                double *non_zero_compute_row = compute_rows + diag_row_offset + non_zeros;

                // reset diagonal element to 1.0
                double diag_elem = *non_zero_compute_row;
                mult_row(non_zero_compute_row, 1.0 / diag_elem, curr_row_len);
                det *= diag_elem;

                // copy computed row to buffer
                memcpy(non_zero_diag_row, non_zero_compute_row, curr_row_len * sizeof(double));
            }

            // send new diagonal row to all processes
            MPI_Bcast(diag_row + non_zeros, curr_row_len, MPI_DOUBLE, diag_assigned_working_rank, working_comm);

            // reset elements under diagonal to 0 for all assigned rows
            for (int row_idx = 0; row_idx < assigned_rows; ++row_idx) {
                int row_offset = row_idx * len;
                double *non_zero_row = compute_rows + row_offset + non_zeros;

                double elem = *non_zero_row;
                mult_row(non_zero_row, -1.0 / elem, curr_row_len);
                det *= -1.0 * elem;

                add_row(non_zero_row, non_zero_diag_row, curr_row_len);
            }
        }

        MPI_Reduce(&det, &res, 1, MPI_DOUBLE, MPI_PROD, MASTER_RANK, working_comm);

        free(diag_row);
        free(compute_row);
    }

    MPI_Group_free(&world_group);
    MPI_Group_free(&working_group);

    free(working_ranks);

    return res;
}

void
init_matrix(double *matrix, size_t len, double max_val)
{
    for (size_t i = 0; i < len; i++) {
        for (size_t j = 0; j < len; j++) {
            matrix[i * len + j] = ((double)rand() / (double)RAND_MAX) * max_val;
        }
    }
}

double *
alloc_matrix(size_t len)
{
    return malloc(sizeof(double) * len * len);
}

int
main(int argc, char **argv)
{
    srand(SEED);

    MPI_Init(&argc, &argv);

    int threads, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &threads);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t n[N_MATRIX_LENS] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};

    double timer_mpi, avg_time, true_det, parallel_det;
    double *matrix;
    if (!rank) {
        printf("running on %d processes\n", threads);
        printf("<OUTPUT>");
    }

    size_t len;
    for (int i = 0; i < N_MATRIX_LENS; i++)
        len = n[i];

        if (!rank) {
            matrix = alloc_matrix(len);
            init_matrix(matrix, len, MAX_DET_VALUE / len / len);
            true_det = det(matrix, len);
        }
        avg_time = 0.0;
        for (int k = 0; k < N_RUNS; k++) {
            MPI_Barrier(MPI_COMM_WORLD);

            timer_mpi = MPI_Wtime();

            parallel_det = ((threads < 2)? det(matrix, len) : mpi__det(matrix, len, threads, rank));
            if (!rank && !double_close(parallel_det, true_det)) {
                printf("WRONG: (parallel) %d != %d (true)\n", parallel_det, true_det);
                break;
            }

            avg_time += MPI_Wtime() - timer_mpi;
        }
        avg_time /= (double)N_RUNS;

        if (!rank) {
            printf("%d\t%d\t%f\n", len, threads, avg_time);
            free(matrix);
        }
    }

    if (!rank) {
        printf("<OUTPUT>");
    }

    MPI_Finalize();
    return 0;
}

