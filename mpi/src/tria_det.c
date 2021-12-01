#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <mpi.h>

#define N_RUNS 1
#define N_MATRIX_LENS 8
#define SEED 42
#define MAX_DET_VALUE 10000.0
#define MASTER_RANK 0

void
mult_row_from_idx(double *row, double num, size_t row_len, size_t from_idx)
{
    for (int i = from_idx; i < row_len; ++i) {
        row[i] *= num;
    }
}

void
add_row_from_idx(double *row_dest, double *row_to_add, size_t row_len, size_t from_idx)
{
    for (int i = from_idx; i < row_len; ++i) {
        row_dest[i] += row_to_add[i];
    }
}

double
det(double **matrix, size_t len)
{
    double det = 1.0;
    for (int diag_idx = 0; diag_idx < len; ++diag_idx) {
        // reset diagonal element to 1.0
        double diag_elem = matrix[diag_idx][diag_idx];
        mult_row_from_idx(matrix[diag_idx], 1.0 / diag_elem, len, diag_idx);
        det *= diag_elem;

        // reset elements under diagonal to 0
        int col_idx = diag_idx;
        for (int row_idx = diag_idx + 1; row_idx < len; ++row_idx) {
            double elem = matrix[row_idx][col_idx];
            mult_row_from_idx(matrix[row_idx], -1.0 / elem, len, col_idx);
            det *= -1.0 * elem;
            add_row_from_idx(matrix[row_idx], matrix[diag_idx], len, col_idx);
        }
    }
    return det
}


double
mpi__det(double **matrix, size_t len, size_t threads, int rank)
{
    double det = 1.0, res = 1.0;

    double *diag_row = malloc(sizeof(double) * len);
    for (int diag_idx = 0; diag_idx < len; ++diag_idx) {
        if (!rank) {
            diag_row = matrix[diag_idx];
            double diag_elem = diag_row[diag_idx];
            mult_row_from_idx(diag_row, 1.0 / diag_elem, len, diag_idx);
            det *= diag_elem;
        }

        MPI_Bcast(diag_row, len, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        int slave_threads = threads - 1;

        if (!rank) {
            int total_requests = (len - diag_idx - 1) * 2;
            MPI_Request *requests = malloc(sizeof(MPI_Request) * total_requests);

            int curr_req_idx = 0, dest;
            // send data to slave processes
            for (int row_idx = diag_idx + 1; row_idx < len; ++row_idx) {
                dest = row_idx % slave_threads + 1;
                MPI_Isend(matrix[row_idx], len, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, requests + curr_req_idx);
                curr_req_idx += 1;
            }
            // receive data from slave processes
            for (int row_idx = diag_idx + 1; row_idx < len; ++row_idx) {
                dest = row_idx % slave_threads + 1;
                MPI_Irecv(matrix[row_idx], len, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, requests + curr_req_idx);
                curr_req_idx += 1;
            }
            // wait for requests completion
            for (int req_idx = 0; req_idx < total_requests; ++req_idx) {
                MPI_Wait(requests + req_idx, MPI_STATUS_IGNORE);
            }

            free(requests);
        } else {
            int total_requests = len / slave_threads + (int)(rank < len);
            MPI_Request *requests = malloc(sizeof(MPI_Request) * total_requests);

            int curr_req = 0;
            int col_idx = diag_idx;
            int assigned_row = rank;
            double *compute_row = malloc(sizeof(double) * len);
            while (assigned_row < len) {
                // receive data from master process
                MPI_Recv(compute_row, len, MPI_DOUBLE, MASTER_RANK, 0, MPI_COMM_WORLD);

                // reset (assigned_row, col_idx) element to zero
                double elem = compute_row[col_idx];
                mult_row_from_idx(compute_row, -1.0 / elem, len, col_idx);
                det *= -1.0 * elem;
                add_row_from_idx(compute_row, diag_row, len, col_idx);

                // send modified data back to master
                MPI_Isend(compute_row, len, MPI_DOUBLE, MASTER_RANK, 0, MPI_COMM_WORLD, requests + curr_req);
                curr_req += 1;

                assigned_row += slave_threads;
            }
            // wait for requests completion
            for (int req_idx = 0; req_idx < total_requests; ++req_idx) {
                MPI_Wait(requests + req_idx, MPI_STATUS_IGNORE);
            }

            free(compute_row);
            free(requests);
        }

    }

    free(diag_row);

    MPI_Reduce(&det, &res, 1, MPI_DOUBLE, MPI_PROD, 0, MPI_COMM_WORLD);

    return res;
}

void
init_matrix(double **matrix, size_t n, double maxval)
{
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            matrix[i][j] = ((double)rand() / (double)RAND_MAX) * maxval;
        }
    }
}

double **
create_matrix(size_t n)
{
    double **matrix = malloc(sizeof(double *) * n);
    for (size_t i = 0; i < n; i++) {
        matrix[i] = malloc(sizeof(double) * n);
    }
    return matrix;
}

void 
free_matrix(double **matrix, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        free(matrix[i]);
    }
}

int
main(int argc, char **argv)
{
    srand(SEED);

    MPI_Init(&argc, &argv);

    int threads, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &threads);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n[N_MATRIX_LENS] = {2, 4, 8, 16, 32, 64, 128, 256};

    double timer_mpi, avg_time, maxval;
    if (!rank) {
        printf("<OUTPUT>");
    }

    for (int i = 0; i < 10; i++) {
        double **matrix = create_matrix(n[i]);
        avg_time = 0.0;
        maxval = MAX_DET_VALUE / n[i] / n[i];
        for (int k = 0; k < N_RUNS; k++) {
            if (!rank) {
                init_matrix(matrix, n[i], maxval);
            }

            MPI_Barrier(MPI_COMM_WORLD);

            timer_mpi = MPI_Wtime();

            double d
            if (threads == 1) {
                d = det(matrix, n[i]);
            } else {
                d = mpi__det(matrix, n[i], threads, rank);
            }

            avg_time += MPI_Wtime() - timer_mpi;
        }
        avg_time /= (double)N_RUNS;

        if (!rank) {
            printf("%d\t%d\t%f\n", n[i], threads, avg_time);
        }
        free_matrix(matrix, n[i]);
    }
    if (!rank) {
        printf("<OUTPUT>");
    }
    MPI_Finalize();
    return 0;
}
