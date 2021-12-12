#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <mpi.h>

#define N_RUNS 10
#define N_MATRIX_LENS 10
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
    return det;
}


double
mpi__det(double **matrix, size_t len, size_t threads, int rank)
{
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    double det = 1.0, res = 1.0;

    double *diag_row;
    double *compute_row;

    if (rank) {
        compute_row = malloc(sizeof(double) * len);
        diag_row = malloc(sizeof(double) * len);
    }

    for (int diag_idx = 0; diag_idx < len; ++diag_idx) {
        if (!rank) {
            diag_row = matrix[diag_idx];
            double diag_elem = diag_row[diag_idx];
            mult_row_from_idx(diag_row, 1.0 / diag_elem, len, diag_idx);
            det *= diag_elem;
        }

        int slave_threads = threads - 1;
        int rows_to_process = len - (diag_idx + 1);
        int working_threads = ((slave_threads > rows_to_process)? rows_to_process : slave_threads) + 1;
        int *working_ranks = malloc(sizeof(int) * working_threads);

        for (int worker_idx = 0; worker_idx < working_threads; ++worker_idx) {
            working_ranks[worker_idx] = worker_idx;
        }

        // working group consists of slaves (less or equal then rows_to_process) and master
        MPI_Group working_group;
        MPI_Group_incl(world_group, working_threads, working_ranks, &working_group);

        MPI_Comm working_comm;
        MPI_Comm_create_group(MPI_COMM_WORLD, working_group, 0, &working_comm);

        // if process is not included in working_group it should skip rows computation
        if (working_comm != MPI_COMM_NULL) {
            slave_threads = working_threads - 1;

            MPI_Comm_rank(working_comm, &rank);
            MPI_Bcast(diag_row, len, MPI_DOUBLE, 0, working_comm);  // point of synchronization

            if (!rank) {
                // send data to slave processes
                MPI_Request request;
                int dest, curr_tag;
                for (int row_idx = diag_idx + 1; row_idx < len; ++row_idx) {
                    dest = (row_idx - 1) % slave_threads + 1;
                    curr_tag = (row_idx - diag_idx - 1) / slave_threads;

                    MPI_Isend(matrix[row_idx], len, MPI_DOUBLE, dest, curr_tag, working_comm, &request);
                    MPI_Request_free(&request);
                }

                // make recv requests from processes
                int total_requests = rows_to_process, next_diag = diag_idx + 1;
                for (int row_idx = diag_idx + 1; row_idx < len; ++row_idx) {
                    dest = (row_idx - 1) % slave_threads + 1;
                    curr_tag = (row_idx - diag_idx - 1) / slave_threads;

                    if (row_idx == next_diag) {
                        // need only next diag row to continue computation
                        MPI_Recv(matrix[row_idx], len, MPI_DOUBLE, dest, curr_tag, working_comm, MPI_STATUS_IGNORE);
                    } else {
                        MPI_Irecv(matrix[row_idx], len, MPI_DOUBLE, dest, curr_tag, working_comm, &request);
                        MPI_Request_free(&request);
                    }
                }
            } else {
                int col_idx = diag_idx;
                int assigned_row = rank;
                while (assigned_row <= diag_idx) {
                    assigned_row += slave_threads;
                }
                int curr_tag = 0;
                MPI_Request request;
                while (assigned_row < len) {
                    // receive data from master process
                    MPI_Recv(compute_row, len, MPI_DOUBLE, MASTER_RANK, curr_tag, working_comm, MPI_STATUS_IGNORE);

                    // reset (assigned_row, col_idx) element to zero
                    double elem = compute_row[col_idx];
                    mult_row_from_idx(compute_row, -1.0 / elem, len, col_idx);
                    det *= -1.0 * elem;
                    add_row_from_idx(compute_row, diag_row, len, col_idx);

                    // send modified data back to master
                    MPI_Isend(compute_row, len, MPI_DOUBLE, MASTER_RANK, curr_tag, working_comm, &request);
                    MPI_Request_free(&request);

                    assigned_row += slave_threads;
                    curr_tag += 1;
                }
            }
        }

        MPI_Group_free(&working_group);
        MPI_Comm_free(&working_comm);
        free(working_ranks);
    }

    MPI_Reduce(&det, &res, 1, MPI_DOUBLE, MPI_PROD, MASTER_RANK, MPI_COMM_WORLD);

    MPI_Group_free(&world_group);
    if (rank) {
        free(diag_row);
        free(compute_row);
    }

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

    int n[N_MATRIX_LENS] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};

    double timer_mpi, avg_time, maxval;
    if (!rank) {
        printf("running on %d processes\n", threads);
        printf("<OUTPUT>");
    }

    for (int i = 0; i < N_MATRIX_LENS; i++) {
        double **matrix;
        if (!rank) {
            matrix = create_matrix(n[i]);
        }
        avg_time = 0.0;
        maxval = MAX_DET_VALUE / n[i] / n[i];
        for (int k = 0; k < N_RUNS; k++) {
            if (!rank) {
                init_matrix(matrix, n[i], maxval);
            }

            MPI_Barrier(MPI_COMM_WORLD);

            timer_mpi = MPI_Wtime();

            double d;
            if (threads < 2) {
                // cannot use master-slave parallelization with one process :)
                d = det(matrix, n[i]);
            } else {
                d = mpi__det(matrix, n[i], threads, rank);
            }

            avg_time += MPI_Wtime() - timer_mpi;
        }
        avg_time /= (double)N_RUNS;

        if (!rank) {
            printf("%d\t%d\t%f\n", n[i], threads, avg_time);
            free_matrix(matrix, n[i]);
        }
    }
    if (!rank) {
        printf("<OUTPUT>");
    }
    MPI_Finalize();
    return 0;
}

