#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <mpi.h>

#define TIME_LIMIT 10
#define INIT_LEN 2
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
mpi__det(double **matrix, size_t len, MPI_Comm comm)
{
    int threads, rank, err;
    MPI_Comm_size(MPI_COMM_WORLD, &threads);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double det = 1.0, res = 1.0, curr_res = 1.0;

    double *diag_row;
    double *compute_row;

    if (rank) {
        compute_row = malloc(sizeof(double) * len);
        diag_row = malloc(sizeof(double) * len);
    }

    bool restart = false;
    for (int diag_idx = 0; diag_idx < len; ++diag_idx) {
        if (restart) {
            MPIX_Comm_shrink(comm, &comm);
            MPI_Comm_size(MPI_COMM_WORLD, &threads);
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            restart = false;
            diag_idx -= 1;
        }
        if (!rank) {
            diag_row = matrix[diag_idx];
            double diag_elem = diag_row[diag_idx];
            mult_row_from_idx(diag_row, 1.0 / diag_elem, len, diag_idx);
            det *= diag_elem;
        }

        int slave_threads = threads - 1;
        int rows_to_process = len - (diag_idx + 1);
        int working_threads = ((slave_threads > rows_to_process)? rows_to_process : slave_threads) + 1;  // master + all slaves

        if (rank < working_threads) {
            slave_threads = working_threads - 1;  // recompute actual number of working slaves 

            err = MPI_Bcast(diag_row, len, MPI_DOUBLE, 0, comm);  // point of synchronization
            if (err != MPI_SUCCESS) {
                // restart iteration with smaller comm
                MPIX_Comm_shrink(comm, &comm);
                diag_idx -= 1;
                continue
            }

            if (!rank) {
                // send data to slave processes
                MPI_Request request;
                int dest, curr_tag;
                for (int row_idx = diag_idx + 1; row_idx < len; ++row_idx) {
                    dest = (row_idx - 1) % slave_threads + 1;
                    curr_tag = (row_idx - diag_idx - 1) / slave_threads;

                    err = MPI_Isend(matrix[row_idx], len, MPI_DOUBLE, dest, curr_tag, comm);
                    if (err != MPI_SUCCESS) {
                        restart = true;
                    }
                }

                // make recv requests from processes
                int total_requests = rows_to_process, next_diag = diag_idx + 1;
                for (int row_idx = diag_idx + 1; row_idx < len && !restart; ++row_idx) {
                    dest = (row_idx - 1) % slave_threads + 1;
                    curr_tag = (row_idx - diag_idx - 1) / slave_threads;

                    if (row_idx == next_diag) {
                        // need only next diag row to continue computation
                        err = MPI_Recv(matrix[row_idx], len, MPI_DOUBLE, dest, curr_tag, comm, MPI_STATUS_IGNORE);
                        if (err != MPI_SUCCESS) {
                            restart = true;
                        }
                    } else {
                        err = MPI_Irecv(matrix[row_idx], len, MPI_DOUBLE, dest, curr_tag, comm, &request);
                        if (err != MPI_SUCCESS) {
                            restart = true;
                        }
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

                while (assigned_row < len) {
                    // receive data from master process
                    err = MPI_Recv(compute_row, len, MPI_DOUBLE, MASTER_RANK, curr_tag, comm, MPI_STATUS_IGNORE);
                    if (err != MPI_SUCCESS) {
                        restart = true;
                    }

                    // reset (assigned_row, col_idx) element to zero
                    double elem = compute_row[col_idx];
                    mult_row_from_idx(compute_row, -1.0 / elem, len, col_idx);
                    det *= -1.0 * elem;
                    add_row_from_idx(compute_row, diag_row, len, col_idx);

                    // send modified data back to master
                    err = MPI_Isend(compute_row, len, MPI_DOUBLE, MASTER_RANK, curr_tag, comm);
                    if (err != MPI_SUCCESS) {
                        restart = true;
                    }

                    assigned_row += slave_threads;
                    curr_tag += 1;
                }

                err = MPI_Reduce(&det, &curr_res, 1, MPI_DOUBLE, MPI_PROD, MASTER_RANK, comm);  // synchronize current result
                if (err != MPI_SUCCESS) {
                    restart = true;
                }
                det = 1.0

                if (!restart) {
                    // if by the end of the procedure there is no need to restart, update current result
                    res *= curr_res;
                }
            }
        }
    }

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
        printf("<OUTPUT>\n");
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
                d = mpi__det(matrix, n[i], MPI_COMM_WORLD);
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
        printf("<OUTPUT>\n");
    }
    MPI_Finalize();
    return 0;
}

