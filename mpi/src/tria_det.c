#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <mpi.h>

#define N_RUNS 10
#define N_MATRIX_LENS 10
#define SEED 42
#define MAX_DET_VALUE 10000.0

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

int
mpi__in_thread_group(int rank, int mpi__thread_group, int mpi__thread_group_len)
{
    int mpi__thread_group_start = mpi__thread_group * mpi__thread_group_len;
    int mpi__thread_group_end = mpi__thread_group_start + mpi__thread_group_len;

    return ((rank >= mpi__thread_group_start) && (rank < mpi__thread_group_end));
}

void
mpi__mult_row_from_idx(double *row, double num, size_t row_len, int from_idx, int mpi__thread_group, int mpi__thread_group_len, int rank)
{
    int proc_len = row_len - from_idx;
    int mpi__in_group_rank = rank - mpi__thread_group * mpi__thread_group_len;
    int mpi__new_group_len = mpi__thread_group_len % proc_len;

    if (mpi__in_thread_group(mpi__in_group_rank, 0, mpi__new_group_len)) {
        int mpi__elems_per_thread = proc_len / mpi__new_group_len;
        int mpi__start_elem = mpi__in_group_rank * mpi__elems_per_thread;
        int mpi__end_elem = mpi__start_elem + mpi__elems_per_thread;

        if (mpi__in_group_rank == mpi__new_group_len - 1) {
            mpi__end_elem = row_len - 1;
        }

        for (int elem_idx = mpi__start_elem; elem_idx < mpi__end_elem; ++elem_idx) {
            row[from_idx + elem_idx] *= num;
        }

        MPI_Bcast(row + from_idx + mpi__start_elem, mpi__end_elem - mpi__start_elem, MPI_DOUBLE, rank, MPI_COMM_WORLD);
    }
}

void
mpi__add_row_from_idx(double *row_dest, double *row_to_add, size_t row_len, int from_idx, int mpi__thread_group, int mpi__thread_group_len, int rank)
{
    int proc_len = row_len - from_idx;
    int mpi__in_group_rank = rank - mpi__thread_group * mpi__thread_group_len;
    int mpi__new_group_len = mpi__thread_group_len % proc_len;

    if (mpi__in_thread_group(mpi__in_group_rank, 0, mpi__new_group_len)) {
        int mpi__elems_per_thread = proc_len / mpi__new_group_len;
        int mpi__start_elem = mpi__in_group_rank * mpi__elems_per_thread;
        int mpi__end_elem = mpi__start_elem + mpi__elems_per_thread;

        if (mpi__in_group_rank == mpi__new_group_len - 1) {
            mpi__end_elem = row_len - 1;
        }

        for (int elem_idx = mpi__start_elem; elem_idx < mpi__end_elem; ++elem_idx) {
            row_dest[from_idx + elem_idx] += row_to_add[from_idx + elem_idx];
        }

        MPI_Bcast(row_dest + from_idx + mpi__start_elem, mpi__end_elem - mpi__start_elem, MPI_DOUBLE, rank, MPI_COMM_WORLD);
    }
}

double
mpi__det(double **matrix, size_t len, size_t threads, int rank)
{
    double det = 1.0;

    for (int diag_idx = 0; diag_idx < len; ++diag_idx) {
        MPI_Barrier(MPI_COMM_WORLD);

        // reset to 1.0 diagonal element
        double diag = matrix[diag_idx][diag_idx];
        mpi__mult_row_from_idx(matrix[diag_idx], 1.0 / diag, len, diag_idx, 0, threads, rank);
        if (!rank) {
            det *= diag;
        }

        int mpi__start_row, mpi__end_row, mpi__threads_per_row, mpi__rows_per_thread;
        if (threads > len) {
            mpi__rows_per_thread = 1;
            mpi__threads_per_row = threads / len;
            mpi__start_row = rank / mpi__threads_per_row;
            mpi__end_row = mpi__start_row + 1;
        } else {
            mpi__rows_per_thread = len / threads;
            mpi__threads_per_row = 1;
            mpi__start_row = rank * mpi__rows_per_thread;
            mpi__end_row = mpi__start_row + mpi__rows_per_thread;

            if (rank == threads - 1) {
                mpi__end_row = len - 1;
            }
        }

        int mpi__thread_group, mpi__in_group_rank, mpi__thread_group_len;
        mpi__thread_group_len = mpi__threads_per_row;
        mpi__thread_group = rank / mpi__threads_per_row;
        mpi__in_group_rank = rank - mpi__thread_group * mpi__threads_per_row;

        MPI_Barrier(MPI_COMM_WORLD);

        // reset to zero elements under diagonal
        int col_idx = diag_idx;
        for (int row_idx = mpi__start_row; row_idx < mpi__end_row; ++row_idx) {
            double elem = matrix[row_idx][col_idx];
            mpi__mult_row_from_idx(matrix[row_idx], -1.0 / elem, len, col_idx, mpi__thread_group, mpi__thread_group_len, rank);
            if (!mpi__in_group_rank) {
                det *= -1.0 * elem;
            }

            mpi__add_row_from_idx(matrix[row_idx], matrix[diag_idx], len, col_idx, mpi__thread_group, mpi__thread_group_len, rank);
        }
    }

    double res = 1.0;
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

    int n[N_MATRIX_LENS] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};

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
            double d = mpi__det(matrix, n[i], threads, rank);
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
    return 0;
}
