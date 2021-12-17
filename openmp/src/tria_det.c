#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <omp.h>

#define N_RUNS 10
#define N_MATRIX_LENS 12
#define SEED 42
#define MAX_DET_VALUE 10.0

#define MASTER_RANK 0
#define NO_TAG 0

#define EPS 1e-07
#define ABS(a) ((a) < 0 ? -(a) : (a))

int
double_close(double num1, double num2)
{
    return ABS(num1 - num2) < EPS;
}

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


void
init_matrix(double *matrix, size_t len, double max_val)
{
    for (size_t row_idx = 0; row_idx < len; ++row_idx) {
        for (size_t col_idx = 0; col_idx < len; ++col_idx) {
            matrix[row_idx * len + col_idx] = ((double)rand() / (double)RAND_MAX) * max_val;
        }
    }
}

double *
alloc_matrix(size_t len)
{
    return malloc(sizeof(double) * len * len);
}

void
print_matrix(double *matrix, size_t len)
{
    printf("\n");
    for (int row_idx = 0; row_idx < len; ++row_idx) {
        for (int col_idx = 0; col_idx < len; ++col_idx) {
            printf("%f ", matrix[row_idx * len + col_idx]);
        }
        printf("\n");
    }
}

double
det(double *matrix, size_t len)
{
    double *matrix_copy = alloc_matrix(len);
    memcpy(matrix_copy, matrix, len * len * sizeof(double));

    double det = 1.0;
    for (int diag_idx = 0; diag_idx < len; ++diag_idx) {
        int diag_row_offset = len * diag_idx;
        int curr_row_len = len - diag_idx;
        double *non_zero_diag_row = matrix_copy + diag_row_offset + diag_idx;

        // reset diagonal element to 1.0
        double diag_elem = *non_zero_diag_row;
        mult_row(non_zero_diag_row, 1.0 / diag_elem, curr_row_len);
        det *= diag_elem;

        // reset elements under diagonal to 0
        int col_idx = diag_idx;
        for (int row_idx = diag_idx + 1; row_idx < len; ++row_idx) {
            int row_offset = row_idx * len;
            double *non_zero_row = matrix_copy + row_offset + col_idx;

            double elem = *non_zero_row;
            mult_row(non_zero_row, -1.0 / elem, curr_row_len);
            det *= -1.0 * elem;

            add_row(non_zero_row, non_zero_diag_row, curr_row_len);
        }
    }

    free(matrix_copy);
    return det;
}

double
omp__det(double *matrix, size_t len, int threads)
{
    int diag_idx, col_idx, row_idx;
    double *matrix_copy = alloc_matrix(len), *non_zero_diag_row, *non_zero_row;
    memcpy(matrix_copy, matrix, len * len * sizeof(double));

    double det = 1.0, diag_elem, elem;
#pragma omp parallel for private(diag_idx, row_idx, col_idx, diag_elem, elem) shared(matrix_copy, non_zero_row, non_zero_diag_row, det) num_threads(threads)
    for (diag_idx = 0; diag_idx < len; ++diag_idx) {
        non_zero_diag_row = matrix_copy + len * diag_idx + diag_idx;

        // reset diagonal element to 1.0
        diag_elem = *non_zero_diag_row;
        mult_row(non_zero_diag_row, 1.0 / diag_elem, len - diag_idx);
        det *= diag_elem;

        // reset elements under diagonal to 0
        col_idx = diag_idx;
        for (row_idx = diag_idx + 1; row_idx < len; ++row_idx) {
            non_zero_row = matrix_copy + row_idx * len + col_idx;

            elem = *non_zero_row;
            mult_row(non_zero_row, -1.0 / elem, len - diag_idx);
            det *= -1.0 * elem;

            add_row(non_zero_row, non_zero_diag_row, len - diag_idx);
        }
    }

    free(matrix_copy);
    return det;
}

int
main(int argc, char **argv)
{
    srand(SEED);

    size_t n[N_MATRIX_LENS] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};

    double timer_omp, avg_time, true_det, parallel_det;
    double *matrix;

    int threads;
    sscanf(argv[1], "%d", &threads);

    printf("running on %d threads\n", threads);
    printf("<OUTPUT>");

    size_t len;
    for (int i = 0; i < N_MATRIX_LENS; i++) {
        len = n[i];

        matrix = alloc_matrix(len);
        init_matrix(matrix, len, MAX_DET_VALUE / len / len);
        true_det = det(matrix, len);

        avg_time = 0.0;
        for (int k = 0; k < N_RUNS; k++) {
            timer_omp = omp_get_wtime();

            if (threads < 2) {
                parallel_det = det(matrix, len);
            } else {
                parallel_det = omp__det(matrix, len, threads);
            }

            if (!double_close(parallel_det, true_det)) {
                printf("WRONG: (parallel) %f != %f (true)\n", parallel_det, true_det);
            }

            avg_time += omp_get_wtime() - timer_omp;
        }
        avg_time /= (double)N_RUNS;

        printf("%zu\t%d\t%f\n", len, threads, avg_time);
        free(matrix);
    }

    printf("<OUTPUT>");
    return 0;
}
