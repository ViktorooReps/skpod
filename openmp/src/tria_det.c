#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include <omp.h>

#define TIME_LIMIT 3.0
#define INIT_LEN 2
#define LEN_MULTIPLIER 2
#define LEN_STEP 0
#define N_RUNS 10
#define SEED 42
#define MAX_DET_VALUE 10.0

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
    int col_idx;
    for (col_idx = 0; col_idx < row_len; ++col_idx) {
        row[col_idx] *= num;
    }
}

void
add_row(double *row_dest, double *row_to_add, size_t row_len)
{
    int col_idx;
    for (col_idx = 0; col_idx < row_len; ++col_idx) {
        row_dest[col_idx] += row_to_add[col_idx];
    }
}


void
init_matrix(double *matrix, size_t len, double max_val)
{
    int row_idx, col_idx;
    for (row_idx = 0; row_idx < len; ++row_idx) {
        for (col_idx = 0; col_idx < len; ++col_idx) {
            matrix[row_idx * len + col_idx] = ((double)rand() / (double)RAND_MAX) * max_val;
        }
    }
}

void
print_matrix(double *matrix, size_t len)
{
    printf("\n");
    int row_idx, col_idx;
    for (row_idx = 0; row_idx < len; ++row_idx) {
        for (col_idx = 0; col_idx < len; ++col_idx) {
            printf("%f ", matrix[row_idx * len + col_idx]);
        }
        printf("\n");
    }
}

double
det(double *matrix, size_t len)
{
    double *matrix_copy = malloc(sizeof(double) * len * len);
    memcpy(matrix_copy, matrix, len * len * sizeof(double));

    double det = 1.0;
    int diag_idx, row_idx;
    for (diag_idx = 0; diag_idx < len; ++diag_idx) {
        int diag_row_offset = len * diag_idx;
        int curr_row_len = len - diag_idx;
        double *non_zero_diag_row = matrix_copy + diag_row_offset + diag_idx;

        // reset diagonal element to 1.0
        double diag_elem = *non_zero_diag_row;
        mult_row(non_zero_diag_row, 1.0 / diag_elem, curr_row_len);
        det *= diag_elem;

        // reset elements under diagonal to 0
        int col_idx = diag_idx;
        for (row_idx = diag_idx + 1; row_idx < len; ++row_idx) {
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
    double *matrix_copy = malloc(sizeof(double) * len * len);
    memcpy(matrix_copy, matrix, len * len * sizeof(double));

    double det = 1.0;
    int diag_idx, row_idx;
    #pragma omp parallel shared(matrix_copy) reduction(* : det) num_threads(threads)
    {
        for (diag_idx = 0; diag_idx < len; ++diag_idx) {
            // reset diagonal element to 1.0
            #pragma omp single
            {
                double diag_elem = matrix_copy[len * diag_idx + diag_idx];
                mult_row(matrix_copy + len * diag_idx + diag_idx, 1.0 / diag_elem, len - diag_idx);
                det *= diag_elem;
            }

            #pragma omp barrier

            // reset elements under diagonal to 0
            #pragma omp for schedule(static)
            for (row_idx = diag_idx + 1; row_idx < len; ++row_idx) {
                double elem = matrix_copy[row_idx * len + diag_idx];
                mult_row(matrix_copy + row_idx * len + diag_idx, -1.0 / elem, len - diag_idx);
                det *= -1.0 * elem;

                add_row(matrix_copy + row_idx * len + diag_idx, matrix_copy + len * diag_idx + diag_idx,
                        len - diag_idx);
            }
        }
    }

    free(matrix_copy);
    return det;
}

int
main(int argc, char **argv)
{
    srand(SEED);

    double timer_omp, avg_time = 0.0, true_det, parallel_det;
    double *matrix;

    int threads;
    sscanf(argv[1], "%d", &threads);

    printf("running on %d threads\n", threads);
    printf("<OUTPUT>\n");

    size_t len = INIT_LEN;
    int run_idx;
    while (avg_time < TIME_LIMIT) {
        matrix = malloc(sizeof(double) * len * len);
        init_matrix(matrix, len, MAX_DET_VALUE / len / len);
        true_det = det(matrix, len);

        avg_time = 0.0;
        for (run_idx = 0; run_idx < N_RUNS; ++run_idx) {
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

        len = len * LEN_MULTIPLIER + LEN_STEP;
    }

    printf("<OUTPUT>\n");
    return 0;
}
