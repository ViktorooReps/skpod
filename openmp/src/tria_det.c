#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <omp.h>

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
det(double **matrix, size_t len, size_t threads)
{
    double det = 1.0;

    int diag_idx, c_idx;
#pragma omp parallel for private(diag_idx, c_idx) shared(matrix, det) num_threads(threads)
    for (diag_idx = 0; diag_idx < len; ++diag_idx) {
        // reset all elements before diagonal to zero
        for (c_idx = 0; c_idx < diag_idx; ++c_idx) {
            double elem = matrix[diag_idx][c_idx];
            double inv_elem = 1.0 / elem;

            mult_row_from_idx(matrix[diag_idx], -1.0 * inv_elem, len, c_idx);
            det *= -1.0 * elem;

            add_row_from_idx(matrix[diag_idx], matrix[c_idx], len, c_idx);
        }

        double diag_elem = matrix[diag_idx][diag_idx];
        double inv_diag_elem = 1.0 / diag_elem;

        mult_row_from_idx(matrix[diag_idx], inv_diag_elem, len, diag_idx);
        det *= diag_elem;
    }

    return det;
}

double **
init_matrix(int n, double maxval)
{
    double **matrix = malloc(sizeof(double *) * n);
    for (size_t i = 0; i < n; i++)
    {
        matrix[i] = malloc(sizeof(double) * n);
        for (size_t j = 0; j < n; j++)
        {
            matrix[i][j] = ((double)rand() / (double)RAND_MAX) * maxval;
        }
    }
    return matrix;
}

void
free_matrix(double **matrix, int n)
{
    for (size_t i = 0; i < n; i++)
    {
        free(matrix[i]);
    }
}

int
main(int argc, char **argv)
{
    srand(42);
    if (argc != 2) {
        return -1;
    }

    double **matrix;
    int n[10] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    int threads;
    sscanf(argv[1], "%d", &threads);
    double timer_omp, avg_time, maxval;
    printf("<OUTPUT>");
    for (int i = 0; i < 10; i++)
    {
        avg_time = 0.0;
        maxval = 100000.0 / n[i] / n[i];
        int runs = 10;
        for (int k = 0; k < runs; k++)
        {
            matrix = init_matrix(n[i], maxval);

            timer_omp = omp_get_wtime();
            double d = det(matrix, n[i], threads);
            avg_time += omp_get_wtime() - timer_omp;
        }
        avg_time /= (double)runs;
        printf("%d\t%d\t%f\n", n[i], threads, avg_time);
        free_matrix(matrix, n[i]);
    }
    printf("<OUTPUT>");
    return 0;
}
