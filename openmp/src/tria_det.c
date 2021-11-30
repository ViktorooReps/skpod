#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <omp.h>

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
det(double **matrix, size_t len, size_t threads)
{
    double det = 1.0;
    
    int r_idx, c_idx;
#pragma omp parallel for private(r_idx, c_idx) shared(matrix, det) num_threads(threads)
    for (r_idx = 0; r_idx < len; ++r_idx) {
        for (c_idx = 0; c_idx < r_idx; ++c_idx) {
            double elem = matrix[r_idx][c_idx];
            double inv_elem = 1.0 / elem;

            mult_row(matrix[r_idx], -1.0 * inv_elem, len);
            det *= -1.0 * elem;

            add_row(matrix[r_idx], matrix[c_idx], len);
        }

        double diag_elem = matrix[r_idx][r_idx];
        double inv_diag_elem = 1.0 / diag_elem;

        mult_row(matrix[r_idx], inv_diag_elem, len);
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
main()
{
    srand(time(NULL));
    double **matrix;
    int n[11] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    int threads[8] = {1, 2, 4, 8, 16, 32, 64, 128};
    double timer_omp, avg_time, maxval;
    printf("size\tn_thread\taverage_time\n");
    for (int i = 0; i < 11; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            avg_time = 0.0;
            maxval = 100000.0 / n[i] / n[i];
            int runs = 5 * (11 - i);
            for (int k = 0; k < runs; k++)
            {
                matrix = init_matrix(n[i], maxval);

                timer_omp = omp_get_wtime();
                double d = det(matrix, n[i], threads[j]);
                avg_time += omp_get_wtime() - timer_omp;
            }
            avg_time /= (double)runs;
            printf("%d\t%d\t%f\n", n[i], threads[j], avg_time);
        }
        free_matrix(matrix, n[i]);
    }
    return 0;
}