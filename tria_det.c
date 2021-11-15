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

#pragma omp parallel for private(r_idx, c_idx) shared(matrix, det) num_threads(threads)
    for (int r_idx = 0; r_idx < len; ++r_idx) {
        for (int c_idx = 0; c_idx < r_idx; ++c_idx) {
            double elem = matrix[r_idx][c_idx];
            double inv_elem = 1.0 / elem;

            mult_row(matrix[r_idx], -1.0 * inv_elem, len);
            det *= -1.0 * elem;

            add_row(matrix[r_idx], matrix[c_idx], len)
        }

        double diag_elem = matrix[r_idx][r_idx];
        double inv_diag_elem = 1.0 / diag_elem;

        mult_row(matrix[r_idx], inv_diag_elem);
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
    int n[7] = {16, 32, 64, 128, 256, 512, 1024};
    int threads[7] = {1, 2, 4, 8, 16, 32, 64};
    double timer_omp, avg_time;
    double maxval = 100000.0 / n / n / n;
    printf("size\tn_thread\taverage_time\n");
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 7; j++)
        {
            avg_time = 0.0;
            for (int k = 0; k < 5; k++)
            {
                matrix = init_matrix(n[i], maxval);

                timer_omp = omp_get_wtime();
                double d = det(matrix, n[i], threads[j]);
                avg_time += omp_get_wtime() - timer_omp;
            }
            avg_time /= 5.0;
            printf("%d\t%d\t%f\n", n[i], threads[j], avg_time);
        }
        free_matrix(matrix, n[i]);
    }
    return 0;
}
