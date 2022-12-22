#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void
RecvNumberFrom(int rank, int coord[2], int *result, MPI_Comm cart) {
    MPI_Status status;
    int from_rank;
    MPI_Cart_rank(cart, coord, &from_rank);

    MPI_Recv(result, 1, MPI_INT, from_rank, rank, cart, &status);
}

void
ISendNumberTo(int rank, int coord[2], int number, MPI_Comm cart) {
    MPI_Request request;
    int to_rank;
    MPI_Cart_rank(cart, coord, &to_rank);

    MPI_Isend(&number, 1, MPI_INT, to_rank, rank, cart, &request);
}

void
mpi_bcast_to_neighbours(int rank, int coord[2], int number, int dims[2], MPI_Comm cart) {
    if (coord[0] > 0) {
        coord[0] -= 1;
        ISendNumberTo(rank, coord, number, cart);
        coord[0] += 1;
    }
    if (coord[1] > 0) {
        coord[1] -= 1;
        ISendNumberTo(rank, coord, number, cart);
        coord[1] += 1;
    }
    if (coord[0] < dims[0]) {
        coord[0] += 1;
        ISendNumberTo(rank, coord, number, cart);
        coord[0] -= 1;
    }
    if (coord[1] < dims[1]) {
        coord[1] += 1;
        ISendNumberTo(rank, coord, number, cart);
        coord[1] -= 1;
    }
}

void
mpi_reduce_from_neighbours(int rank, int coord[2], int *result, int dims[2], void (*reduce) (int *, int *, int *), MPI_Comm cart) {
    int received = 0;
    if (coord[0] > 0) {
        coord[0] -= 1;
        RecvNumberFrom(rank, coord, &received, cart);
        reduce(result, &received, result);
        coord[0] += 1;
    }
    if (coord[1] > 0) {
        coord[1] -= 1;
        RecvNumberFrom(rank, coord, &received, cart);
        reduce(result, &received, result);
        coord[1] += 1;
    }
    if (coord[0] < dims[0]) {
        coord[0] += 1;
        RecvNumberFrom(rank, coord, &received, cart);
        reduce(result, &received, result);
        coord[0] -= 1;
    }
    if (coord[1] < dims[1]) {
        coord[1] += 1;
        RecvNumberFrom(rank, coord, &received, cart);
        reduce(result, &received, result);
        coord[1] -= 1;
    }
}

void
mpi_all_reduce(int number, int dims[2], MPI_Comm cart) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int coords[2];
    MPI_Cart_coords(cart, rank, 2, coords);

    int max_pulses = dims[0] + dims[1]
    for (int i = 0; i < max_pulses; ++i) {
        mpi_bcast_to_neighbours(rank, coord, number, dims, cart);
        mpi_reduce_from_neighbours(rank, coord, &number, dims, cart);
    }
}

void
max_reduce(int *num1_ptr, int *num2_ptr, int *result) {
    int num1 = *num1_ptr;
    int num2 = *num2_ptr;

    if (num1 > num2) {
        *result = num1;
    } else {
        *result = num2;
    }
}

int
main(int argc, char *argv[]) {
    int size;
    const int dims[2] = {5, 5};
    const int periods[2] = {0, 0};
    MPI_Comm cart;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Cart_create(MPI_COMM_WORLD, 2 /*ndims*/, dims, periods, 0 /*reorder*/, &cart);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    mpi_all_reduce(rank, dims, cart)

    MPI_Finalize();
    return 0;
}