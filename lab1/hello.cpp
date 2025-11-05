#include <iostream>
#include "mpi.h"

int main4(int argc, char** argv) {
    int rank, size;

    // Initialize MPI
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        std::cerr << "MPI_Init failed" << std::endl;
        return 1;
    }

    // Get rank and size
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Synchronize processes
    MPI_Barrier(MPI_COMM_WORLD);

    // Print in order by rank
    for (int i = 0; i < size; ++i) {
        if (rank == i) {
            std::cout << "Hello from process " << rank << " of " << size << std::endl;
            std::cout.flush();  // Force output
        }
        MPI_Barrier(MPI_COMM_WORLD);  // Wait for each rank to print
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
