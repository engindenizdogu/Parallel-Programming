#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include "helper.h"
#include "life.h"
#include "gui.h"

void evolve_par(int section_height, int width, int size_shift, int buffer[section_height + size_shift][width]);

void simulate(int height, int width, int grid[height][width], int num_iterations)
{
  /*
    Write your parallel solution here. You first need to distribute the data to all of the
    processes from the root. Note that you cannot naively use the evolve function used in the
    sequential version of the code - you might need to rewrite it depending on how you parallelize
    your code.

    For more details, see the attached readme file.
  */

  int rank, num_procs;

  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (num_procs == 1) // Sequential
  {
    copy_edges(height, width, grid);
    for (int i = 0; i < num_iterations; i++)
    {
      evolve(height, width, grid);
    }
  }
  else // Parallel
  {
    if (rank == 0)
    {
      copy_edges(height, width, grid);
    }

    // We are going to distribute the data row by row
    int rows_per_proc = height / num_procs;
    int remainder = height % num_procs;

    // Arrays to be used in MPI_Scatterv
    int *locations = malloc(2 * num_procs * sizeof(int));
    int *counts = malloc(num_procs * sizeof(int));
    int *displs = malloc(num_procs * sizeof(int));

    int shift = 0;
    int start, end;
    for (int i = 0; i < num_procs; i++)
    {
      start = (i * rows_per_proc) + shift;

      if (shift < remainder)
        shift += 1;

      end = ((i + 1) * rows_per_proc) + shift;

      locations[2 * i] = start; // Save 'start' and 'end' of each section
      locations[2 * i + 1] = end;
      counts[i] = (end - start) * width; // Amount of data to be sent
      displs[i] = start * width;         // Starting location of each section (displacement)
    }

    // TODO: instead of locations try -> (counts[rank] / width)
    int section_height = locations[2 * rank + 1] - locations[2 * rank]; // Height of a section
    int size_shift = ((rank == 0) || (rank == num_procs - 1) ? 1 : 2);
    int(*buffer)[width] = malloc(sizeof(int[section_height + size_shift][width])); // +2 is for ghost rows (cells)

    // TODO: instead of (section_height * width) try -> counts[rank]
    // Pad one row on top for ghost cells
    int buffer_shift = (rank == 0 ? 0 : 1);
    MPI_Scatterv(grid, counts, displs, MPI_INT, buffer + buffer_shift, section_height * width, MPI_INT, 0, MPI_COMM_WORLD);

    // Find the processes above and below
    int proc_above = ((rank == 0) ? (num_procs - 1) : (rank - 1));
    int proc_below = ((rank == num_procs - 1) ? 0 : (rank + 1));

    for (int j = 0; j < num_iterations; j++)
    {
      // Sending and receiving the ghost lines
      if (rank == 0)
      {
        MPI_Sendrecv(buffer + section_height - 1, width, MPI_INT, proc_below, 0, buffer + section_height, width, MPI_INT, proc_below, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }

      if (rank == num_procs - 1)
      {
        MPI_Sendrecv(buffer + 1, width, MPI_INT, proc_above, 0, buffer, width, MPI_INT, proc_above, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }

      if (rank > 0 && rank < num_procs - 1)
      {
        MPI_Sendrecv(buffer + 1, width, MPI_INT, proc_above, 0, buffer, width, MPI_INT, proc_above, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(buffer + section_height, width, MPI_INT, proc_below, 0, buffer + section_height + 1, width, MPI_INT, proc_below, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }

      // TODO: Parallel adjusted implementation of evolve
      evolve_par(section_height, width, size_shift, buffer);

      if (rank == num_procs - 1)
      {
        MPI_Send(buffer + section_height - 1, width, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Recv(buffer + section_height, width, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }

      if (rank == 0)
      {
        MPI_Recv(buffer, width, MPI_INT, num_procs - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(buffer + 1, width, MPI_INT, num_procs - 1, 0, MPI_COMM_WORLD);
      }
    }

    MPI_Gatherv(buffer + buffer_shift, section_height * width, MPI_INT, grid, counts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    /*
    if (rank == 0)
    {
      save_to_file(height, width, grid, "grid_par");
    }
    */

    free(locations);
    free(counts);
    free(displs);
    free(buffer);
  }
}

void evolve_par(int section_height, int width, int size_shift, int buffer[section_height + size_shift][width])
{
  int(*temp)[width] = malloc(sizeof(int[section_height + size_shift][width]));
  for (int i = 1; i < section_height + size_shift - 1; i++)
  {
    for (int j = 1; j < width - 1; j++)
    {
      int sum = buffer[i - 1][j - 1] + buffer[i - 1][j] + buffer[i - 1][j + 1] +
                buffer[i][j - 1] + buffer[i][j + 1] +
                buffer[i + 1][j - 1] + buffer[i + 1][j] + buffer[i + 1][j + 1];

      if (buffer[i][j] == 0)
      {
        // Reproduction
        if (sum == 3)
        {
          temp[i][j] = 1;
        }
        else
        {
          temp[i][j] = 0;
        }
      }
      // Alive
      else
      {
        // Stays alive
        if (sum == 2 || sum == 3)
        {
          temp[i][j] = 1;
        }
        // Dies due to under or overpopulation
        else
        {
          temp[i][j] = 0;
        }
      }
    }
  }

  // Copy boundaries
  for (int i = 1; i < section_height + size_shift - 1; i++)
  {
    // join rows together
    temp[i][0] = temp[i][width - 2];
    temp[i][width - 1] = temp[i][1];
  }

  // Copy to buffer
  for (int i = 0; i < section_height + size_shift; i++)
  {
    for (int j = 0; j < width; j++)
    {
      buffer[i][j] = temp[i][j];
    }
  }

  free(temp);
}
