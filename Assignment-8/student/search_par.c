#include <string.h>
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include "search.h"
#include "helper.h"

void search_text(char *text, int num_lines, int line_length, char *search_string, int *occurences)
{
  /*
    Counts occurences of substring "search_string" in "text". "text" contains multiple lines and each line
    has been placed at text + line_length * num_lines since line length in the original text file can vary.
    "line_length" includes space for '\0'.

    Writes result at location pointed to by "occurences".

    *************************** PARALLEL VERSION **************************
    NOTE: For the parallel version, distribute the lines to each processor. You should only write
    to "occurences" from the root process and only access the text pointer from the root (all other processes
    call this function with text = NULL) 
  */

  int rank, num_procs;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  /*
  int lines_per_proc = num_lines / num_procs;

  if(rank==0){
    MPI_Send(text, lines_per_proc * line_length, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
  }
  */

  int count = 0;
  int lines_per_proc = num_lines / num_procs;

  char *buffer = malloc(lines_per_proc * line_length * sizeof(char));

  MPI_Scatter(text, lines_per_proc * line_length, MPI_CHAR,
              buffer, lines_per_proc * line_length, MPI_CHAR,
              0, MPI_COMM_WORLD);

  for (int i = 0; i < lines_per_proc; i++)
  {
    count += count_occurences(&buffer[i * line_length], search_string);
  }

  MPI_Send(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

  if (rank == 0)
  {
    int running_count = 0;
    for (int j = 0; j < num_procs; j++)
    {
      MPI_Recv(&count, 1, MPI_INT, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      running_count += count;
    }

    *occurences = running_count;
  }

  free(buffer);
}
