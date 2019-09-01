#include <stdlib.h>
#include "life.h"
#include "helper.h"
#include "gui.h"

void simulate(int height, int width, int grid[height][width], int num_iterations)
{
  /*
   */

  save_to_file(height, width, grid, "grid_seq_before");

  // Make torus initially - evolve calls this after every iteration
  copy_edges(height, width, grid);

  for (int i = 0; i < num_iterations; i++)
  {
    evolve(height, width, grid);
    if (global_show_gui)
      gui_draw(height, width, grid[0]);
  }

  save_to_file(height, width, grid, "grid_seq_after");
}
