#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "nbody.h"
#include <mpi.h>

#define SIZE 500
#define MAX_LEAVING 128
#define NUM_LEAVES 31

// Used for data transfers during MPI_Allgather
typedef struct Container
{
    int dest[MAX_LEAVING];
    float x[SIZE];
    float y[SIZE];
    float vx[SIZE];
    float vy[SIZE];
    float mass[SIZE];
    int count;
    int timestamp_high_res;
} Container;

void simulate_par(int steps, int num_leaves, Node *root)
{
    // You do not need to call MPI_Init and MPI_Finalize, this is already done.

    // root is the root node of the whole domain. The data will be initialised in each process, so
    // you do not need to copy it at the start. However, the main process (rank 0) has to have the
    // full data at the end of the simulation, so you will need to collect it.

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // These will be used in MPI_Allgather
    Container container; // Temporary container that carries data
    Container container_buffer[NUM_LEAVES];

    Container traveller; // Store information of nodes that are moving to other nodes
    Container travel_buffer[NUM_LEAVES];

    /* Create a custom MPI datatype */
    int count = 8;
    int blocklens[8] = {MAX_LEAVING, SIZE, SIZE, SIZE, SIZE, SIZE, 1, 1};
    MPI_Datatype datatypes[8] = {MPI_INT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_INT, MPI_INT};
    MPI_Datatype MPI_Container;
    MPI_Aint disp[8];

    disp[0] = offsetof(Container, dest);
    disp[1] = offsetof(Container, x);
    disp[2] = offsetof(Container, y);
    disp[3] = offsetof(Container, vx);
    disp[4] = offsetof(Container, vy);
    disp[5] = offsetof(Container, mass);
    disp[6] = offsetof(Container, count);
    disp[7] = offsetof(Container, timestamp_high_res);

    MPI_Type_create_struct(count, blocklens, disp, datatypes, &MPI_Container);
    MPI_Type_commit(&MPI_Container);

    // No need to search the whole tree, just start with the leaf node
    Node *node = node_find(root, rank);

    // TODO: Try broadcasting from one process
    Node **nearby = malloc(sizeof(Node *) * num_leaves);
    int nearby_count = 0;
    node_find_nearby(nearby, &nearby_count, root, root); // Sequential search

    float leaving[5 * MAX_LEAVING];

    for (int step = 0; step < steps; ++step)
    {
        // Do the n-body computation
        compute_acceleration(node, root);
        int left = move_objects(node, leaving);

        traveller.count = 0; // Reset traveller count

        // Re-distribute all the objects that have left their original boundary
        for (int i = 0; i < left; ++i)
        {
            for (int n = 0; n < nearby_count; ++n)
            {
                if (node_contains_point(nearby[n], leaving[5 * i], leaving[5 * i + 1]))
                {
                    traveller.dest[traveller.count] = nearby[n]->id;
                    traveller.x[traveller.count] = leaving[5 * i];
                    traveller.y[traveller.count] = leaving[5 * i + 1];
                    traveller.vx[traveller.count] = leaving[5 * i + 2];
                    traveller.vy[traveller.count] = leaving[5 * i + 3];
                    traveller.mass[traveller.count] = leaving[5 * i + 4];
                    traveller.count += 1;

                    break;
                }
            }

            // If the object is not within the boundary of any node, it has left
            // the domain and will be discarded.
        }

        travel_buffer[0] = traveller;
        MPI_Allgather(travel_buffer, 1, MPI_Container, travel_buffer, 1, MPI_Container, MPI_COMM_WORLD);

        for (int i = 0; i < NUM_LEAVES; i++)
        {
            if (travel_buffer[i].count > 0)
            {
                for (int j = 0; j < travel_buffer[i].count; j++)
                {
                    if (travel_buffer[i].dest[j] == rank)
                    {
                        node->leaf.x[node->leaf.count] = travel_buffer[i].x[j];
                        node->leaf.y[node->leaf.count] = travel_buffer[i].y[j];
                        node->leaf.vx[node->leaf.count] = travel_buffer[i].vx[j];
                        node->leaf.vy[node->leaf.count] = travel_buffer[i].vy[j];
                        node->leaf.mass[node->leaf.count] = travel_buffer[i].mass[j];
                        node->leaf.count += 1;
                    }
                }
            }
        }

        // Prepare the container
        for (int j = 0; j < node->leaf.count; j++)
        {
            container.x[j] = node->leaf.x[j];
            container.y[j] = node->leaf.y[j];
            container.vx[j] = node->leaf.vx[j];
            container.vy[j] = node->leaf.vy[j];
            container.mass[j] = node->leaf.mass[j];
        }

        container.count = node->leaf.count;
        container.timestamp_high_res = node->leaf.timestamp_high_res;

        // Gather results
        container_buffer[0] = container;
        MPI_Allgather(container_buffer, 1, MPI_Container, container_buffer, 1, MPI_Container, MPI_COMM_WORLD);

        // Update nodes with new values
        for (int i = 0; i < NUM_LEAVES; i++)
        {
            if (i != rank)
            {
                Node *temp_node = node_find(root, i);

                for (int j = 0; j < container_buffer[i].count; j++)
                {
                    temp_node->leaf.x[j] = container_buffer[i].x[j];
                    temp_node->leaf.y[j] = container_buffer[i].y[j];
                    temp_node->leaf.vx[j] = container_buffer[i].vx[j];
                    temp_node->leaf.vy[j] = container_buffer[i].vy[j];
                    temp_node->leaf.mass[j] = container_buffer[i].mass[j];
                }

                temp_node->leaf.count = container_buffer[i].count;
                temp_node->leaf.timestamp_high_res = container_buffer[i].timestamp_high_res;
            }
        }

        // Update bookkeeping data
        node_update_low_res(root);
    }
}
