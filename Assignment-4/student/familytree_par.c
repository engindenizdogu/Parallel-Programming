#include "familytree.h"

int process_node(tree *node)
{
    if (node == NULL)
        return 0;

    int father_iq, mother_iq;

#pragma omp task
    father_iq = process_node(node->father);

#pragma omp task
    mother_iq = process_node(node->mother);

    node->IQ = compute_IQ(node->data, father_iq, mother_iq);
    genius[node->id] = node->IQ;

    return node->IQ;
}

int traverse(tree *node, int num_threads)
{
#pragma omp parallel
#pragma omp single
    process_node(node);
    
    return 0;
}
