#ifndef _X_CONV_SEQ
#define _X_CONV_SEQ
#include <iostream>
#include <vector>
#include <assert.h>
#include <cmath>
#include <png++/png.hpp>

using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;

Image x_convolution(Image &image, Matrix &filter, int num_threads)
{
    assert(image.size() == 3 && filter.size() != 0);

    int height = image[0].size();
    int width = image[0][0].size();
    int filterHeight = filter.size();
    int filterWidth = filter[0].size();
    int newImageHeight = height - filterHeight + 1;
    int newImageWidth = width - filterWidth + 1;
    int d, i, j, h, w;

    Image newImage(3, Matrix(newImageHeight, Array(newImageWidth)));

    for (i = 0; i < newImageHeight; i++)
    {
        for (j = 0; j < newImageWidth; j++)
        {
            for (h = i; h < i + filterHeight; h++)
            {
                for (w = j; w < j + filterWidth; w++)
                {
                    for (d = 0; d < 3; d++)
                    {
                        newImage[d][i][j] += filter[h - i][w - j] * image[d][h][w];
                    }
                }
            }
        }
    }
    return newImage;
}

#endif
