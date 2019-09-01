#ifndef _X_CONV_PAR
#define _X_CONV_PAR
#include <iostream>
#include <vector>
#include <assert.h>
#include <cmath>
#include <png++/png.hpp>
#include <omp.h>

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

	Image newImage(3, Matrix(newImageHeight, Array(newImageWidth)));

	#pragma omp parallel num_threads(num_threads)
	{
		#pragma omp for
		for (int i = 0; i < newImageHeight; i++)
		{
			for (int j = 0; j < newImageWidth; j++)
			{
				for (int h = i; h < i + filterHeight; h++)
				{
					for (int w = j; w < j + filterWidth; w++)
					{
						#pragma atomic
						newImage[0][i][j] += filter[h - i][w - j] * image[0][h][w];
						newImage[1][i][j] += filter[h - i][w - j] * image[1][h][w];
						newImage[2][i][j] += filter[h - i][w - j] * image[2][h][w];
					}
				}
			}
		}
	}
	return newImage;
}

#endif // !_X_CONV_PAR_