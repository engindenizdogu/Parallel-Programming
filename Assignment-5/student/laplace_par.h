#ifndef _LAPLACE_PAR_
#define _LAPLACE_PAR_

#include <omp.h>

template <int SIZE>
inline void initialize(double a[SIZE + 2][SIZE + 2], double b[SIZE + 2][SIZE + 2])
{
#pragma omp parallel for schedule(static) proc_bind(spread)
	for (int i = 0; i < SIZE + 2; i++)
		for (int j = 0; j < SIZE + 2; j++)
		{
			a[i][j] = 0.0;
			b[i][j] = 0.0;
		}
}

template <int SIZE>
inline void time_step(double a[SIZE + 2][SIZE + 2], double b[SIZE + 2][SIZE + 2], int n)
{
	//int max_threads = omp_get_max_threads();
	//int max_rows = ((SIZE + 2) / max_threads) + 2;
	//int max_cells = max_rows * SIZE;
	//schedule(static, max_rows)

	if (n % 2 == 0)
	{
#pragma omp parallel for schedule(static) shared(a, b) proc_bind(spread) collapse(2)
		for (int i = 1; i < SIZE + 1; i++)
			for (int j = 1; j < SIZE + 1; j++)
				b[i][j] = (a[i + 1][j] + a[i - 1][j] + a[i][j - 1] + a[i][j + 1]) / 4.0;
	}
	else
	{
#pragma omp parallel for schedule(static) shared(a, b) proc_bind(spread) collapse(2)
		for (int i = 1; i < SIZE + 1; i++)
			for (int j = 1; j < SIZE + 1; j++)
				a[i][j] = (b[i + 1][j] + b[i - 1][j] + b[i][j - 1] + b[i][j + 1]) / 4.0;
	}
}

#endif // !_LAPLACE_PAR_