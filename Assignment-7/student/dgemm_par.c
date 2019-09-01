#include "dgemm.h"
#include <immintrin.h>
#include <inttypes.h>

void dgemm(float *a, float *b, float *c, int n)
{
    int ub = n - (n % 8); // 8 floats per SIMD register
    __m256 va, vb, tmp;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < ub; k += 8)
            {
                // TODO: Line to be vectorized
                // c[i * n + j] += a[i * n  + k] * b[j * n  + k];

                // Load vectorized variables
                va = _mm256_loadu_ps(&a[i * n + k]);
                vb = _mm256_loadu_ps(&b[j * n + k]);

                // Perform perations
                tmp = _mm256_mul_ps(va, vb); // tmp = va * vb
                tmp = _mm256_hadd_ps(tmp, tmp); // Horizontal sum

                // Store value
                float result = tmp[0] + tmp[1] + tmp[4] + tmp[5];
                c[i * n + j] += result;

                /* Note: To calculate the sum, one can also use repeated
                        'hadd's and 'permute's to accumulate the result
                        to tmp[0]. This is not the best or the only solution.
                */
            }
        }
    }

    // Remainder (this can also be vectorized)
    // TODO: Vectorize remainder
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = ub; k < n; k++)
            {
                c[i * n + j] += a[i * n + k] * b[j * n + k];
            }
        }
    }
}
