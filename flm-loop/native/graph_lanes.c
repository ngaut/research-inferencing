/* CSR (n x n, row = postsynaptic) times dense X (n x lanes, row-major) -> Y (n x lanes).
 * A lane is an independent sequence; lanes never mix. Rows are independent, so OpenMP
 * row parallelism keeps every (row, lane) sum in CSR order: results do not depend on
 * the thread count. IEEE float32, no fused multiply-add, no fast-math.
 * The same routine serves W^T products when handed the CSR arrays of the transpose.
 */
#include <stdint.h>

#define MAX_LANES 64

void flmloop_csr_lanes(int32_t n, int32_t lanes, const int32_t *restrict ptr,
                       const int32_t *restrict ix, const float *restrict w,
                       const float *restrict x, float *restrict y) {
    if (lanes < 1 || lanes > MAX_LANES) return;
    #pragma omp parallel for schedule(dynamic, 2048)
    for (int32_t row = 0; row < n; ++row) {
        float acc[MAX_LANES];
        for (int32_t j = 0; j < lanes; ++j) acc[j] = 0.0f;
        for (int32_t p = ptr[row]; p < ptr[row + 1]; ++p) {
            const float wv = w[p];
            const float *v = x + (int64_t)ix[p] * lanes;
            for (int32_t j = 0; j < lanes; ++j) acc[j] += wv * v[j];
        }
        float *out = y + (int64_t)row * lanes;
        for (int32_t j = 0; j < lanes; ++j) out[j] = acc[j];
    }
}
