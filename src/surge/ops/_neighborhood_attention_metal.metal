#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant int NW = 8 + KW - 1;
constant int NH = TH + KH - 1;
constant int N = ((NH * NW + 31) / 32) * 32;

kernel void attention(
    device const T* q,
    device const T* k,
    device const T* v,
    device T* out,
    constant long& h,
    constant long& w,
    constant long& heads,
    constant float& scale,
    constant long* qs,
    constant long* ks,
    constant long* vs,
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const int x = group.x * 8, y = group.y * TH;
    const int batch = group.z / heads, head = group.z % heads;
    const int sy = clamp(y - KH / 2, 0, max(int(h) - NH, 0));
    const int sx = clamp(x - KW / 2, 0, max(int(w) - NW, 0));
    threadgroup S queries[TH * 8 * D];
    threadgroup S kv[32 * D];
    threadgroup float scores[TH * 8 * N];

    for (int i = tid; i < TH * 8 * D; i += 128) {
        const int row = i / D, d = i % D;
        queries[i] = x + row % 8 < w && y + row / 8 < h
            ? float(q[batch * qs[0] + (y + row / 8) * qs[1] + (x + row % 8) * qs[2]
                                              + head * qs[3] + d * qs[4]]) : 0.0f;
    }
    for (int start = 0; start < N; start += 32) {
        for (int i = tid; i < 32 * D; i += 128) {
            const int n = start + i / D, d = i % D;
            const int ky = sy + n / NW, kx = sx + n % NW;
            kv[i] = n < NH * NW && ky < h && kx < w ? float(k[batch * ks[0] + ky * ks[1] + kx * ks[2]
                                                        + head * ks[3] + d * ks[4]]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int row = 0; row < TH; ++row) {
            simdgroup_float8x8 dot(0.0f);
            for (int d = 0; d < D; d += 8) {
                simdgroup_matrix<S, 8, 8> a, b;
                simdgroup_load(a, queries + row * 8 * D + d, D);
                simdgroup_load(b, kv + simd * 8 * D + d, D, ulong2(0), true);
                simdgroup_multiply_accumulate(dot, a, b, dot);
            }
            simdgroup_store(dot, scores + row * 8 * N + start + simd * 8, N);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Probabilities stay in threadgroup memory between the two matrix products.
    for (int row = simd; row < TH * 8; row += 4) {
        const int wx = clamp(x + row % 8 - KW / 2, 0, int(w) - KW);
        const int wy = clamp(y + row / 8 - KH / 2, 0, int(h) - KH);
        float maximum = -INFINITY;
        for (int n = lane; n < N; n += 32) {
            const int ky = sy + n / NW, kx = sx + n % NW;
            const float value = ky >= wy && ky < wy + KH && kx >= wx && kx < wx + KW
                ? scores[row * N + n] * scale : -INFINITY;
            scores[row * N + n] = value;
            maximum = max(maximum, value);
        }
        maximum = simd_max(maximum);
        float denominator = 0.0f;
        for (int n = lane; n < N; n += 32) {
            const float p = exp(scores[row * N + n] - maximum);
            scores[row * N + n] = p;
            denominator += p;
        }
        denominator = simd_sum(denominator);
        for (int n = lane; n < N; n += 32) {
            scores[row * N + n] /= denominator;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 acc[TH * ((D + 31) / 32)];
    for (int j = 0; j < TH * ((D + 31) / 32); ++j) {
        acc[j] = simdgroup_float8x8(0.0f);
    }
    for (int start = 0; start < N; start += 32) {
        for (int i = tid; i < 32 * D; i += 128) {
            const int n = start + i / D, d = i % D;
            const int ky = sy + n / NW, kx = sx + n % NW;
            kv[i] = n < NH * NW && ky < h && kx < w ? float(v[batch * vs[0] + ky * vs[1] + kx * vs[2]
                                                        + head * vs[3] + d * vs[4]]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < TH * ((D + 31) / 32); ++j) {
            const int d = (j % ((D + 31) / 32)) * 32 + simd * 8;
            const int row = j / ((D + 31) / 32);
            if (d < D) {
                for (int n = 0; n < 32; n += 8) {
                    simdgroup_float8x8 a;
                    simdgroup_matrix<S, 8, 8> b;
                    simdgroup_load(a, scores + row * 8 * N + start + n, N);
                    simdgroup_load(b, kv + n * D + d, D);
                    simdgroup_multiply_accumulate(acc[j], a, b, acc[j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Reuse K/V staging memory for FP32 output before converting to the input dtype.
    for (int j = 0; j < TH * ((D + 31) / 32); ++j) {
        const int d = (j % ((D + 31) / 32)) * 32 + simd * 8;
        const int row = j / ((D + 31) / 32);
        if (d < D) {
            simdgroup_store(acc[j], reinterpret_cast<threadgroup float*>(kv) + row * 8 * D + d, D);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tid; i < TH * 8 * D; i += 128) {
        const int row = i / D, d = i % D;
        if (x + row % 8 < w && y + row / 8 < h) {
            out[((batch * h * w + (y + row / 8) * w + x + row % 8) * heads + head) * D + d]
                = T(reinterpret_cast<threadgroup float*>(kv)[i]);
        }
    }
}
