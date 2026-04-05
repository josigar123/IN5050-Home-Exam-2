#include <inttypes.h>
#include <math.h>
#include <stdlib.h>

#include <time.h>

#include "dsp.h"
#include "tables.h"

// struct timespec t0, t1;
//   clock_gettime(CLOCK_MONOTONIC, &t0);
//   for (int i = 0; i < 1000000; i++)
//     scale_block(mb2, mb);
//   clock_gettime(CLOCK_MONOTONIC, &t1);

//   double per_call_ns = ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / 1000000.0;
//   printf("%.1f ns per call\n", per_call_ns);

static inline void transpose_block_f16(float16_t *in_data, float16_t *out_data)
{
  // Load all 8 rows of f16s
  float16x8_t row0 = vld1q_f16(in_data);
  float16x8_t row1 = vld1q_f16(in_data + 8);
  float16x8_t row2 = vld1q_f16(in_data + 16);
  float16x8_t row3 = vld1q_f16(in_data + 24);
  float16x8_t row4 = vld1q_f16(in_data + 32);
  float16x8_t row5 = vld1q_f16(in_data + 40);
  float16x8_t row6 = vld1q_f16(in_data + 48);
  float16x8_t row7 = vld1q_f16(in_data + 56);

  // Transpose entire rows pair-wise, trans*.vals[*] will hold first two values of their rows (even and odd) a 2x2 transpoe
  float16x8x2_t trans01 = vtrnq_f16(row0, row1);
  float16x8x2_t trans23 = vtrnq_f16(row2, row3);
  float16x8x2_t trans45 = vtrnq_f16(row4, row5);
  float16x8x2_t trans67 = vtrnq_f16(row6, row7);

  // Transpose the 4x4 sub-block, reinterpreted as f32 to move 2 f16 as a unit
  float32x4x2_t trans0123_0 = vtrnq_f32(vreinterpretq_f32_f16(trans01.val[0]), vreinterpretq_f32_f16(trans23.val[0]));
  float32x4x2_t trans0123_1 = vtrnq_f32(vreinterpretq_f32_f16(trans01.val[1]), vreinterpretq_f32_f16(trans23.val[1]));
  float32x4x2_t trans4567_0 = vtrnq_f32(vreinterpretq_f32_f16(trans45.val[0]), vreinterpretq_f32_f16(trans67.val[0]));
  float32x4x2_t trans4567_1 = vtrnq_f32(vreinterpretq_f32_f16(trans45.val[1]), vreinterpretq_f32_f16(trans67.val[1]));

  // Final transpose to get complete rows, interpret af f64 to move 4 f16 as a unit
  // vtrn1q takes the even indexed values, vtrn2q takes the odd indexed values, so together they give us the full rows back but transposed
  float64x2_t trans_row0 = vtrn1q_f64(vreinterpretq_f64_f32(trans0123_0.val[0]), vreinterpretq_f64_f32(trans4567_0.val[0]));
  float64x2_t trans_row4 = vtrn2q_f64(vreinterpretq_f64_f32(trans0123_0.val[0]), vreinterpretq_f64_f32(trans4567_0.val[0]));

  float64x2_t trans_row2 = vtrn1q_f64(vreinterpretq_f64_f32(trans0123_0.val[1]), vreinterpretq_f64_f32(trans4567_0.val[1]));
  float64x2_t trans_row6 = vtrn2q_f64(vreinterpretq_f64_f32(trans0123_0.val[1]), vreinterpretq_f64_f32(trans4567_0.val[1]));

  float64x2_t trans_row1 = vtrn1q_f64(vreinterpretq_f64_f32(trans0123_1.val[0]), vreinterpretq_f64_f32(trans4567_1.val[0]));
  float64x2_t trans_row5 = vtrn2q_f64(vreinterpretq_f64_f32(trans0123_1.val[0]), vreinterpretq_f64_f32(trans4567_1.val[0]));

  float64x2_t trans_row3 = vtrn1q_f64(vreinterpretq_f64_f32(trans0123_1.val[1]), vreinterpretq_f64_f32(trans4567_1.val[1]));
  float64x2_t trans_row7 = vtrn2q_f64(vreinterpretq_f64_f32(trans0123_1.val[1]), vreinterpretq_f64_f32(trans4567_1.val[1]));

  // Write rows in-order
  vst1q_f16(out_data, vreinterpretq_f16_f64(trans_row0));
  vst1q_f16(out_data + 8, vreinterpretq_f16_f64(trans_row1));
  vst1q_f16(out_data + 16, vreinterpretq_f16_f64(trans_row2));
  vst1q_f16(out_data + 24, vreinterpretq_f16_f64(trans_row3));
  vst1q_f16(out_data + 32, vreinterpretq_f16_f64(trans_row4));
  vst1q_f16(out_data + 40, vreinterpretq_f16_f64(trans_row5));
  vst1q_f16(out_data + 48, vreinterpretq_f16_f64(trans_row6));
  vst1q_f16(out_data + 56, vreinterpretq_f16_f64(trans_row7));
}

void init_idct_lookup()
{
  transpose_block_f16((float16_t *)dctlookup, (float16_t *)idctlookup);
}

// Calculates coefficients for 1 row
static inline void dct_1d(float16_t *in_data, float16_t *out_data)
{
  // Accumulator holding what is to be 8 coefficients (1 row)
  float16x8_t acc0 = vdupq_n_f16((__fp16)0.0f), acc1 = vdupq_n_f16((__fp16)0.0f);
  for (int i = 0; i < 8; i += 2)
  {
    // Do fused multiply-accumulate for DCT coeff
    acc0 = vfmaq_f16(acc0, vdupq_n_f16(in_data[i]), vld1q_f16(&dctlookup[i][0]));
    acc1 = vfmaq_f16(acc1, vdupq_n_f16(in_data[i + 1]), vld1q_f16(&dctlookup[i + 1][0]));
  }

  vst1q_f16(out_data, vaddq_f16(acc0, acc1));
}

static inline void idct_1d(float16_t *in_data, float16_t *out_data)
{

  float16x8_t acc0 = vdupq_n_f16((__fp16)0.0f), acc1 = vdupq_n_f16((__fp16)0.0f);
  for (int i = 0; i < 8; i += 2)
  {
    acc0 = vfmaq_f16(acc0, vdupq_n_f16(in_data[i]), vld1q_f16(&idctlookup[i][0]));
    acc1 = vfmaq_f16(acc1, vdupq_n_f16(in_data[i + 1]), vld1q_f16(&idctlookup[i + 1][0]));
  }

  vst1q_f16(out_data, vaddq_f16(acc0, acc1));
}

static inline void quantize_block(float16_t *in_data, float16_t *out_data, float16_t *quant_scale)
{
  float16_t gathered_coeffs[64] __attribute((aligned(16))); // 16 byte aligned array
#pragma unroll
  // Gather scattered coefficients with zigzag_index LUT
  for (int z = 0; z < 64; ++z)
  {
    gathered_coeffs[z] = in_data[zigzag_index[z]];
  }

#pragma unroll
  for (int zigzag = 0; zigzag < 64; zigzag += 8)
  {
    // Quantize 4 coefficients: First load 4 coeffs and 4 quant scales, then multiply, then round, then store to out_data
    vst1q_f16(&out_data[zigzag], vrndiq_f16(vmulq_f16(vld1q_f16(gathered_coeffs + zigzag), vld1q_f16(&quant_scale[zigzag]))));
  }
}

static inline void dequantize_block(int16_t *in_data, float16_t *out_data,
                                    float16_t *dequant_scale)
{
  float16_t dequantized_coeffs[64] __attribute__((aligned(16))); // 16 byte aligned array

#pragma unroll
  // Gather coeffs in zigzag order
  for (int i = 0; i < 64; i += 8)
  {
    // Load 8 coeffs and dequant them, convert to f16 from input
    vst1q_f16(dequantized_coeffs + i, vmulq_f16(vcvtq_f16_s16(vld1q_s16(in_data + i)), vld1q_f16(dequant_scale + i)));
  }

#pragma unroll
  // Write in de-zigzag, so its in normal row-major mb
  for (int i = 0; i < 64; ++i)
  {
    out_data[zigzag_index[i]] = dequantized_coeffs[i];
  }
}

__attribute__((always_inline)) inline void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data,
                                                               float16_t *quant_scale)
{
  float16_t mb[64] __attribute__((aligned(16)));
  float16_t mb2[64] __attribute__((aligned(16)));

#pragma unroll
  for (int i = 0; i < 64; i += 8)
  {
    // Store converted values in mb2
    vst1q_f16(mb2 + i, vcvtq_f16_s16(vld1q_s16(in_data + i)));
  }

/* Two 1D DCT operations with transpose */
#pragma unroll
  for (int v = 0; v < 8; ++v)
  {
    dct_1d(mb2 + v * 8, mb + v * 8);
  }
  transpose_block_f16(mb, mb2);

#pragma unroll
  for (int v = 0; v < 8; ++v)
  {
    dct_1d(mb2 + v * 8, mb + v * 8);
  }
  transpose_block_f16(mb, mb2);

  quantize_block(mb2, mb, quant_scale);

#pragma unroll
  for (int i = 0; i < 64; i += 8)
  {
    // Convert to s16 from f16, 8 vals at a time
    vst1q_s16(out_data + i, vcvtq_s16_f16(vld1q_f16(mb + i)));
  }
}

__attribute__((always_inline)) inline void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data,
                                                                  float16_t *dequant_scale)
{
  float16_t mb[64] __attribute__((aligned(16)));
  float16_t mb2[64] __attribute__((aligned(16)));

  dequantize_block(in_data, mb2, dequant_scale);

  /* Two 1D inverse DCT operations with transpose */
#pragma unroll
  for (int v = 0; v < 8; ++v)
  {
    idct_1d(mb2 + v * 8, mb + v * 8);
  }
  transpose_block_f16(mb, mb2);
#pragma unroll
  for (int v = 0; v < 8; ++v)
  {
    idct_1d(mb2 + v * 8, mb + v * 8);
  }
  transpose_block_f16(mb, mb2);

#pragma unroll
  for (int i = 0; i < 64; i += 8)
  {
    vst1q_s16(out_data + i, vcvtq_s16_f16(vld1q_f16(mb2 + i)));
  }
}
