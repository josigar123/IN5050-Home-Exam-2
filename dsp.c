#include <inttypes.h>
#include <math.h>
#include <stdlib.h>

#include <time.h>

#include "dsp.h"
#include "tables.h"

// Scale block orig takes 27-29 ns, new version takes 12ns

// struct timespec t0, t1;
//   clock_gettime(CLOCK_MONOTONIC, &t0);
//   for (int i = 0; i < 1000000; i++)
//     scale_block(mb2, mb);
//   clock_gettime(CLOCK_MONOTONIC, &t1);

//   double per_call_ns = ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / 1000000.0;
//   printf("%.1f ns per call\n", per_call_ns);

// A look-up table for scale-block to avoid branching in inner-loop
static const float scale_lut[8][8] = {
    {0.5, ISQRT2, ISQRT2, ISQRT2, ISQRT2, ISQRT2, ISQRT2, ISQRT2},
    {ISQRT2, 1, 1, 1, 1, 1, 1, 1},
    {ISQRT2, 1, 1, 1, 1, 1, 1, 1},
    {ISQRT2, 1, 1, 1, 1, 1, 1, 1},
    {ISQRT2, 1, 1, 1, 1, 1, 1, 1},
    {ISQRT2, 1, 1, 1, 1, 1, 1, 1},
    {ISQRT2, 1, 1, 1, 1, 1, 1, 1},
    {ISQRT2, 1, 1, 1, 1, 1, 1, 1},
};

static inline void transpose_4x4(float *in_data, float *out_data, int in_stride, int out_stride)
{
  // Extract rows for the 4x4 matrix, MB is split into four 4x4 regions, stride is 8 for in and out
  float32x4_t row0 = vld1q_f32(in_data + 0 * in_stride);
  float32x4_t row1 = vld1q_f32(in_data + 1 * in_stride);
  float32x4_t row2 = vld1q_f32(in_data + 2 * in_stride);
  float32x4_t row3 = vld1q_f32(in_data + 3 * in_stride);

  // Treats rows as two 2x2 matrices and transposes them, trans*.val[*] hold partial columns
  float32x4x2_t trans01 = vtrnq_f32(row0, row1);
  float32x4x2_t trans23 = vtrnq_f32(row2, row3);

  // Combines columns and stores in out data so we have a transposed 4x4 block
  vst1q_f32(out_data + 0 * out_stride, vcombine_f32(vget_low_f32(trans01.val[0]), vget_low_f32(trans23.val[0])));
  vst1q_f32(out_data + 1 * out_stride, vcombine_f32(vget_low_f32(trans01.val[1]), vget_low_f32(trans23.val[1])));
  vst1q_f32(out_data + 2 * out_stride, vcombine_f32(vget_high_f32(trans01.val[0]), vget_high_f32(trans23.val[0])));
  vst1q_f32(out_data + 3 * out_stride, vcombine_f32(vget_high_f32(trans01.val[1]), vget_high_f32(trans23.val[1])));
}

static void transpose_block(float *in_data, float *out_data)
{
  transpose_4x4(in_data, out_data, 8, 8);
  transpose_4x4(in_data + 4, out_data + 32, 8, 8);
  transpose_4x4(in_data + 32, out_data + 4, 8, 8);
  transpose_4x4(in_data + 36, out_data + 36, 8, 8);
}

// Calculates coefficients for 1 row
static void dct_1d(float *in_data, float *out_data)
{
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
#pragma unroll
  for (int i = 0; i < 8; ++i)
  {
    // Broadcast input value to calculate coefficients for
    float32x4_t in_i = vdupq_n_f32(in_data[i]);

    // Do fused multiply-accumulate for DCT coeff
    acc0 = vfmaq_f32(acc0, in_i, vld1q_f32(&dctlookup[i][0]));
    acc1 = vfmaq_f32(acc1, in_i, vld1q_f32(&dctlookup[i][4]));
  }

  vst1q_f32(&out_data[0], acc0);
  vst1q_f32(&out_data[4], acc1);
}

static void idct_1d(float *in_data, float *out_data)
{
  int i, j;

  for (i = 0; i < 8; ++i)
  {
    float idct = 0;

    for (j = 0; j < 8; ++j)
    {
      idct += in_data[j] * dctlookup[i][j];
    }

    out_data[i] = idct;
  }
}

static void scale_block(float *in_data, float *out_data)
{
#pragma unroll
  for (int v = 0; v < 8; ++v)
  {
    // Load the appropriate scales into 4 lanes (1 row total)
    float32x4_t scale0 = vld1q_f32(&scale_lut[v][0]);
    float32x4_t scale1 = vld1q_f32(&scale_lut[v][4]);

    // Load the input data for the row (first 4 values and next 4 values)
    float32x4_t in_data0 = vld1q_f32(&in_data[v * 8]);
    float32x4_t in_data1 = vld1q_f32(&in_data[v * 8 + 4]);

    // Write scaled values to out-data two calls, half-row each
    vst1q_f32(&out_data[v * 8], vmulq_f32(in_data0, scale0));
    vst1q_f32(&out_data[v * 8 + 4], vmulq_f32(in_data1, scale1));
  }
}

static void quantize_block(float *in_data, float *out_data, uint8_t *quant_tbl)
{
  int zigzag;

  for (zigzag = 0; zigzag < 64; ++zigzag)
  {
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];

    float dct = in_data[v * 8 + u];

    /* Zig-zag and quantize */
    out_data[zigzag] = (float)round((dct / 4.0) / quant_tbl[zigzag]);
  }
}

static void dequantize_block(float *in_data, float *out_data,
                             uint8_t *quant_tbl)
{
  int zigzag;

  for (zigzag = 0; zigzag < 64; ++zigzag)
  {
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];

    float dct = in_data[zigzag];

    /* Zig-zag and de-quantize */
    out_data[v * 8 + u] = (float)round((dct * quant_tbl[zigzag]) / 4.0);
  }
}

void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data,
                         uint8_t *quant_tbl)
{
  float mb[8 * 8] __attribute((aligned(16)));
  float mb2[8 * 8] __attribute((aligned(16)));

  int i, v;

  for (i = 0; i < 64; ++i)
  {
    mb2[i] = in_data[i];
  }

  /* Two 1D DCT operations with transpose */
  for (v = 0; v < 8; ++v)
  {
    dct_1d(mb2 + v * 8, mb + v * 8);
  }
  transpose_block(mb, mb2);

  for (v = 0; v < 8; ++v)
  {
    dct_1d(mb2 + v * 8, mb + v * 8);
  }
  transpose_block(mb, mb2);

  scale_block(mb2, mb);

  quantize_block(mb, mb2, quant_tbl);

  for (i = 0; i < 64; ++i)
  {
    out_data[i] = mb2[i];
  }
}

void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data,
                            uint8_t *quant_tbl)
{
  float mb[8 * 8] __attribute((aligned(16)));
  float mb2[8 * 8] __attribute((aligned(16)));

  int i, v;

  for (i = 0; i < 64; ++i)
  {
    mb[i] = in_data[i];
  }

  dequantize_block(mb, mb2, quant_tbl);
  scale_block(mb2, mb);

  /* Two 1D inverse DCT operations with transpose */
  for (v = 0; v < 8; ++v)
  {
    idct_1d(mb + v * 8, mb2 + v * 8);
  }
  transpose_block(mb2, mb);
  for (v = 0; v < 8; ++v)
  {
    idct_1d(mb + v * 8, mb2 + v * 8);
  }
  transpose_block(mb2, mb);

  for (i = 0; i < 64; ++i)
  {
    out_data[i] = mb[i];
  }
}
