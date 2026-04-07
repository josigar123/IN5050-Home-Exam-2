#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"
#include "dsp.h"

inline void dequantize_idct_row(int16_t *in_data, uint8_t *prediction, int w, int h,
                                int y, uint8_t *out_data, float16_t *dequant_scale)
{
  // Process 2 MBs at a time
  int16_t block_a[64] __attribute__((aligned(16)));
  int16_t block_b[64] __attribute__((aligned(16)));

/* Perform the dequantization and iDCT */
#pragma unroll
  for (int x = 0; x < w; x += 16)
  {
    dequant_idct_block_8x8(in_data + (x * 8), block_a, dequant_scale);
    dequant_idct_block_8x8(in_data + ((x + 8) * 8), block_b, dequant_scale);

#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
      // Load two rows from two different mbs
      uint8x16_t pred = vld1q_u8(prediction + i * w + x);

      // Reconstruct the first row of the first block then second block
      int16x8_t sum_a = vaddq_s16(vld1q_s16(block_a + i * 8), vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(pred))));
      int16x8_t sum_b = vaddq_s16(vld1q_s16(block_b + i * 8), vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(pred))));

      // Write back uint8 data, vqmovun_s16 narrows and saturates to uint8 removing manual clamping
      vst1q_u8(out_data + i * w + x, vcombine_u8(vqmovun_s16(sum_a), vqmovun_s16(sum_b)));
    }
  }
}

void dequantize_idct(int16_t *in_data, uint8_t *prediction, uint32_t width,
                     uint32_t height, uint8_t *out_data, float16_t *dequant_scale)
{
  int y;
  for (y = 0; y < height; y += 8)
  {
    dequantize_idct_row(in_data + y * width, prediction + y * width, width, height, y,
                        out_data + y * width, dequant_scale);
  }
}

inline void dct_quantize_row(uint8_t *in_data, uint8_t *prediction, int w,
                             int16_t *out_data, float16_t *quant_scale)
{
  // Process 2 MBs at a time
  int16_t block_a[64] __attribute__((aligned(16)));
  int16_t block_b[64] __attribute__((aligned(16)));

/* Perform the DCT and quantization */
#pragma unroll
  for (int x = 0; x < w; x += 16)
  {
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
      uint8x16_t in_row = vld1q_u8(in_data + i * w + x);      // Reads 2 rows from two different MBs
      uint8x16_t pred_row = vld1q_u8(prediction + i * w + x); // Reads 2 rows from two diff pred MBs

      // Process first block using the low 8 lanes (vget_low_u8), then widen to 16x8 (vmovl_u8), then reinterpres as signed  before dowing qword sub vsubq_s16
      vst1q_s16(block_a + i * 8, vsubq_s16(
                                     vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(in_row))),
                                     vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(pred_row)))));

      // Process second block using the high 8 lanes (vget_high_u8), then widen to 16x8 (vmovl_u8), then reinterpres as signed  before dowing qword sub vsubq_s16
      vst1q_s16(block_b + i * 8, vsubq_s16(
                                     vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(in_row))),
                                     vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(pred_row)))));
    }

    // Calc for first block, then second block
    dct_quant_block_8x8(block_a, out_data + (x * 8), quant_scale);
    dct_quant_block_8x8(block_b, out_data + ((x + 8) * 8), quant_scale);
  }
}

void dct_quantize(uint8_t *in_data, uint8_t *prediction, uint32_t width,
                  uint32_t height, int16_t *out_data, float16_t *quant_scale)
{
#pragma unroll
  for (int y = 0; y < height; y += 8)
  {
    dct_quantize_row(in_data + y * width, prediction + y * width, width,
                     out_data + y * width, quant_scale);
  }
}

void destroy_frame(struct frame *f)
{
  /* First frame doesn't have a reconstructed frame to destroy */
  if (!f)
  {
    return;
  }

  free(f->recons->Y);
  free(f->recons->U);
  free(f->recons->V);
  free(f->recons);

  free(f->residuals->Ydct);
  free(f->residuals->Udct);
  free(f->residuals->Vdct);
  free(f->residuals);

  free(f->predicted->Y);
  free(f->predicted->U);
  free(f->predicted->V);
  free(f->predicted);

  free(f->mbs[Y_COMPONENT]);
  free(f->mbs[U_COMPONENT]);
  free(f->mbs[V_COMPONENT]);

  free(f);
}

struct frame *create_frame(struct c63_common *cm, yuv_t *image)
{
  struct frame *f = malloc(sizeof(struct frame));

  f->orig = image;

  f->recons = malloc(sizeof(yuv_t));
  f->recons->Y = malloc(cm->ypw * cm->yph);
  f->recons->U = malloc(cm->upw * cm->uph);
  f->recons->V = malloc(cm->vpw * cm->vph);

  f->predicted = malloc(sizeof(yuv_t));
  f->predicted->Y = calloc(cm->ypw * cm->yph, sizeof(uint8_t));
  f->predicted->U = calloc(cm->upw * cm->uph, sizeof(uint8_t));
  f->predicted->V = calloc(cm->vpw * cm->vph, sizeof(uint8_t));

  f->residuals = malloc(sizeof(dct_t));
  f->residuals->Ydct = calloc(cm->ypw * cm->yph, sizeof(int16_t));
  f->residuals->Udct = calloc(cm->upw * cm->uph, sizeof(int16_t));
  f->residuals->Vdct = calloc(cm->vpw * cm->vph, sizeof(int16_t));

  f->mbs[Y_COMPONENT] =
      calloc(cm->mb_rows * cm->mb_cols, sizeof(struct macroblock));
  f->mbs[U_COMPONENT] =
      calloc(cm->mb_rows / 2 * cm->mb_cols / 2, sizeof(struct macroblock));
  f->mbs[V_COMPONENT] =
      calloc(cm->mb_rows / 2 * cm->mb_cols / 2, sizeof(struct macroblock));

  return f;
}

void dump_image(yuv_t *image, int w, int h, FILE *fp)
{
  fwrite(image->Y, 1, w * h, fp);
  fwrite(image->U, 1, w * h / 4, fp);
  fwrite(image->V, 1, w * h / 4, fp);
}
