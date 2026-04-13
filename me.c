#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nvtx3/nvToolsExt.h"

#include "dsp.h"
#include "me.h"

/* Motion estimation for 8x8 block */
static void me_block_8x8(struct c63_common *cm, int mb_x, int mb_y,
                         uint8_t *orig, uint8_t *ref, int color_component)
{
  struct macroblock *mb =
      &cm->curframe->mbs[color_component][mb_y * cm->padw[color_component] / 8 + mb_x];

  int range = cm->me_search_range;

  /* Quarter resolution for chroma channels. */
  if (color_component > 0)
  {
    range /= 2;
  }

  int left = mb_x * 8 - range;
  int top = mb_y * 8 - range;
  int right = mb_x * 8 + range;
  int bottom = mb_y * 8 + range;

  int w = cm->padw[color_component];
  int h = cm->padh[color_component];

  /* Make sure we are within bounds of reference frame. */
  // Kan bruke neon compare instruksjoner?
  if (left < 0)
  {
    left = 0;
  }
  if (top < 0)
  {
    top = 0;
  }
  if (right > (w - 8))
  {
    right = w - 8;
  }
  if (bottom > (h - 8))
  {
    bottom = h - 8;
  }

  int x, y;

  int mx = mb_x * 8;
  int my = mb_y * 8;

  // Write to stack, then heap at end
  int8_t best_mv_x = 0;
  int8_t best_mv_y = 0;
  int best_sad = INT_MAX;

  uint8_t *orig_addr = orig + my * w + mx; // Unchanging, hoisted out of loop
  uint8x8_t orig_rows[8];

  // Preload whole orig block into NEON registers
#pragma GCC unroll 8
  for (int r = 0; r < 8; ++r)
  {
    orig_rows[r] = vld1_u8(orig_addr + r * w);
  }

  for (y = top; y < bottom; ++y)
  {
    uint8_t *ref_addr = ref + y * w;                    // Hoist
    __builtin_prefetch(ref + (y + 3) * w + left, 0, 1); // Hint at prefetch of ref block
    // Unroll inner loop by 4 to process more candidate blocks at once
    for (x = left; x + 3 < right; x += 4)
    {
      uint16_t sad0 = sad_block_8x8(orig_rows, ref_addr + x, w);
      uint16_t sad1 = sad_block_8x8(orig_rows, ref_addr + x + 1, w);
      uint16_t sad2 = sad_block_8x8(orig_rows, ref_addr + x + 2, w);
      uint16_t sad3 = sad_block_8x8(orig_rows, ref_addr + x + 3, w);

      if (sad0 < best_sad)
      {
        best_sad = sad0;
        best_mv_x = x - mx;
        best_mv_y = y - my;
      }
      if (sad1 < best_sad)
      {
        best_sad = sad1;
        best_mv_x = x + 1 - mx;
        best_mv_y = y - my;
      }
      if (sad2 < best_sad)
      {
        best_sad = sad2;
        best_mv_x = x + 2 - mx;
        best_mv_y = y - my;
      }
      if (sad3 < best_sad)
      {
        best_sad = sad3;
        best_mv_x = x + 3 - mx;
        best_mv_y = y - my;
      }
    }
    // Handle tail after clamping
    for (; x < right; ++x)
    {
      uint16_t s = sad_block_8x8(orig_rows, ref_addr + x, w);
      if (s < best_sad)
      {
        best_mv_x = x - mx;
        best_mv_y = y - my;
        best_sad = s;
      }
    }
  }

  mb->use_mv = 1;
  mb->mv_x = best_mv_x;
  mb->mv_y = best_mv_y;
}

void c63_motion_estimate(struct c63_common *cm)
{
  /* Compare this frame with previous reconstructed frame */
  int mb_x, mb_y;

  /* Luma */
  nvtxRangePush("ME Luma");
#pragma omp parallel for schedule(dynamic) private(mb_x)
  for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
    {
      me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->Y,
                   cm->refframe->recons->Y, Y_COMPONENT);
    }
  }
  nvtxRangePop();

  /* Chroma */
  nvtxRangePush("ME Chroma");
#pragma omp parallel for schedule(dynamic) private(mb_x)
  for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
    {
      me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->U,
                   cm->refframe->recons->U, U_COMPONENT);
      me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->V,
                   cm->refframe->recons->V, V_COMPONENT);
    }
  }
  nvtxRangePop();
}

/* Motion compensation for 8x8 block */
static void mc_block_8x8(struct c63_common *cm, int mb_x, int mb_y,
                         uint8_t *predicted, uint8_t *ref, int color_component)
{
  struct macroblock *mb =
      &cm->curframe->mbs[color_component][mb_y * cm->padw[color_component] / 8 + mb_x];

  if (!mb->use_mv)
  {
    return;
  }

  int left = mb_x * 8;
  int top = mb_y * 8;

  int w = cm->padw[color_component];

  // Initialize ref and pred pointer for first row
  uint8_t *ref_src = ref + (top + mb->mv_y) * w + (left + mb->mv_x);
  uint8_t *pred_dst = predicted + top * w + left;

#pragma GCC unroll 8
  /* Copy block from ref mandated by MV */
  for (int row = 0; row < 8; ++row)
  {
    // Load 8 pixels from ref and store to pred for row
    vst1_u8(pred_dst + row * w, vld1_u8(ref_src + row * w));
  }
}

void c63_motion_compensate(struct c63_common *cm)
{
  int mb_x, mb_y;

  /* Luma */
  nvtxRangePush("MC Luma");
#pragma omp parallel for schedule(static) private(mb_x)
  for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
    {
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->Y,
                   cm->refframe->recons->Y, Y_COMPONENT);
    }
  }
  nvtxRangePop();

  /* Chroma */
  nvtxRangePush("MC Chroma");
#pragma omp parallel for schedule(static) private(mb_x)
  for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
    {
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->U,
                   cm->refframe->recons->U, U_COMPONENT);
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->V,
                   cm->refframe->recons->V, V_COMPONENT);
    }
  }
  nvtxRangePop();
}
