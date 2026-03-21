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
#pragma unroll
  for (y = top; y < bottom; ++y)
  {
    uint8_t *ref_addr = ref + y * w;                    // Hoist
    __builtin_prefetch(ref + (y + 1) * w + left, 0, 0); // Hint at prefetch of next ref block, no locality, can be removed after access
#pragma unroll
    for (x = left; x < right; ++x)
    {
      int sad = sad_block_8x8(orig_addr, ref_addr + x, w);

      if (sad < best_sad)
      {
        best_mv_x = x - mx;
        best_mv_y = y - my;
        best_sad = sad;
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
  int right = left + 8;
  int bottom = top + 8;

  int w = cm->padw[color_component];

  /* Copy block from ref mandated by MV */
  int x, y;
  for (y = top; y < bottom; ++y)
  {
    for (x = left; x < right; ++x)
    {
      predicted[y * w + x] = ref[(y + mb->mv_y) * w + (x + mb->mv_x)];
    }
  }
}

void c63_motion_compensate(struct c63_common *cm)
{
  int mb_x, mb_y;

  /* Luma */
  nvtxRangePush("MC Luma");
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
