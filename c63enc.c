#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "c63.h"
#include "c63_write.h"
#include "common.h"
#include "me.h"
#include "tables.h"

#include "nvtx3/nvToolsExt.h"

static char *output_file, *input_file;
FILE *outfile;

static int limit_numframes = 0;

static uint32_t width;
static uint32_t height;

/* getopt */
extern int optind;
extern char *optarg;

static yuv_t *alloc_input_image(struct c63_common *cm)
{

  yuv_t *image = malloc(sizeof(*image));

  image->Y = calloc(1, cm->padw[Y_COMPONENT] * cm->padh[Y_COMPONENT]);
  image->U = calloc(1, cm->padw[U_COMPONENT] * cm->padh[U_COMPONENT]);
  image->V = calloc(1, cm->padw[V_COMPONENT] * cm->padh[V_COMPONENT]);

  return image;
}

static void free_input_image(yuv_t *image)
{
  free(image->Y);
  free(image->U);
  free(image->V);
  free(image);
}

/* Read planar YUV frames with 4:2:0 chroma sub-sampling */
static int read_yuv_into_from(yuv_t *image, FILE *file)
{
  size_t len = 0;

  len += fread(image->Y, 1, width * height, file);
  len += fread(image->U, 1, (width * height) / 4, file);
  len += fread(image->V, 1, (width * height) / 4, file);

  if (ferror(file))
  {
    perror("ferror");
    exit(EXIT_FAILURE);
  }

  if (feof(file))
  {
    return 0;
  }
  else if (len != width * height * 1.5)
  {
    fprintf(stderr, "Reached end of file, but incorrect bytes read.\n");
    fprintf(stderr, "Wrong input? (height: %d width: %d)\n", height, width);
    return 0;
  }

  return 1;
}

static void c63_encode_image(struct c63_common *cm, yuv_t *image)
{
  /* Advance to next frame */
  struct frame *old_refframe = cm->refframe;
  cm->refframe = cm->curframe;
  cm->curframe = old_refframe;
  cm->curframe->orig = image; // curframe must point to the newly read image

  /* Check if keyframe */
  if (cm->framenum == 0 || cm->frames_since_keyframe == cm->keyframe_interval)
  {
    cm->curframe->keyframe = 1;
    cm->frames_since_keyframe = 0;

    fprintf(stderr, " (keyframe) ");
  }
  else
  {
    cm->curframe->keyframe = 0;
  }

  if (!cm->curframe->keyframe)
  {
    c63_motion_estimate(cm);
    c63_motion_compensate(cm);
  }
  else
  {
    // Must reset predicted frame on keyframes, as frames are ping-ponged
    yuv_t *pred = cm->curframe->predicted;
    memset(pred->Y, 0, cm->ypw * cm->yph);
    memset(pred->U, 0, cm->upw * cm->uph);
    memset(pred->V, 0, cm->vpw * cm->vph);
  }

  /* DCT and Quantization */
  nvtxRangePush("DCT+Q Y");
  dct_quantize(image->Y, cm->curframe->predicted->Y, cm->padw[Y_COMPONENT],
               cm->padh[Y_COMPONENT], cm->curframe->residuals->Ydct,
               cm->quanttbl[Y_COMPONENT]);
  nvtxRangePop();

  nvtxRangePush("DCT+Q U");
  dct_quantize(image->U, cm->curframe->predicted->U, cm->padw[U_COMPONENT],
               cm->padh[U_COMPONENT], cm->curframe->residuals->Udct,
               cm->quanttbl[U_COMPONENT]);
  nvtxRangePop();

  nvtxRangePush("DCT+Q V");
  dct_quantize(image->V, cm->curframe->predicted->V, cm->padw[V_COMPONENT],
               cm->padh[V_COMPONENT], cm->curframe->residuals->Vdct,
               cm->quanttbl[V_COMPONENT]);
  nvtxRangePop();

  /* Reconstruct frame for inter-prediction */
  nvtxRangePush("IDCT+DQ Y");
  dequantize_idct(cm->curframe->residuals->Ydct, cm->curframe->predicted->Y,
                  cm->ypw, cm->yph, cm->curframe->recons->Y,
                  cm->quanttbl[Y_COMPONENT]);
  nvtxRangePop();

  nvtxRangePush("IDCT+DQ U");
  dequantize_idct(cm->curframe->residuals->Udct, cm->curframe->predicted->U,
                  cm->upw, cm->uph, cm->curframe->recons->U,
                  cm->quanttbl[U_COMPONENT]);
  nvtxRangePop();

  nvtxRangePush("IDCT+DQ V");
  dequantize_idct(cm->curframe->residuals->Vdct, cm->curframe->predicted->V,
                  cm->vpw, cm->vph, cm->curframe->recons->V,
                  cm->quanttbl[V_COMPONENT]);
  nvtxRangePop();
}

struct c63_common *init_c63_enc(int width, int height)
{
  int i;

  /* calloc() sets allocated memory to zero */
  struct c63_common *cm = calloc(1, sizeof(struct c63_common));

  cm->width = width;
  cm->height = height;

  cm->padw[Y_COMPONENT] = cm->ypw = (uint32_t)(ceil(width / 16.0f) * 16);
  cm->padh[Y_COMPONENT] = cm->yph = (uint32_t)(ceil(height / 16.0f) * 16);
  cm->padw[U_COMPONENT] = cm->upw =
      (uint32_t)(ceil(width * UX / (YX * 8.0f)) * 8);
  cm->padh[U_COMPONENT] = cm->uph =
      (uint32_t)(ceil(height * UY / (YY * 8.0f)) * 8);
  cm->padw[V_COMPONENT] = cm->vpw =
      (uint32_t)(ceil(width * VX / (YX * 8.0f)) * 8);
  cm->padh[V_COMPONENT] = cm->vph =
      (uint32_t)(ceil(height * VY / (YY * 8.0f)) * 8);

  cm->mb_cols = cm->ypw / 8;
  cm->mb_rows = cm->yph / 8;

  /* Quality parameters -- Home exam deliveries should have original values,
   i.e., quantization factor should be 25, search range should be 16, and the
   keyframe interval should be 100. */
  cm->qp = 25;                 // Constant quantization factor. Range: [1..50]
  cm->me_search_range = 16;    // Pixels in every direction
  cm->keyframe_interval = 100; // Distance between keyframes

  /* Initialize quantization tables */
  for (i = 0; i < 64; ++i)
  {
    cm->quanttbl[Y_COMPONENT][i] = yquanttbl_def[i] / (cm->qp / 10.0);
    cm->quanttbl[U_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
    cm->quanttbl[V_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
  }

  return cm;
}

static void print_help()
{
  printf("Usage: ./c63enc [options] input_file\n");
  printf("Commandline options:\n");
  printf("  -h                             Height of images to compress\n");
  printf("  -w                             Width of images to compress\n");
  printf("  -o                             Output file (.c63)\n");
  printf("  [-f]                           Limit number of frames to encode\n");
  printf("\n");

  exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
  int c;

  if (argc == 1)
  {
    print_help();
  }

  while ((c = getopt(argc, argv, "h:w:o:f:i:")) != -1)
  {
    switch (c)
    {
    case 'h':
      height = atoi(optarg);
      break;
    case 'w':
      width = atoi(optarg);
      break;
    case 'o':
      output_file = optarg;
      break;
    case 'f':
      limit_numframes = atoi(optarg);
      break;
    default:
      print_help();
      break;
    }
  }

  if (optind >= argc)
  {
    fprintf(stderr, "Error getting program options, try --help.\n");
    exit(EXIT_FAILURE);
  }

  outfile = fopen(output_file, "wb");

  if (outfile == NULL)
  {
    perror("fopen");
    exit(EXIT_FAILURE);
  }

  struct c63_common *cm = init_c63_enc(width, height);
  cm->e_ctx.fp = outfile;

  input_file = argv[optind];

  if (limit_numframes)
  {
    printf("Limited to %d frames.\n", limit_numframes);
  }

  FILE *infile = fopen(input_file, "rb");

  if (infile == NULL)
  {
    perror("fopen");
    exit(EXIT_FAILURE);
  }

  /* Encode input frames */
  int numframes = 0;

  // Allocate image once, reuse
  yuv_t *image = alloc_input_image(cm);

  struct frame *frame_a = create_frame(cm, image);
  struct frame *frame_b = create_frame(cm, image);
  cm->refframe = frame_a;
  cm->curframe = frame_b;
  cm->framenum = 0;
  cm->frames_since_keyframe = 0;

  while (1)
  {
    if (!read_yuv_into_from(image, infile))
    {
      break;
    }

    printf("Encoding frame %d, ", numframes);
    c63_encode_image(cm, image);

    nvtxRangePush("Write frame");
    write_frame(cm);
    nvtxRangePop();
    ++cm->framenum;
    ++cm->frames_since_keyframe;

    printf("Done!\n");

    ++numframes;

    if (limit_numframes && numframes >= limit_numframes)
    {
      break;
    }
  }

  destroy_frame(frame_a);
  destroy_frame(frame_b);
  free_input_image(image);
  free(cm);
  fclose(outfile);
  fclose(infile);

  return EXIT_SUCCESS;
}
