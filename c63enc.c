#define _GNU_SOURCE
#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <semaphore.h>

#include "c63.h"
#include "c63_write.h"
#include "common.h"
#include "me.h"
#include "dsp.h"
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

// A context struct passed to thread functions so they can execute correctly
typedef struct
{
  struct c63_common *cm;
  yuv_t *images[3];
  struct frame *frames[3];
  FILE *infile;
  sem_t slot_free;
  sem_t ready_to_encode;
  sem_t ready_to_write;
  int num_frames_total; // Set by reader thread at EOF
  int limit_numframes;
} pipeline_ctx_t;

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
    return -1;
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

static void c63_encode_image(struct c63_common *cm, struct frame **frames, int frame_index)
{
  /* Advance to next frame */
  // Modulo to move throuhg 3 item ring-buffer
  struct frame *curframe = frames[frame_index % 3];
  struct frame *refframe = frames[(frame_index + 2) % 3];
  cm->curframe = curframe;
  cm->refframe = refframe;
  yuv_t *image = curframe->orig;

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

  nvtxRangePush("DCT+Q Y");
  dct_quantize(image->Y, cm->curframe->predicted->Y, cm->padw[Y_COMPONENT],
               cm->padh[Y_COMPONENT], cm->curframe->residuals->Ydct,
               cm->quant_scale[Y_COMPONENT]);
  nvtxRangePop();

  nvtxRangePush("DCT+Q U");
  dct_quantize(image->U, cm->curframe->predicted->U, cm->padw[U_COMPONENT],
               cm->padh[U_COMPONENT], cm->curframe->residuals->Udct,
               cm->quant_scale[U_COMPONENT]);
  nvtxRangePop();

  nvtxRangePush("DCT+Q V");
  dct_quantize(image->V, cm->curframe->predicted->V, cm->padw[V_COMPONENT],
               cm->padh[V_COMPONENT], cm->curframe->residuals->Vdct,
               cm->quant_scale[V_COMPONENT]);
  nvtxRangePop();

  /* Reconstruct frame for inter-prediction */
  nvtxRangePush("IDCT+DQ Y");
  dequantize_idct(cm->curframe->residuals->Ydct, cm->curframe->predicted->Y,
                  cm->ypw, cm->yph, cm->curframe->recons->Y,
                  cm->dequant_scale[Y_COMPONENT]);
  nvtxRangePop();

  nvtxRangePush("IDCT+DQ U");
  dequantize_idct(cm->curframe->residuals->Udct, cm->curframe->predicted->U,
                  cm->upw, cm->uph, cm->curframe->recons->U,
                  cm->dequant_scale[U_COMPONENT]);
  nvtxRangePop();

  nvtxRangePush("IDCT+DQ V");
  dequantize_idct(cm->curframe->residuals->Vdct, cm->curframe->predicted->V,
                  cm->vpw, cm->vph, cm->curframe->recons->V,
                  cm->dequant_scale[V_COMPONENT]);
  nvtxRangePop();

  ++cm->framenum;
  ++cm->frames_since_keyframe;
}

// Reader thread
void *reader_thread(void *arg)
{
  pipeline_ctx_t *ctx = (pipeline_ctx_t *)arg;
  for (int i = 0;; i++)
  {
    printf("Reading frame %d, ", i);
    // Wait for free slot
    sem_wait(&ctx->slot_free);
    int slot = i % 3;
    if (!read_yuv_into_from(ctx->images[slot], ctx->infile))
    {
      ctx->num_frames_total = i;
      // Post to encode last frame
      sem_post(&ctx->ready_to_encode);
      break;
    }
    // Post that frame is ready to encode
    sem_post(&ctx->ready_to_encode);

    // Encode last frame if a limit on frames to encode is set
    if (ctx->limit_numframes && (i + 1) >= ctx->limit_numframes)
    {
      ctx->num_frames_total = i + 1;
      sem_post(&ctx->ready_to_encode);
      break;
    }
  }
  return NULL;
}

void *encoder_thread(void *arg)
{
  pipeline_ctx_t *ctx = (pipeline_ctx_t *)arg;
  for (int i = 0;; i++)
  {
    sem_wait(&ctx->ready_to_encode); // Wait for a frame ready for encoding
    if (i == ctx->num_frames_total)
    {
      break;
    }
    int slot = i % 3;
    printf("Encoding frame %d, ", i);
    c63_encode_image(ctx->cm, ctx->frames, i);
    sem_post(&ctx->ready_to_write);
  }
  sem_post(&ctx->ready_to_write); // Signal ready to write when finished encoding
  return NULL;
}

void *writer_thread(void *arg)
{
  pipeline_ctx_t *ctx = (pipeline_ctx_t *)arg;
  for (int i = 0;; i++)
  {
    printf("Writing frame %d, ", i);
    sem_wait(&ctx->ready_to_write); // Wait for a frame ready for writing
    if (i == ctx->num_frames_total)
      break;
    int slot = i % 3;
    write_frame(ctx->cm, ctx->frames[slot]);
    printf("Done!\n");
    sem_post(&ctx->slot_free); // Signal free slot after writing
  }
  return NULL;
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

    // Precompute zig-zag indeces and write to table (defined in tables.h, declared in tables.c)
    zigzag_index[i] = (zigzag_V[i] * 8) + zigzag_U[i];

    // Init scale quant so we only do mul in hot-loop for quantization, also apply scale to deprecate scale_block
    cm->quant_scale[Y_COMPONENT][i] = (__fp16)((0.25f / cm->quanttbl[Y_COMPONENT][i]) * (float)scale_lut[zigzag_V[i]][zigzag_U[i]]);
    cm->quant_scale[U_COMPONENT][i] = (__fp16)((0.25f / cm->quanttbl[U_COMPONENT][i]) * (float)scale_lut[zigzag_V[i]][zigzag_U[i]]);
    cm->quant_scale[V_COMPONENT][i] = (__fp16)((0.25f / cm->quanttbl[V_COMPONENT][i]) * (float)scale_lut[zigzag_V[i]][zigzag_U[i]]);

    cm->dequant_scale[Y_COMPONENT][i] = (__fp16)(0.25f * cm->quanttbl[Y_COMPONENT][i] * (float)scale_lut[zigzag_V[i]][zigzag_U[i]]);
    cm->dequant_scale[U_COMPONENT][i] = (__fp16)(0.25f * cm->quanttbl[U_COMPONENT][i] * (float)scale_lut[zigzag_V[i]][zigzag_U[i]]);
    cm->dequant_scale[V_COMPONENT][i] = (__fp16)(0.25f * cm->quanttbl[V_COMPONENT][i] * (float)scale_lut[zigzag_V[i]][zigzag_U[i]]);
  }

  // Transposes dctlookup so we get idctlookup
  init_idct_lookup();

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
  setbuf(stdout, NULL); // To not buffer I/O, so prints appear

  // ctx for threads
  pipeline_ctx_t ctx = {0};
  ctx.num_frames_total = -1;
  pthread_t reader, encoder, writer;
  // Init semaphores for 3 in-flight frames
  sem_init(&ctx.slot_free, 0, 3);
  sem_init(&ctx.ready_to_encode, 0, 0);
  sem_init(&ctx.ready_to_write, 0, 0);

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

  int numframes = 0;

  // Allocate for 3 in-flight frames
  yuv_t *images[3];
  images[0] = alloc_input_image(cm);
  images[1] = alloc_input_image(cm);
  images[2] = alloc_input_image(cm);

  // Allocate for 3 in-flight frames
  struct frame *frames[3];
  frames[0] = create_frame(cm, images[0]);
  frames[1] = create_frame(cm, images[1]);
  frames[2] = create_frame(cm, images[2]);

  cm->framenum = 0;
  cm->frames_since_keyframe = 0;

  // Populate thread context struct for pipeline
  ctx.cm = cm;
  ctx.infile = infile;
  ctx.limit_numframes = limit_numframes;
  ctx.images[0] = images[0];
  ctx.images[1] = images[1];
  ctx.images[2] = images[2];
  ctx.frames[0] = frames[0];
  ctx.frames[1] = frames[1];
  ctx.frames[2] = frames[2];

  // Start threads for pipeline
  pthread_create(&reader, NULL, reader_thread, &ctx);
  pthread_create(&encoder, NULL, encoder_thread, &ctx);
  pthread_create(&writer, NULL, writer_thread, &ctx);

  // Join to mainthread
  pthread_join(reader, NULL);
  pthread_join(encoder, NULL);
  pthread_join(writer, NULL);

  // Cleanup
  destroy_frame(frames[0]);
  destroy_frame(frames[1]);
  destroy_frame(frames[2]);
  free_input_image(images[0]);
  free_input_image(images[1]);
  free_input_image(images[2]);
  free(cm);
  fclose(outfile);
  fclose(infile);

  return EXIT_SUCCESS;
}
