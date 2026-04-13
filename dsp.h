#ifndef C63_DSP_H_
#define C63_DSP_H_

#define ISQRT2 0.70710678118654f

#include <inttypes.h>
#include <arm_neon.h>

void init_idct_lookup();

void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data,
                         float16_t *quant_scale);

void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data,
                            float16_t *dequant_scale);

// Define as inline to remove function call overhead, as its called quite frequently
__attribute__((always_inline)) static inline uint16_t sad_block_8x8(uint8x8_t orig_rows[8], uint8_t *block2, int stride)
{
    // Row ptrs (avoid mul when loading rows from block2)
    uint8_t *row1 = block2 + stride;
    uint8_t *row2 = row1 + stride;
    uint8_t *row3 = row2 + stride;
    uint8_t *row4 = row3 + stride;
    uint8_t *row5 = row4 + stride;
    uint8_t *row6 = row5 + stride;
    uint8_t *row7 = row6 + stride;

    // Load rows from ref frame and accumulate SAD
    // Use two accumulators to break dependency chain for OOO execution
    uint16x8_t acc0 = vabdl_u8(orig_rows[0], vld1_u8(block2));
    uint16x8_t acc1 = vabdl_u8(orig_rows[1], vld1_u8(row1));
    acc0 = vabal_u8(acc0, orig_rows[2], vld1_u8(row2));
    acc1 = vabal_u8(acc1, orig_rows[3], vld1_u8(row3));
    acc0 = vabal_u8(acc0, orig_rows[4], vld1_u8(row4));
    acc1 = vabal_u8(acc1, orig_rows[5], vld1_u8(row5));
    acc0 = vabal_u8(acc0, orig_rows[6], vld1_u8(row6));
    acc1 = vabal_u8(acc1, orig_rows[7], vld1_u8(row7));

    // Horizontally add to get final SAD value
    return vaddvq_u16(vaddq_u16(acc0, acc1));
}

#endif /* C63_DSP_H_ */
