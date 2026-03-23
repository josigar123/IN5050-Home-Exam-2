#ifndef C63_DSP_H_
#define C63_DSP_H_

#define ISQRT2 0.70710678118654f

#include <inttypes.h>
#include <arm_neon.h>

void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data,
                         float *quant_scale);

void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data,
                            uint8_t *quant_tbl);

// Define as inline to remove function call overhead, as its called quite frequently
__attribute__((always_inline)) static inline uint16_t sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride)
{
    uint16x8_t acc = vabdl_u8(vld1_u8(block1 + 0 * stride), // SAD first row
                              vld1_u8(block2 + 0 * stride));

    // Load rows from blocks and accumulate SAD
    acc = vabal_u8(acc, vld1_u8(block1 + 1 * stride), vld1_u8(block2 + 1 * stride));
    acc = vabal_u8(acc, vld1_u8(block1 + 2 * stride), vld1_u8(block2 + 2 * stride));
    acc = vabal_u8(acc, vld1_u8(block1 + 3 * stride), vld1_u8(block2 + 3 * stride));
    acc = vabal_u8(acc, vld1_u8(block1 + 4 * stride), vld1_u8(block2 + 4 * stride));
    acc = vabal_u8(acc, vld1_u8(block1 + 5 * stride), vld1_u8(block2 + 5 * stride));
    acc = vabal_u8(acc, vld1_u8(block1 + 6 * stride), vld1_u8(block2 + 6 * stride));
    acc = vabal_u8(acc, vld1_u8(block1 + 7 * stride), vld1_u8(block2 + 7 * stride));

    // Horizontally add to get final SAD value
    return vaddvq_u16(acc);
}

#endif /* C63_DSP_H_ */
