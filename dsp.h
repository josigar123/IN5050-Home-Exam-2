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

// __attribute__((always_inline)) static inline uint32x4_t sad_block_8x8_x4(uint8_t *orig, uint8_t *ref, int stride)
// {
//     // Load all 8 orig rows, shared across all 4 search positions
//     uint8x8_t orig_row0 = vld1_u8(orig);
//     uint8x8_t orig_row1 = vld1_u8(orig + stride);
//     uint8x8_t orig_row2 = vld1_u8(orig + 2 * stride);
//     uint8x8_t orig_row3 = vld1_u8(orig + 3 * stride);
//     uint8x8_t orig_row4 = vld1_u8(orig + 4 * stride);
//     uint8x8_t orig_row5 = vld1_u8(orig + 5 * stride);
//     uint8x8_t orig_row6 = vld1_u8(orig + 6 * stride);
//     uint8x8_t orig_row7 = vld1_u8(orig + 7 * stride);

//     // 4 accumulators for 4 search positions (x, x+1, x+2, x+3)
//     uint16x8_t acc0 = vabdl_u8(orig_row0, vld1_u8(ref));     // SAD for position x
//     uint16x8_t acc1 = vabdl_u8(orig_row0, vld1_u8(ref + 1)); // SAD for position x+1
//     uint16x8_t acc2 = vabdl_u8(orig_row0, vld1_u8(ref + 2)); // SAD for position x+2
//     uint16x8_t acc3 = vabdl_u8(orig_row0, vld1_u8(ref + 3)); // SAD for position x+3

//     acc0 = vabal_u8(acc0, orig_row1, vld1_u8(ref + stride));     // Accumulate SAD for position x
//     acc1 = vabal_u8(acc1, orig_row1, vld1_u8(ref + 1 + stride)); // Accumulate SAD for position x+1
//     acc2 = vabal_u8(acc2, orig_row1, vld1_u8(ref + 2 + stride)); // Accumulate SAD for position x+2
//     acc3 = vabal_u8(acc3, orig_row1, vld1_u8(ref + 3 + stride)); // Accumulate SAD for position x+3

//     acc0 = vabal_u8(acc0, orig_row2, vld1_u8(ref + 2 * stride));     // Accumulate SAD for position x
//     acc1 = vabal_u8(acc1, orig_row2, vld1_u8(ref + 1 + 2 * stride)); // Accumulate SAD for position x+1
//     acc2 = vabal_u8(acc2, orig_row2, vld1_u8(ref + 2 + 2 * stride)); // Accumulate SAD for position x+2
//     acc3 = vabal_u8(acc3, orig_row2, vld1_u8(ref + 3 + 2 * stride)); // Accumulate SAD for position x+3

//     acc0 = vabal_u8(acc0, orig_row3, vld1_u8(ref + 3 * stride));     // Accumulate SAD for position x
//     acc1 = vabal_u8(acc1, orig_row3, vld1_u8(ref + 1 + 3 * stride)); // Accumulate SAD for position x+1
//     acc2 = vabal_u8(acc2, orig_row3, vld1_u8(ref + 2 + 3 * stride)); // Accumulate SAD for position x+2
//     acc3 = vabal_u8(acc3, orig_row3, vld1_u8(ref + 3 + 3 * stride)); // Accumulate SAD for position x+3

//     acc0 = vabal_u8(acc0, orig_row4, vld1_u8(ref + 4 * stride));     // Accumulate SAD for position x
//     acc1 = vabal_u8(acc1, orig_row4, vld1_u8(ref + 1 + 4 * stride)); // Accumulate SAD for position x+1
//     acc2 = vabal_u8(acc2, orig_row4, vld1_u8(ref + 2 + 4 * stride)); // Accumulate SAD for position x+2
//     acc3 = vabal_u8(acc3, orig_row4, vld1_u8(ref + 3 + 4 * stride)); // Accumulate SAD for position x+3

//     acc0 = vabal_u8(acc0, orig_row5, vld1_u8(ref + 5 * stride));     // Accumulate SAD for position x
//     acc1 = vabal_u8(acc1, orig_row5, vld1_u8(ref + 1 + 5 * stride)); // Accumulate SAD for position x+1
//     acc2 = vabal_u8(acc2, orig_row5, vld1_u8(ref + 2 + 5 * stride)); // Accumulate SAD for position x+2
//     acc3 = vabal_u8(acc3, orig_row5, vld1_u8(ref + 3 + 5 * stride)); // Accumulate SAD for position x+3

//     acc0 = vabal_u8(acc0, orig_row6, vld1_u8(ref + 6 * stride));     // Accumulate SAD for position x
//     acc1 = vabal_u8(acc1, orig_row6, vld1_u8(ref + 1 + 6 * stride)); // Accumulate SAD for position x+1
//     acc2 = vabal_u8(acc2, orig_row6, vld1_u8(ref + 2 + 6 * stride)); // Accumulate SAD for position x+2
//     acc3 = vabal_u8(acc3, orig_row6, vld1_u8(ref + 3 + 6 * stride)); // Accumulate SAD for position x+3

//     acc0 = vabal_u8(acc0, orig_row7, vld1_u8(ref + 7 * stride));     // Accumulate SAD for position x
//     acc1 = vabal_u8(acc1, orig_row7, vld1_u8(ref + 1 + 7 * stride)); // Accumulate SAD for position x+1
//     acc2 = vabal_u8(acc2, orig_row7, vld1_u8(ref + 2 + 7 * stride)); // Accumulate SAD for position x+2
//     acc3 = vabal_u8(acc3, orig_row7, vld1_u8(ref + 3 + 7 * stride)); // Accumulate SAD for position x+3

//     uint32_t sads[4] = {
//         vaddvq_u16(acc0), // Final SAD for position x
//         vaddvq_u16(acc1), // Final SAD for position x+1
//         vaddvq_u16(acc2), // Final SAD for position x+2
//         vaddvq_u16(acc3)  // Final SAD for position x+3
//     };

//     return vld1q_u32(sads);
// }

#endif /* C63_DSP_H_ */
