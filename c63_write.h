#ifndef C63_WRITE_H_
#define C63_WRITE_H_

#include "c63.h"

// Declaration
// Add frame to signature, this is to avoid a race when encoding reads from refframe
void write_frame(struct c63_common *cm, struct frame *frame);

#endif /* C63_WRITE_H_ */
