"""fp16 autocast with fp32 weights and a dynamic loss scale, reduced over a bf16 wire."""

PRECISION = "fp16"
REDUCE_DTYPE = "bf16"
