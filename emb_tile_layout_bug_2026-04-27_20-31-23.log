# Embedding Tile Layout Bug Investigation

## Summary

Investigation into incorrect output from `ttnn.embedding` operation when used in a prefill/decode loop context. The bug manifests as zeros in embedding output rows 1-7, with only row 0 containing correct values.

**Root Cause:** The embedding operation's output tensor descriptor (`memref<8x1x!ttcore.tile>`) does not match how data is actually stored in memory. This is a **tt-mlir compiler bug** in tile layout generation, not a ttnn runtime bug.

## Problem Description

### Observed Behavior

When running embedding with:
- **Input indices:** shape `[8, 1]` with values `[0, 1, 2, 3, 3, 3, 3, 3]`
- **Weight table:** shape `[4, 32]` with 4 embedding vectors
- **Expected output:** shape `[8, 1, 32]` where each row contains the looked-up embedding

**Actual output:**
```
Row 0: [0.4863, 1.7109, ...]  ← CORRECT (weight[0])
Row 1: [0.0000, 0.0000, ...]  ← WRONG (should be weight[1])
Row 2: [0.0000, 0.0000, ...]  ← WRONG (should be weight[2])
Row 3-7: [0.0000, 0.0000, ...] ← WRONG (should be weight[3])
```

### Key Evidence

After a reshape from `[8, 1, 32]` to `[1, 8, 32]`, the **correct values appeared**:

```
Row 0: [0.4863, ...]  ← weight[0]
Row 1: [0.6914, ...]  ← weight[1] - NOW CORRECT
Row 2: [0.8164, ...]  ← weight[2] - NOW CORRECT
Row 3: [1.2344, ...]  ← weight[3] - NOW CORRECT
Row 4-7: [1.2344, ...] ← weight[3] - NOW CORRECT
```

This proves the data IS in the buffer correctly, but the tile layout descriptor is wrong.

## Root Cause Analysis

### MLIR Type Analysis

**Embedding output tensor:**
```mlir
tensor<8x1x32xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>,
  memref<8x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
```

**After reshape:**
```mlir
tensor<1x8x32xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>,
  memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
```

**Critical difference:** `memref<8x1x!ttcore.tile>` vs `memref<1x1x!ttcore.tile>`

### What's Happening

1. **Embedding** writes data correctly into the tile buffer in a specific physical layout
2. **`to_string()` for `[8, 1, 32]`** with `memref<8x1x!ttcore.tile>` interprets tile offsets incorrectly → reads zeros
3. **Reshape** physically re-lays out data from `8x1` tiles to `1x1` tiles, reading from correct positions
4. **`to_string()` for `[1, 8, 32]`** with `memref<1x1x!ttcore.tile>` now reads correctly

### Conclusion

The embedding op's output layout descriptor doesn't match the actual data layout. Operations consuming the tensor directly (before reshape) read from wrong memory positions.

This is a **tt-mlir compiler bug** in how it generates the embedding output's tile layout, NOT a ttnn runtime bug.

## Debugging Instrumentation Added

### Files Modified

All in `/localdev/jameszianxu/clean/tt-xla/third_party/tt-mlir/src/tt-mlir/runtime/lib/ttnn/operations/`:

#### 1. `eltwise/ternary/where.cpp`

```cpp
#include "ttnn/tensor/tensor_impl.hpp"
#include <iostream>

// In runEltwiseTernaryWhereOp():
if (std::getenv("PRINT_WHERE")) {
    LOG_INFO("Where op inputs in first/second/third:");
    ::ttnn::set_printoptions(::ttnn::TensorPrintProfile::Full);
    std::cout << "first: " << tt::tt_metal::tensor_impl::to_string(first) << std::endl;
    std::cout << "second: " << tt::tt_metal::tensor_impl::to_string(second) << std::endl;
    std::cout << "third: " << tt::tt_metal::tensor_impl::to_string(third) << std::endl;
}
// ... after ::ttnn::where() call:
if (std::getenv("PRINT_WHERE")) {
    LOG_INFO("Where op output:");
    ::ttnn::set_printoptions(::ttnn::TensorPrintProfile::Full);
    std::cout << "out: " << tt::tt_metal::tensor_impl::to_string(out) << std::endl;
}
```

**Enable with:** `PRINT_WHERE=1`

#### 2. `embedding/embedding.cpp`

```cpp
#include "ttnn/tensor/tensor_impl.hpp"
#include <iostream>

// Before embedding call:
if (std::getenv("PRINT_EMBEDDING")) {
    LOG_INFO("Embedding op inputs:");
    ::ttnn::set_printoptions(::ttnn::TensorPrintProfile::Full);
    std::cout << "input: " << tt::tt_metal::tensor_impl::to_string(input) << std::endl;
    std::cout << "weight: " << tt::tt_metal::tensor_impl::to_string(weight) << std::endl;
}
// After embedding call:
if (std::getenv("PRINT_EMBEDDING")) {
    LOG_INFO("Embedding op output:");
    ::ttnn::set_printoptions(::ttnn::TensorPrintProfile::Full);
    std::cout << "out: " << tt::tt_metal::tensor_impl::to_string(out) << std::endl;
}
```

**Enable with:** `PRINT_EMBEDDING=1`

#### 3. `data_movement/reshape.cpp`

```cpp
#include "ttnn/tensor/tensor_impl.hpp"
#include <iostream>

// Before reshape:
if (std::getenv("PRINT_RESHAPE")) {
    LOG_INFO("Reshape op input:");
    ::ttnn::set_printoptions(::ttnn::TensorPrintProfile::Full);
    std::cout << "in: " << tt::tt_metal::tensor_impl::to_string(in) << std::endl;
}
// Print target shape and after reshape:
if (std::getenv("PRINT_RESHAPE")) {
    std::cout << "Reshape target shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << (i + 1 < shape.size() ? ", " : "");
    }
    std::cout << "]" << std::endl;
    LOG_INFO("Reshape op output:");
    ::ttnn::set_printoptions(::ttnn::TensorPrintProfile::Full);
    std::cout << "out: " << tt::tt_metal::tensor_impl::to_string(out) << std::endl;
}
```

**Enable with:** `PRINT_RESHAPE=1`

#### 4. `data_movement/slice.cpp`

```cpp
#include "ttnn/tensor/tensor_impl.hpp"
#include <iostream>

// In runSliceStaticOp():
if (std::getenv("PRINT_SLICE")) {
    LOG_INFO("SliceStatic op input:");
    ::ttnn::set_printoptions(::ttnn::TensorPrintProfile::Full);
    std::cout << "in: " << tt::tt_metal::tensor_impl::to_string(in) << std::endl;
}
// Print params and output:
if (std::getenv("PRINT_SLICE")) {
    std::cout << "SliceStatic params: begins=[...], ends=[...], step=[...]" << std::endl;
    LOG_INFO("SliceStatic op output:");
    std::cout << "out: " << tt::tt_metal::tensor_impl::to_string(out) << std::endl;
}
```

**Enable with:** `PRINT_SLICE=1`

### Running with All Debug Output

```bash
PRINT_EMBEDDING=1 PRINT_RESHAPE=1 PRINT_SLICE=1 PRINT_WHERE=1 <your_test_command>
```

## Test Cases Added

### File: `tests/torch/ops/test_embedding.py`

#### 1. `test_embedding_small_vocab_repeated_indices`

Exact reproduction of the failing scenario:
```python
def test_embedding_small_vocab_repeated_indices(request):
    """Regression test for embedding with small vocabulary and repeated indices."""
    dtype = torch.bfloat16
    num_embeddings = 4
    embedding_dim = 32
    embed = Embedding(num_embeddings, embedding_dim, dtype=dtype)

    # Specific input pattern: indices [0, 1, 2, 3, 3, 3, 3, 3] with shape [8, 1]
    input_ids = torch.tensor([[0], [1], [2], [3], [3], [3], [3], [3]], dtype=torch.long)
    # ... run test
```

#### 2. `test_embedding_small_vocab`

Parametrized test for small vocabulary edge cases:
- `batch_size`: 1, 4, 8
- `num_embeddings`: 4, 8, 16
- `embedding_dim`: 32, 64

## Debugging Timeline

1. **Initial symptom:** `where` operation producing incorrect output (only first column correct)
2. **Added `where.cpp` instrumentation:** Revealed `third` input tensor was already corrupted
3. **Traced upstream:** Found `third` came from embedding → reshape → other ops
4. **Added `embedding.cpp` instrumentation:** Showed embedding output had zeros in rows 1-7
5. **Added `reshape.cpp` instrumentation:** **Key discovery** - reshape output had CORRECT values
6. **Root cause identified:** Tile layout descriptor mismatch, not data corruption

## Next Steps

1. **Investigate tt-mlir embedding lowering:**
   - Check how embedding op generates output tensor layout
   - Compare with other ops that work correctly
   - Look at `memref<8x1x!ttcore.tile>` vs `memref<1x1x!ttcore.tile>` generation

2. **Potential fixes:**
   - Fix embedding op to generate correct tile layout descriptor
   - Or ensure a layout-fixing reshape is always inserted after embedding

3. **Test validation:**
   - Run `test_embedding_small_vocab_repeated_indices` to verify fix
   - Run full test suite to check for regressions

## Related Files

- **Test file:** `tests/torch/ops/test_embedding.py`
- **Mechanism tests:** `tests/torch/models/deepseek_v4/test_deepseek_v4_mechanisms.py`
- **Loop tests:** `tests/torch/models/deepseek_v4/test_deepseek_v4_prefill_decode_loop.py`

## tt-metal Tensor Print API Reference

The old API (`tensor.print()` and `set_printoptions("full")`) is deprecated. Use:

```cpp
#include "ttnn/tensor/tensor_impl.hpp"

// Set print options (enum, not string)
::ttnn::set_printoptions(::ttnn::TensorPrintProfile::Full);

// Print tensor
std::cout << tt::tt_metal::tensor_impl::to_string(tensor) << std::endl;
```

Available `TensorPrintProfile` values:
- `Empty` - prints nothing
- `Short` - abbreviated (default, 4 elements per dim)
- `Full` - prints all values
