# `fill_pad` program-cache stale-address bug (Gemma accuracy regression)

## Summary

After `tt-metal` commit **`9cb25bb`** ("Improve FillPadDeviceOperation performance", #42673),
the Gemma model shows an accuracy drop. Disabling the program cache makes it disappear.

Root cause: the rewritten **`FillPadL1ShardedProgramFactory`** fails to re-patch the
shard's L1 base address on program-cache hits, because its "slot not created" sentinel
is the literal value `0` — which is also a **valid `KernelHandle`**. The reader kernel is
the first kernel created on the program, so it legitimately gets handle `0`, and the
cache-hit update path skips it. On every cache hit the op then reads/writes the *first*
tensor's L1 address instead of the current input's, so padding is never filled (and the
stale location is clobbered).

This only affects **width/L1-sharded** tensors. DRAM/interleaved tensors use a different
factory (`FillPadProgramFactory`) whose override has no sentinel collision and works fine.

## The bug

`FillPadL1ShardedProgramFactory` creates one reader/writer kernel per `has_right_pad`
value (slot `0` = no right pad, slot `1` = right pad). In `create()`, the reader kernel is
the first `CreateKernel` call on the program, so it gets `KernelHandle 0`:

```cpp
for (uint32_t rp_idx = 0; rp_idx <= 1; ++rp_idx) {
    if (rw_ranges[rp_idx].empty()) continue;
    reader_kernel_ids[rp_idx] = CreateKernel(...);  // first kernel -> handle 0
}
```

On a program-cache hit, `override_runtime_arguments` must re-patch runtime arg `[0]`
(the shard's L1 base address) for the new input tensor — but it skips any slot whose
kernel id is `0`:

```cpp
void FillPadL1ShardedProgramFactory::override_runtime_arguments(...) {
    const uint32_t buf_addr = static_cast<uint32_t>(tensor_args.input.buffer()->address());
    ...
    for (uint32_t rp_idx = 0; rp_idx <= 1; ++rp_idx) {
        if (sv.reader_kernel_ids[rp_idx] == 0) {   // BUG: 0 is a VALID KernelHandle
            continue;
        }
        ...
        rrt[core.x][core.y][0] = buf_addr;          // never reached for the ID-0 slot
        wrt[core.x][core.y][0] = buf_addr;
    }
}
```

The `== 0` test is intended to mean "this `rp_idx` slot was never created," but it
collides with the real handle `0`.

## Why it hits Gemma's rms-norm exactly

- Gemma's rms-norm input is `16 x 2304`, width-sharded in L1.
- `W = 2304` is a multiple of 32, so `has_right_pad = (W % 32 != 0) = false` for **every** core.
- Only slot `0` is used, and its reader id is exactly `0`.

So `override_runtime_arguments` `continue`s past the only slot in use and **never updates
the shard base address** on cache hits. Reader/writer keep the L1 base address from the
first (cache-miss) tensor, `fill_pad` operates on the wrong tensor, the current input's
padding (rows 16–31) stays unfilled, and `mean`/`sum` reduce over garbage padding →
accuracy drop.

Disabling the program cache hides the bug because every call becomes a cache miss and
re-runs `create()` with the correct address.

## Evidence

- **Localized repro** (`repro_localize.py`): with program cache on and several live
  tensors at distinct L1 addresses, WIDTH_SHARDED L1 cache hits leave padding unfilled
  (`bottom_filled=False`); DRAM interleaved is correct for all calls.

  ```
  === WIDTH_SHARDED L1 ===
    out#0 addr=1490944 valid=True bottom_filled=True
    out#1 addr=1482752 valid=True bottom_filled=False  <-- PADDING NOT FILLED
    out#2 addr=1474560 valid=True bottom_filled=False  <-- PADDING NOT FILLED
    ...
  === DRAM interleaved ===
    out#0..4 valid=True bottom_filled=True
  ```

- DRAM uses `FillPadProgramFactory`, whose override unconditionally patches arg `[0]` for
  all cores (no sentinel collision):

  ```cpp
  for (...) {
      reader_runtime_args[core.x][core.y][0] = tens_buffer->address();
      writer_runtime_args[core.x][core.y][0] = tens_buffer->address();
  }
  ```

- **Debug log** (`test.log`, built with `-DTT_METAL_ENABLE_LOGGING=ON`): 42
  `FillPadDeviceOperation` launches, all sharing program hash `1267660898`; first is a
  cache miss, the remaining 41 are cache hits — all `fill_value=0`, `WIDTH_SHARDED` L1.

- **Bisect alignment**: the built binary is at `75ad1dc`, which contains `9cb25bb`. The
  only fill_pad change in between (`f7045af`, reshape pad_value) doesn't touch the
  override; line 652 of the running code still has the `== 0` sentinel.

## Fix

Don't use `0` as the "not created" sentinel for a `KernelHandle`. Initialize the slot ids
to a real sentinel (e.g. `-1` / `std::optional`) and test against that, or gate on the
same condition `create()` used (`rw_ranges[rp_idx].empty()`):

```cpp
for (uint32_t rp_idx = 0; rp_idx <= 1; ++rp_idx) {
    if (sv.reader_kernel_ids[rp_idx] == kUncreated) continue;  // not literal 0
    ...
}
```

## Upstream status — already fixed

Fixed by commit **`1ff24daf07b`** — "data_movement: migrate transpose + permute +
untilize + untilize_with_unpadding to ProgramDescriptor (#45071)", Diego Gomez,
**Mon May 25 2026**.

The fix is incidental to that PR's headline: it also migrated `fill_pad` to the
`ProgramDescriptor` framework, which removed the hand-written
`override_runtime_arguments` (and its `== 0` sentinel) entirely. The new factory:

- uses a real slot sentinel `std::array<int, 2> reader_kernel_idx = {-1, -1};`, and
- passes the `Buffer*` directly via `emplace_runtime_args({tens_buffer, ...})`, which the
  descriptor framework auto-registers as a `BufferBinding` and patches on every cache hit.

The built binary `75ad1dc` predates `1ff24daf07b`, so it still carries the bug.

## Commit references

- Introduced bug: `9cb25bbbbe6` — Improve FillPadDeviceOperation performance (#42673), Nathan Maurice
- Fixed upstream: `1ff24daf07b` — migrate ... to ProgramDescriptor (#45071), Diego Gomez, 2026-05-25
- Built/under-test binary: `75ad1dc18e6` (contains the bug; predates the fix)
