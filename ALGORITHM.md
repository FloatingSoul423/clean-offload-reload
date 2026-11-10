# Algorithm and evolution

## Problem

Long tool-using sessions leave large KV prefixes idle while external tools run.
Keeping every prefix in HBM limits concurrency. Offloading too much creates a
burst of host-to-device traffic when several tools return together, increasing
time to first token (TTFT). The scheduler must decide both how much KV may leave
HBM and when it must return.

## Shared-link service curve

For tool-waiting session `i`, let `deadline_i` be a conditional early-return
quantile and `missing_i` the bytes that must be restored. From the current time:

```text
C(t) = safety * B_available * t
D(t) = sum(missing_i for deadline_i <= t)
```

`B_available` is the minimum of the configured ceiling and the median of the
five most recent completed-transfer throughput samples. With one preemptible
shared H2D link, an admitted state is feasible only if:

```text
D(t) <= C(t) for every deadline prefix
```

The maximum additional bytes session `i` may evict are bounded by the minimum
unused capacity of every later prefix:

```text
max_evict_i = min(C(t) - D(t) for t >= deadline_i)
```

Eviction considers far deadlines first. Reload follows earliest deadline first
and is submitted in bounded chunks.

## v9.1 demand gating

The service curve is a safety ceiling, not an instruction to fill all available
PCIe capacity. v9.1 creates new eviction work only when measured HBM headroom
falls below the low watermark after pending allocation is included:

```text
if HBM_available - predicted_allocation < reserve_low:
    need = reserve_high - HBM_available + predicted_allocation
    new_evict = min(need, service_curve_safe_evictable)
else:
    new_evict = 0
```

Existing missing work may still reload when no new eviction is allowed. When
HBM telemetry is unavailable, v9.1 fails closed for new eviction. The runtime
also maintains a conservative deadline floor and performs reserve maintenance
only when its scheduler and transfer pipeline are idle.

## Evolution

### v8.1 runtime

The runtime milestone moved deadline-floor enforcement into SGLang and fixed
residency invariants around asynchronous reload, radix-tree mutation, producer
reservation, and host-resource release ordering. This is the required base for
all later controllers.

### v9.0 cumulative service curve

v9.0 replaced fixed per-request bandwidth with a global cumulative EDF
feasibility test. A single session can use the whole idle link, while many
sessions share one prefix budget. Its weakness was treating all safe capacity as
desirable eviction capacity, which increased H2D traffic under large-model
stress.

### v9.1 selected default

v9.1 separated how much must be evicted for HBM space from how much could be
evicted without missing a reload deadline. Demand gating answers the first; the
service curve bounds the second. A 5%/30% HBM hysteresis band, 0.5 GiB chunks,
measured bandwidth, and a one-second reload lead reduced repeated emergency
recovery in the tested configuration.

### Variants not selected

A Goodput-first v9.2 added foreground request gating, 0.125 GiB foreground
chunks, and prioritization around a fixed 250 ms threshold. It produced 29/46
SLO successes on one 14B seed but 24/48 on another. The two-seed total was 53/94
(56.4%), so the apparent gain was not stable enough to replace v9.1.

An aggressive peak-aware v9.3 doubled the look-ahead and shortened the idle
submission period. It reduced emergency evictions but increased H2D traffic and
dropped Goodput to 24/46 in its screening run. It was reverted. These variants
remain in the research repository history, not in this clean release.

## Where v9.1 fits

v9.1 targets multi-session, long-context inference where KV restores are much
larger than new tool observations and H2D bandwidth is shared. It is most useful
when tool waits provide time to schedule restore work and HBM pressure varies.
Native LRU remains simpler when the working set fits in HBM, tool returns are
not predictable, or transfer telemetry is unavailable.

## Known limitations

- The deadline is a probability quantile, not a hard completion time.
- HTTP acceptance confirms a residency target was queued, not that H2D finished.
  The client lacks a runtime-owned intent reconciler and exact in-flight byte
  accounting.
- The model assumes one preemptible shared H2D link. NUMA- or GPU-bound transfer
  paths need one service curve per link.
- The performance runs are targeted screenings, not a full independent
  multi-model evaluation. Re-run paired LRU and controller arms on the target
  hardware before drawing deployment conclusions.
