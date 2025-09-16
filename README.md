### Ant Mania – High-performance Simulation

This project simulates wandering ants on a colony graph. Two ants entering the same colony in the same tick destroy each other and the colony. The map graph updates accordingly (edges removed), potentially trapping ants.

### Goals and approach

- **Simulation latency focus**: The implementation prioritizes time spent stepping the simulation. Map parsing is separated from the hot loop. The built-in benchmark mode reports average per-run latencies.
- **Language**: Implemented in Rust for performance and safety.
- **Assumptions**:
  - Input map follows the described format: one colony per line, followed by up to 4 directional edges (`north|south|east|west=name`).
  - Colony names are unique and contain no `=` or whitespace.
  - Maximum degree per colony is 4.
  - On destruction, the colony and all incident edges are removed in the same tick. Ants targeting a just-destroyed colony die that tick.

### Key optimizations

- **Iterate only live ants**: Maintain a dense `live_ant_ids` list and an index map. Dead ants are removed with O(1) `swap_remove`, eliminating branches over dead ants and improving cache locality.
- **Compact neighbor list per colony**: Store neighbors as a fixed-size array `[u32; 4]` plus `neighbor_len`. On destruction, back-edges are removed via swap-with-last in O(degree). Ant move choice is then `rng % neighbor_len` with direct indexing—no per-tick filtering.
- **Sparse clearing**: Track only colonies touched in a tick and reset per-tick scratch (`in_count`, fighter pairs) for those colonies, avoiding full-vector clears.
- **Tight hot loop**: Inlined hot functions and RNG; minimized bounds checks where safe.
- **Removed redundant structures**: Dropped `ant_available`, `live_count`, and the original adjacency once compact neighbors are built.

### Usage

- Build:
  - `cargo build --release`

- Run a single simulation:
  - `cargo run --release -- <map_path> <num_ants> [seed]`
  - Example: `cargo run --release -- maps/hiveum_map_small.txt 100 42`
  - Output: fight log, then summary lines with total time, destroyed colonies, ants remaining.

- Benchmark mode (averages across trials):
  - `cargo run --release -- bench <map_path> <num_ants> <trials> [seed]`
  - Example: `cargo run --release -- bench maps/hiveum_map_small.txt 10 20`
  - Prints the parameters and average time across trials.

Notes:
- The timer includes both map load and simulation. On small maps this overhead is negligible, but you can adjust if you need strictly “post-load” simulation latency.
- If a run ends with remaining ants, it’s typically because the 10,000-step cap was reached with ants in separate components or unlucky random walks.

### Benchmarks (Apple MacBook Pro M4, 10-core CPU, 16GB RAM)

Small map (`maps/hiveum_map_small.txt`):

```
ants=2,   trials=10 → Average time: ~218 µs
ants=5,   trials=10 → Average time: ~300 µs
ants=10,  trials=10 → Average time: ~190 µs
ants=15,  trials=10 → Average time: ~254 µs
ants=20,  trials=10 → Average time: ~186 µs
ants=50,  trials=10 → Average time: ~413 µs
ants=100, trials=10 → Average time: ~112 µs
```

Medium map (`maps/hiveum_map_medium.txt`):

```
ants=500,   trials=100 → Average time: ~4.88 ms
ants=1000,  trials=100 → Average time: ~4.92 ms
ants=2000,  trials=100 → Average time: ~4.99 ms
ants=5000,  trials=100 → Average time: ~6.53 ms
ants=10000, trials=100 → Average time: ~36.75 ms
```

### Implementation details

- Map parsing builds a `Graph` with:
  - `names: Vec<String>`
  - `alive: Vec<bool>`
  - `neighbors: Vec<[u32; 4]>` and `neighbor_len: Vec<u8>`
- `AntSim` keeps per-ant `positions`, per-ant RNGs, and the live-ant list/index for O(1) removals.
- Per-tick scratch: `next_pos`, `in_count`, and fighter tracking, sparse-cleared via `touched_cols`.
- Fight resolution selects two incoming ants deterministically, logs the destruction, and marks the colony for removal.
- Destruction (`sever_colony`) clears the colony and removes back-edges from neighbors in O(1) amortized swaps.

### Development

- Toolchain: stable Rust, no external dependencies beyond the standard library.
- Build: `cargo build --release`
- Lint: uses the compiler’s warnings; code compiles clean.