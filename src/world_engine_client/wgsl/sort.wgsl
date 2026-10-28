// Stable LSD radix sort of (key, value) pairs, segmented: one segment per camera
// of a batch, workgroup_id.z = segment, so one dispatch chain sorts every camera
// of the batch (design doc 16 §9.2 step 4). Four passes of 8 bits; each pass is
// reduce-then-scan:
//
//   sort_histogram  per tile of SORT_TILE keys: digit counts
//                   -> hist[(segment * RADIX + digit) * tiles_cap + tile]
//   sort_scan       one workgroup per (digit, segment): exclusive prefix over the tiles,
//                   in place, and the digit total
//   sort_scatter    per tile: global digit offsets, a stable local sort of the tile by the
//                   digit (four 2-bit splits), then each pair to its destination
//
// A segment holds n_splats pairs; only its first state[segment * ST_WORDS + ST_COUNT]
// take part, anything after them is ignored. The x size of the dispatches covers the
// largest segment of the batch; tiles past a segment's end do nothing. Ranks come
// from prefix sums, never from atomic order, and no workgroup waits on another (no
// decoupled look-back), so the output is the same on every run and every adapter,
// and equal keys keep their input order.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> cams: array<Camera, MAX_CAMS>;   // unused; shared layout
@group(0) @binding(2) var<uniform> D: Dispatch;     // D.shift = 0, 8, 16, 24

@group(1) @binding(0) var<storage, read> keys_in: array<u32>;
@group(1) @binding(1) var<storage, read> vals_in: array<u32>;
@group(1) @binding(2) var<storage, read_write> keys_out: array<u32>;
@group(1) @binding(3) var<storage, read_write> vals_out: array<u32>;
@group(1) @binding(4) var<storage, read_write> hist: array<u32>;     // segments * RADIX * tiles_cap
@group(1) @binding(5) var<storage, read_write> totals: array<u32>;   // segments * RADIX
@group(1) @binding(6) var<storage, read> state: array<u32>;

var<workgroup> s_count: array<atomic<u32>, 256>;
var<workgroup> s_items: array<u32, 1024>;     // (digit << 16) | position in the tile
var<workgroup> s_split: array<vec2u, 512>;    // two buffers of per-invocation bucket counts
var<workgroup> s_base: array<u32, 256>;
var<workgroup> s_start: array<u32, 256>;

fn digit_of(key: u32) -> u32 {
  return (key >> D.shift) & (RADIX - 1u);
}

@compute @workgroup_size(256)
fn sort_histogram(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let seg = wid.z;
  let count = state[seg * ST_WORDS + ST_COUNT];
  let tile = wid.x;
  let base = seg * P.n_splats;
  atomicStore(&s_count[lid.x], 0u);
  workgroupBarrier();
  for (var i = 0u; i < SORT_ITEMS; i = i + 1u) {
    let idx = tile * SORT_TILE + i * WG + lid.x;
    if (idx < count) {
      atomicAdd(&s_count[digit_of(keys_in[base + idx])], 1u);
    }
  }
  workgroupBarrier();
  if (tile * SORT_TILE < count) {
    hist[(seg * RADIX + lid.x) * P.tiles_cap + tile] = atomicLoad(&s_count[lid.x]);
  }
}

@compute @workgroup_size(256)
fn sort_scan(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let digit = wid.x;
  let seg = wid.z;
  let n_tiles = (state[seg * ST_WORDS + ST_COUNT] + SORT_TILE - 1u) / SORT_TILE;
  let per = (n_tiles + WG - 1u) / WG;
  let begin = min(lid.x * per, n_tiles);
  let end = min(begin + per, n_tiles);
  let row = (seg * RADIX + digit) * P.tiles_cap;
  var sum = 0u;
  for (var t = begin; t < end; t = t + 1u) {
    sum = sum + hist[row + t];
  }
  let incl = wg_inclusive_scan(lid.x, sum);
  var run = incl - sum;
  for (var t = begin; t < end; t = t + 1u) {
    let c = hist[row + t];
    hist[row + t] = run;
    run = run + c;
  }
  if (lid.x == 0u) {
    totals[seg * RADIX + digit] = wg_scan_total();
  }
}

// Inclusive scan over the invocations of four 16-bit bucket counts packed in a
// vec2u (x = buckets 0, 1; y = buckets 2, 3). Counts stay below 2^16 (a tile has
// 1024 items), so the packed fields never carry into each other. Returns the
// inclusive value; the workgroup total is left in s_split[total_slot()].
fn split_scan(lid: u32, v: vec2u) -> vec2u {
  s_split[lid] = v;
  workgroupBarrier();
  var src = 0u;
  for (var off = 1u; off < WG; off = off << 1u) {
    var x = s_split[src * WG + lid];
    if (lid >= off) {
      x = x + s_split[src * WG + lid - off];
    }
    s_split[(1u - src) * WG + lid] = x;
    workgroupBarrier();
    src = 1u - src;
  }
  return s_split[src * WG + lid];   // eight steps: src is back to 0
}

fn bucket_count(packed: vec2u, q: u32) -> u32 {
  let word = select(packed.x, packed.y, q >= 2u);
  return (word >> (16u * (q & 1u))) & 0xffffu;
}

@compute @workgroup_size(256)
fn sort_scatter(@builtin(local_invocation_id) lid3: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let lid = lid3.x;
  let seg = wid.z;
  let count = state[seg * ST_WORDS + ST_COUNT];
  let tile = wid.x;
  let tile_base = tile * SORT_TILE;
  let base = seg * P.n_splats;
  var valid = 0u;
  if (count > tile_base) {
    valid = min(SORT_TILE, count - tile_base);
  }
  // where each digit starts for this tile: digits before it, then this digit in earlier tiles
  let total = totals[seg * RADIX + lid];
  let below = wg_inclusive_scan(lid, total) - total;
  var prefix = 0u;
  if (valid > 0u) {
    prefix = hist[(seg * RADIX + lid) * P.tiles_cap + tile];
  }
  s_base[lid] = below + prefix;
  // blocked arrangement: invocation lid holds tile positions 4*lid .. 4*lid+3. An item is its
  // digit and its position; positions past the segment get digit 255, so the stable local sort
  // leaves them at the end of the tile.
  var it: array<u32, 4>;
  for (var i = 0u; i < SORT_ITEMS; i = i + 1u) {
    let pos = lid * SORT_ITEMS + i;
    var d = RADIX - 1u;
    if (pos < valid) {
      d = digit_of(keys_in[base + tile_base + pos]);
    }
    it[i] = (d << 16u) | pos;
  }
  // stable local sort by the digit: four 2-bit splits, least significant first
  for (var round = 0u; round < 4u; round = round + 1u) {
    let sh = 16u + 2u * round;
    var cnt = vec2u(0u);
    for (var i = 0u; i < SORT_ITEMS; i = i + 1u) {
      let q = (it[i] >> sh) & 3u;
      cnt = cnt + select(vec2u(1u << (16u * (q & 1u)), 0u), vec2u(0u, 1u << (16u * (q & 1u))), q >= 2u);
    }
    let incl = split_scan(lid, cnt);
    let tot = s_split[WG - 1u];   // workgroup bucket totals (uniform read)
    let excl = incl - cnt;
    let t0 = bucket_count(tot, 0u);
    let t1 = bucket_count(tot, 1u);
    let t2 = bucket_count(tot, 2u);
    var run = vec4u(bucket_count(excl, 0u), t0 + bucket_count(excl, 1u),
                    t0 + t1 + bucket_count(excl, 2u), t0 + t1 + t2 + bucket_count(excl, 3u));
    for (var i = 0u; i < SORT_ITEMS; i = i + 1u) {
      let q = (it[i] >> sh) & 3u;
      s_items[run[q]] = it[i];
      run[q] = run[q] + 1u;
    }
    workgroupBarrier();
    for (var i = 0u; i < SORT_ITEMS; i = i + 1u) {
      it[i] = s_items[lid * SORT_ITEMS + i];
    }
    workgroupBarrier();   // s_split and s_items are reused by the next round
  }
  // first position of each digit in the sorted tile
  for (var i = 0u; i < SORT_ITEMS; i = i + 1u) {
    let pos = lid * SORT_ITEMS + i;
    if (pos < valid) {
      let d = it[i] >> 16u;
      if (pos == 0u || (s_items[pos - 1u] >> 16u) != d) {
        s_start[d] = pos;
      }
    }
  }
  workgroupBarrier();
  for (var i = 0u; i < SORT_ITEMS; i = i + 1u) {
    let pos = lid * SORT_ITEMS + i;
    if (pos < valid) {
      let d = it[i] >> 16u;
      let src = base + tile_base + (it[i] & 0xffffu);
      let dst = base + s_base[d] + (pos - s_start[d]);
      keys_out[dst] = keys_in[src];
      vals_out[dst] = vals_in[src];
    }
  }
}
