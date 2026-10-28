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
//
// D3D12 compiles WGSL with FXC unless a host asks for DXC (wgpu translates it to HLSL
// with naga), and FXC shapes two things here:
//
// - FXC cannot write a private vector or array at an index it does not know: it unrolls
//   the loops around the write to make the index constant, and fails ("array reference
//   cannot be used as an l-value; not natively addressable"). So an invocation's four
//   items are four variables, and a bucket's slot is chosen with select().
// - naga zero-initialises workgroup memory from one invocation, and FXC's time for that
//   grows faster than the variable: one variable of 1024 words takes it seconds, four of
//   256 words half a second each, one of 64 words a twentieth (2026-10-28; the 11 KiB this
//   file had before took 26 s for sort_scatter). A word array inside a struct is stored
//   16 bytes at a time. So sort_scatter's 1024 words of workgroup memory are four
//   variables, s_mem0 .. s_mem3, and its 256 digit offsets four, s_off0 .. s_off3.

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

struct Counts {
  c: array<atomic<u32>, 256>,
};

struct Words256 {
  w: array<u32, 256>,
};

struct Words64 {
  w: array<u32, 64>,
};

// sort_histogram: the tile's digit counts.
var<workgroup> s_count: Counts;
// sort_scatter: first the scans (the first words in s_mem0, the second in s_mem1, and
// s_mem2, s_mem3 as the other buffer of the ping-pong), then the tile's items: word w of
// the tile in s_mem{w % 4}.w[w / 4], so that invocation lid's four are s_mem0 .. s_mem3 at lid.
var<workgroup> s_mem0: Words256;
var<workgroup> s_mem1: Words256;
var<workgroup> s_mem2: Words256;
var<workgroup> s_mem3: Words256;
// sort_scatter, per digit d in s_off{d % 4}.w[d / 4]: where the digit's first pair of the
// tile goes, minus its position in the sorted tile.
var<workgroup> s_off0: Words64;
var<workgroup> s_off1: Words64;
var<workgroup> s_off2: Words64;
var<workgroup> s_off3: Words64;

fn digit_of(key: u32) -> u32 {
  return (key >> D.shift) & (RADIX - 1u);
}

@compute @workgroup_size(256)
fn sort_histogram(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let seg = wid.z;
  let count = state[seg * ST_WORDS + ST_COUNT];
  let tile = wid.x;
  let base = seg * P.n_splats;
  atomicStore(&s_count.c[lid.x], 0u);
  workgroupBarrier();
  for (var i = 0u; i < SORT_ITEMS; i = i + 1u) {
    let idx = tile * SORT_TILE + i * WG + lid.x;
    if (idx < count) {
      atomicAdd(&s_count.c[digit_of(keys_in[base + idx])], 1u);
    }
  }
  workgroupBarrier();
  if (tile * SORT_TILE < count) {
    hist[(seg * RADIX + lid.x) * P.tiles_cap + tile] = atomicLoad(&s_count.c[lid.x]);
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

// Word w of the tile's items.
fn mem_put(w: u32, v: u32) {
  let i = w >> 2u;
  switch (w & 3u) {
    case 0u: { s_mem0.w[i] = v; }
    case 1u: { s_mem1.w[i] = v; }
    case 2u: { s_mem2.w[i] = v; }
    default: { s_mem3.w[i] = v; }
  }
}

fn off_put(d: u32, v: u32) {
  let i = d >> 2u;
  switch (d & 3u) {
    case 0u: { s_off0.w[i] = v; }
    case 1u: { s_off1.w[i] = v; }
    case 2u: { s_off2.w[i] = v; }
    default: { s_off3.w[i] = v; }
  }
}

fn off_get(d: u32) -> u32 {
  let i = d >> 2u;
  let k = d & 3u;
  return select(select(s_off0.w[i], s_off1.w[i], k == 1u),
                select(s_off2.w[i], s_off3.w[i], k == 3u), k >= 2u);
}

// Inclusive scan over the invocations of two words each (Hillis-Steele, ping-pong between
// s_mem0, s_mem1 and s_mem2, s_mem3). Returns the inclusive value; the workgroup totals
// are left in s_mem0.w[WG - 1] and s_mem1.w[WG - 1]. Call from uniform control flow, and
// put a barrier between reading the result and the next writes to s_mem*.
fn pair_scan(lid: u32, v: vec2u) -> vec2u {
  s_mem0.w[lid] = v.x;
  s_mem1.w[lid] = v.y;
  workgroupBarrier();
  for (var off = 1u; off < WG; off = off << 2u) {   // two steps per iteration, eight in all
    var x = vec2u(s_mem0.w[lid], s_mem1.w[lid]);
    if (lid >= off) {
      x = x + vec2u(s_mem0.w[lid - off], s_mem1.w[lid - off]);
    }
    s_mem2.w[lid] = x.x;
    s_mem3.w[lid] = x.y;
    workgroupBarrier();
    let off2 = off << 1u;
    var y = vec2u(s_mem2.w[lid], s_mem3.w[lid]);
    if (lid >= off2) {
      y = y + vec2u(s_mem2.w[lid - off2], s_mem3.w[lid - off2]);
    }
    s_mem0.w[lid] = y.x;
    s_mem1.w[lid] = y.y;
    workgroupBarrier();
  }
  return vec2u(s_mem0.w[lid], s_mem1.w[lid]);
}

// Four 16-bit bucket counts packed in a vec2u (x = buckets 0, 1; y = buckets 2, 3). Counts
// stay below 2^16 (a tile has 1024 items), so the packed fields never carry into each other.
fn bucket_count(packed: vec2u, q: u32) -> u32 {
  let word = select(packed.x, packed.y, q >= 2u);
  return (word >> (16u * (q & 1u))) & 0xffffu;
}

fn bucket_one(q: u32) -> vec2u {
  let bit = 1u << (16u * (q & 1u));
  return select(vec2u(bit, 0u), vec2u(0u, bit), q >= 2u);
}

// component q of v, without indexing the vector
fn pick(v: vec4u, q: u32) -> u32 {
  return select(select(v.x, v.y, q == 1u), select(v.z, v.w, q == 3u), q >= 2u);
}

// The item of tile position pos: its digit and the position. Positions past the segment
// get digit 255, so the stable local sort leaves them at the end of the tile.
fn tile_item(first: u32, pos: u32, valid: u32) -> u32 {
  var d = RADIX - 1u;
  if (pos < valid) {
    d = digit_of(keys_in[first + pos]);
  }
  return (d << 16u) | pos;
}

// The pair of item `it`, at position pos of the sorted tile, to its destination.
fn put_pair(first: u32, base: u32, pos: u32, it: u32, valid: u32) {
  if (pos < valid) {
    let src = first + (it & 0xffffu);
    let dst = base + off_get(it >> 16u) + pos;
    keys_out[dst] = keys_in[src];
    vals_out[dst] = vals_in[src];
  }
}

@compute @workgroup_size(256)
fn sort_scatter(@builtin(local_invocation_id) lid3: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let lid = lid3.x;
  let seg = wid.z;
  let count = state[seg * ST_WORDS + ST_COUNT];
  let tile = wid.x;
  let tile_base = tile * SORT_TILE;
  let base = seg * P.n_splats;
  let first = base + tile_base;
  var valid = 0u;
  if (count > tile_base) {
    valid = min(SORT_TILE, count - tile_base);
  }
  // Digit offsets, invocation lid for digit lid. The pair of digit d at position p of the
  // sorted tile goes to (digits below d in the segment) + (d in earlier tiles) + p - (digits
  // below d in this tile): one exclusive scan over the digits of the digit's count in the
  // segment minus its count in this tile (the next tile's prefix minus this one's, or in
  // the last tile the total minus it).
  let total = totals[seg * RADIX + lid];
  var prefix = 0u;
  var here = 0u;
  if (valid > 0u) {
    let row = (seg * RADIX + lid) * P.tiles_cap;
    prefix = hist[row + tile];
    var next = total;
    if (tile_base + SORT_TILE < count) {
      next = hist[row + tile + 1u];
    }
    here = next - prefix;
  }
  let v = total - here;
  off_put(lid, pair_scan(lid, vec2u(v, 0u)).x - v + prefix);
  // blocked arrangement: invocation lid holds tile positions 4 lid .. 4 lid + 3
  let p0 = SORT_ITEMS * lid;
  var it0 = tile_item(first, p0, valid);
  var it1 = tile_item(first, p0 + 1u, valid);
  var it2 = tile_item(first, p0 + 2u, valid);
  var it3 = tile_item(first, p0 + 3u, valid);
  // stable local sort by the digit: four 2-bit splits, least significant first
  for (var round = 0u; round < 4u; round = round + 1u) {
    let sh = 16u + 2u * round;
    let q0 = (it0 >> sh) & 3u;
    let q1 = (it1 >> sh) & 3u;
    let q2 = (it2 >> sh) & 3u;
    let q3 = (it3 >> sh) & 3u;
    let cnt = bucket_one(q0) + bucket_one(q1) + bucket_one(q2) + bucket_one(q3);
    let incl = pair_scan(lid, cnt);
    let tot = vec2u(s_mem0.w[WG - 1u], s_mem1.w[WG - 1u]);   // bucket totals (uniform read)
    let excl = incl - cnt;
    let t0 = bucket_count(tot, 0u);
    let t1 = bucket_count(tot, 1u);
    let t2 = bucket_count(tot, 2u);
    // where this invocation's first item of each bucket goes; its later ones follow it
    let at = vec4u(bucket_count(excl, 0u), t0 + bucket_count(excl, 1u),
                   t0 + t1 + bucket_count(excl, 2u), t0 + t1 + t2 + bucket_count(excl, 3u));
    let w0 = pick(at, q0);
    let w1 = pick(at, q1) + u32(q1 == q0);
    let w2 = pick(at, q2) + u32(q2 == q0) + u32(q2 == q1);
    let w3 = pick(at, q3) + u32(q3 == q0) + u32(q3 == q1) + u32(q3 == q2);
    workgroupBarrier();   // the scan is read everywhere before the items overwrite it
    mem_put(w0, it0);
    mem_put(w1, it1);
    mem_put(w2, it2);
    mem_put(w3, it3);
    workgroupBarrier();
    it0 = s_mem0.w[lid];
    it1 = s_mem1.w[lid];
    it2 = s_mem2.w[lid];
    it3 = s_mem3.w[lid];
    workgroupBarrier();   // the next round's scan overwrites the items
  }
  put_pair(first, base, p0, it0, valid);
  put_pair(first, base, p0 + 1u, it1, valid);
  put_pair(first, base, p0 + 2u, it2, valid);
  put_pair(first, base, p0 + 3u, it3, valid);
}
