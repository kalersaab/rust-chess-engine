const INPUT: u32 = 768u;
const HIDDEN: u32 = 32768u;
const MAX_ACTIVE: u32 = 32u;
const SENTINEL: u32 = 0xFFFFFFFFu;

struct Params {
    batch: u32,
}

@group(0) @binding(0) var<storage, read> features: array<u32>;
@group(0) @binding(1) var<storage, read> input_w: array<f32>;
@group(0) @binding(2) var<storage, read> input_bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> hidden: array<f32>;
@group(0) @binding(4) var<storage, read> output_w: array<f32>;
@group(0) @binding(5) var<storage, read> output_bias: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn pass1(@builtin(global_invocation_id) gid: vec3u) {
    let id = gid.x;
    if (id >= params.batch * HIDDEN) {
        return;
    }
    let item = id / HIDDEN;
    let h = id % HIDDEN;
    var acc = input_bias[h];
    for (var k = 0u; k < MAX_ACTIVE; k += 1u) {
        let f = features[item * MAX_ACTIVE + k];
        if (f == SENTINEL) {
            break;
        }
        acc = acc + input_w[f * HIDDEN + h];
    }
    hidden[item * HIDDEN + h] = max(acc, 0.0);
}

var<workgroup> sh_sum: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn pass2(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wgid: vec3u,
) {
    let item = wgid.x;
    if (item >= params.batch) {
        return;
    }
    let base = item * HIDDEN;
    var s = 0.0;
    for (var i = lid.x; i < HIDDEN; i += 256u) {
        s = s + hidden[base + i] * output_w[i];
    }
    sh_sum[lid.x] = s;
    workgroupBarrier();
    var stride = 128u;
    while (stride > 0u) {
        if (lid.x < stride) {
            sh_sum[lid.x] = sh_sum[lid.x] + sh_sum[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (lid.x == 0u) {
        out[item] = sh_sum[0] + output_bias[0];
    }
}