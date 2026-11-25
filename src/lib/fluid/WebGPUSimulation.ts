import { FLUID_CELL, AIR_CELL, SOLID_CELL, type FlipFluid } from './FlipFluid';
import { GPUBufferUsage, GPUShaderStage, GPUMapMode } from '../gpu-types';

// ============================================================
// WebGPU Compute Simulation — GPU FLIP pipeline
// ============================================================
// Mirrors FlipFluid.simulate() with parallel-friendly kernels:
//   integrate            per particle
//   buildBuckets         per particle  (fixed-capacity spatial hash)
//   push / pushCommit    per particle  (Jacobi particle separation + color diffusion)
//   walls                per particle
//   p2g                  per particle  (fixed-point atomic scatter to grid)
//   gridNormalize        per cell      (u/v normalize, cell types, density, BCs)
//   pressure red/black   per cell      (red-black Gauss-Seidel with over-relaxation)
//   g2p                  per particle  (PIC/FLIP blend)
//   colors               per particle  (foam / return to base)
//   cellColors           per cell      (only for grid display)
//
// Data is packed into 7 storage buffers so it fits the default
// maxStorageBuffersPerShaderStage limit of 8 (common on mobile).
//
// The GPU buffers are authoritative. Particle state is read back
// asynchronously into the FlipFluid arrays so CPU-side interaction
// (hit-testing, add/remove) can work on a recent copy; after the CPU
// edits particles it calls syncParticles() to push them back.
// ============================================================

const WG = 64;
const BUCKET_CAPACITY = 16;
const FIXED_POINT_SCALE = 65536.0;

// Field offsets (in units of numCells) inside the packed grid buffer
const GRID_U = 0;
const GRID_V = 1;
const GRID_PREV_U = 2;
const GRID_PREV_V = 3;
const GRID_DENSITY = 4;
const GRID_S = 5;
const GRID_TYPE = 6;
const GRID_COLOR = 7; // 3 floats per cell
const GRID_FIELDS = 10;

// Field offsets (in units of numCells) inside the packed atomic accumulator
const ACC_U = 0;
const ACC_V = 1;
const ACC_DU = 2;
const ACC_DV = 3;
const ACC_DENSITY = 4;
const ACC_COUNT = 5;
const ACC_FIELDS = 6;

const shaderWGSL = /* wgsl */ `
struct Params {
    numParticles: u32,
    fNumX: u32,
    fNumY: u32,
    pNumX: u32,
    pNumY: u32,
    compensateDrift: u32,
    maxParticles: u32,
    numCells: u32,
    h: f32,
    fInvSpacing: f32,
    pInvSpacing: f32,
    dt: f32,
    gravityX: f32,
    gravityY: f32,
    damping: f32,
    flipRatio: f32,
    overRelaxation: f32,
    particleRadius: f32,
    restDensity: f32,
    colorDiffusion: f32,
    foamReturnRate: f32,
    baseR: f32,
    baseG: f32,
    baseB: f32,
    foamR: f32,
    foamG: f32,
    foamB: f32,
    _pad: f32,
}

const FLUID_CELL: f32 = ${FLUID_CELL.toFixed(1)};
const AIR_CELL: f32 = ${AIR_CELL.toFixed(1)};
const SOLID_CELL: f32 = ${SOLID_CELL.toFixed(1)};
const BUCKET_CAPACITY: u32 = ${BUCKET_CAPACITY}u;
const FP: f32 = ${FIXED_POINT_SCALE.toFixed(1)};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> pos: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> vel: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> color: array<f32>;
// [0, 2*maxParticles): separated positions, then 3*maxParticles diffused colors
@group(0) @binding(4) var<storage, read_write> tmp: array<f32>;
@group(0) @binding(5) var<storage, read_write> acc: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> grid: array<f32>;
// [0, pNumCells): bucket counts, then pNumCells * BUCKET_CAPACITY particle ids
@group(0) @binding(7) var<storage, read_write> buckets: array<atomic<u32>>;

fn gi(field: u32, cell: u32) -> u32 { return field * params.numCells + cell; }

fn toFixed(x: f32) -> i32 { return i32(round(x * FP)); }
fn fromFixed(x: i32) -> f32 { return f32(x) / FP; }

fn readColor(i: u32) -> vec3f { return vec3f(color[3u * i], color[3u * i + 1u], color[3u * i + 2u]); }
fn writeColor(i: u32, c: vec3f) {
    color[3u * i] = c.x;
    color[3u * i + 1u] = c.y;
    color[3u * i + 2u] = c.z;
}

fn pNumCells() -> u32 { return params.pNumX * params.pNumY; }

fn bucketOf(p: vec2f) -> u32 {
    let xi = u32(clamp(i32(floor(p.x * params.pInvSpacing)), 0, i32(params.pNumX) - 1));
    let yi = u32(clamp(i32(floor(p.y * params.pInvSpacing)), 0, i32(params.pNumY) - 1));
    return xi * params.pNumY + yi;
}

@compute @workgroup_size(${WG})
fn integrate(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.numParticles) { return; }
    var vl = vel[i];
    vl += params.dt * vec2f(params.gravityX, params.gravityY);
    vl *= params.damping;
    vel[i] = vl;
    pos[i] += vl * params.dt;
}

@compute @workgroup_size(${WG})
fn buildBuckets(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.numParticles) { return; }
    let b = bucketOf(pos[i]);
    let slot = atomicAdd(&buckets[b], 1u);
    if (slot < BUCKET_CAPACITY) {
        atomicStore(&buckets[pNumCells() + b * BUCKET_CAPACITY + slot], i);
    }
}

// Jacobi variant of FlipFluid.pushParticlesApart: each particle only moves
// itself (by half the overlap, as the CPU version does per pair).
@compute @workgroup_size(${WG})
fn push(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.numParticles) { return; }

    let p = pos[i];
    let c0 = readColor(i);
    var delta = vec2f(0.0);
    var c = c0;

    let minDist = 2.0 * params.particleRadius;
    let minDist2 = minDist * minDist;

    let pxi = i32(floor(p.x * params.pInvSpacing));
    let pyi = i32(floor(p.y * params.pInvSpacing));
    let x0 = max(pxi - 1, 0);
    let y0 = max(pyi - 1, 0);
    let x1 = min(pxi + 1, i32(params.pNumX) - 1);
    let y1 = min(pyi + 1, i32(params.pNumY) - 1);
    let idBase = pNumCells();

    for (var xi = x0; xi <= x1; xi++) {
        for (var yi = y0; yi <= y1; yi++) {
            let b = u32(xi) * params.pNumY + u32(yi);
            let n = min(atomicLoad(&buckets[b]), BUCKET_CAPACITY);
            for (var k = 0u; k < n; k++) {
                let id = atomicLoad(&buckets[idBase + b * BUCKET_CAPACITY + k]);
                if (id == i) { continue; }
                let d = pos[id] - p;
                let d2 = dot(d, d);
                if (d2 > minDist2 || d2 == 0.0) { continue; }
                let dist = sqrt(d2);
                delta -= d * (0.5 * (minDist - dist) / dist);
                c += ((c0 + readColor(id)) * 0.5 - c0) * params.colorDiffusion;
            }
        }
    }

    let np = p + delta;
    tmp[2u * i] = np.x;
    tmp[2u * i + 1u] = np.y;
    let cBase = 2u * params.maxParticles + 3u * i;
    tmp[cBase] = c.x;
    tmp[cBase + 1u] = c.y;
    tmp[cBase + 2u] = c.z;
}

@compute @workgroup_size(${WG})
fn pushCommit(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.numParticles) { return; }
    pos[i] = vec2f(tmp[2u * i], tmp[2u * i + 1u]);
    let cBase = 2u * params.maxParticles + 3u * i;
    writeColor(i, vec3f(tmp[cBase], tmp[cBase + 1u], tmp[cBase + 2u]));
}

@compute @workgroup_size(${WG})
fn walls(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.numParticles) { return; }
    let h = params.h;
    let r = params.particleRadius;
    let minX = h + r;
    let maxX = f32(params.fNumX - 1u) * h - r;
    let minY = h + r;
    let maxY = f32(params.fNumY - 1u) * h - r;

    var p = pos[i];
    var vl = vel[i];
    if (p.x < minX) { p.x = minX; vl.x = 0.0; }
    if (p.x > maxX) { p.x = maxX; vl.x = 0.0; }
    if (p.y < minY) { p.y = minY; vl.y = 0.0; }
    if (p.y > maxY) { p.y = maxY; vl.y = 0.0; }
    pos[i] = p;
    vel[i] = vl;
}

struct Stencil {
    nr0: u32,
    nr1: u32,
    nr2: u32,
    nr3: u32,
    d0: f32,
    d1: f32,
    d2: f32,
    d3: f32,
}

// Same index/weight computation as FlipFluid.transferVelocities.
fn velocityStencil(p: vec2f, component: u32) -> Stencil {
    let n = params.fNumY;
    let h = params.h;
    let h1 = params.fInvSpacing;
    let h2 = 0.5 * h;
    let dx = select(0.0, h2, component == 1u);
    let dy = select(h2, 0.0, component == 1u);

    let x = clamp(p.x, h, f32(params.fNumX - 1u) * h);
    let y = clamp(p.y, h, f32(params.fNumY - 1u) * h);

    let x0 = min(u32(floor((x - dx) * h1)), params.fNumX - 2u);
    let tx = ((x - dx) - f32(x0) * h) * h1;
    let x1 = min(x0 + 1u, params.fNumX - 1u);
    let y0 = min(u32(floor((y - dy) * h1)), params.fNumY - 2u);
    let ty = ((y - dy) - f32(y0) * h) * h1;
    let y1 = min(y0 + 1u, params.fNumY - 1u);

    let sx = 1.0 - tx;
    let sy = 1.0 - ty;

    var st: Stencil;
    st.nr0 = x0 * n + y0;
    st.nr1 = x1 * n + y0;
    st.nr2 = x1 * n + y1;
    st.nr3 = x0 * n + y1;
    st.d0 = sx * sy;
    st.d1 = tx * sy;
    st.d2 = tx * ty;
    st.d3 = sx * ty;
    return st;
}

fn scatter(valueField: u32, weightField: u32, st: Stencil, value: f32) {
    atomicAdd(&acc[gi(valueField, st.nr0)], toFixed(value * st.d0));
    atomicAdd(&acc[gi(valueField, st.nr1)], toFixed(value * st.d1));
    atomicAdd(&acc[gi(valueField, st.nr2)], toFixed(value * st.d2));
    atomicAdd(&acc[gi(valueField, st.nr3)], toFixed(value * st.d3));
    atomicAdd(&acc[gi(weightField, st.nr0)], toFixed(st.d0));
    atomicAdd(&acc[gi(weightField, st.nr1)], toFixed(st.d1));
    atomicAdd(&acc[gi(weightField, st.nr2)], toFixed(st.d2));
    atomicAdd(&acc[gi(weightField, st.nr3)], toFixed(st.d3));
}

@compute @workgroup_size(${WG})
fn p2g(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.numParticles) { return; }
    let p = pos[i];
    let vl = vel[i];
    let n = params.fNumY;
    let h = params.h;
    let h1 = params.fInvSpacing;

    // Mark the particle's cell as occupied (becomes FLUID unless solid)
    let cxi = u32(clamp(i32(floor(p.x * h1)), 0, i32(params.fNumX) - 1));
    let cyi = u32(clamp(i32(floor(p.y * h1)), 0, i32(params.fNumY) - 1));
    atomicAdd(&acc[gi(${ACC_COUNT}u, cxi * n + cyi)], 1);

    scatter(${ACC_U}u, ${ACC_DU}u, velocityStencil(p, 0u), vl.x);
    scatter(${ACC_V}u, ${ACC_DV}u, velocityStencil(p, 1u), vl.y);

    // Density scatter (FlipFluid.updateParticleDensity)
    let h2 = 0.5 * h;
    let x = clamp(p.x, h, f32(params.fNumX - 1u) * h);
    let y = clamp(p.y, h, f32(params.fNumY - 1u) * h);
    let x0 = u32(floor((x - h2) * h1));
    let tx = ((x - h2) - f32(x0) * h) * h1;
    let x1 = min(x0 + 1u, params.fNumX - 1u);
    let y0 = u32(floor((y - h2) * h1));
    let ty = ((y - h2) - f32(y0) * h) * h1;
    let y1 = min(y0 + 1u, params.fNumY - 1u);
    let sx = 1.0 - tx;
    let sy = 1.0 - ty;
    let D = ${ACC_DENSITY}u;
    if (x0 < params.fNumX && y0 < params.fNumY) { atomicAdd(&acc[gi(D, x0 * n + y0)], toFixed(sx * sy)); }
    if (x1 < params.fNumX && y0 < params.fNumY) { atomicAdd(&acc[gi(D, x1 * n + y0)], toFixed(tx * sy)); }
    if (x1 < params.fNumX && y1 < params.fNumY) { atomicAdd(&acc[gi(D, x1 * n + y1)], toFixed(tx * ty)); }
    if (x0 < params.fNumX && y1 < params.fNumY) { atomicAdd(&acc[gi(D, x0 * n + y1)], toFixed(sx * ty)); }
}

@compute @workgroup_size(${WG})
fn gridNormalize(@builtin(global_invocation_id) gid: vec3u) {
    let cell = gid.x;
    let n = params.fNumY;
    if (cell >= params.numCells) { return; }
    let i = cell / n;
    let j = cell % n;

    let solid = grid[gi(${GRID_S}u, cell)] == 0.0;
    var t = AIR_CELL;
    if (solid) {
        t = SOLID_CELL;
    } else if (atomicLoad(&acc[gi(${ACC_COUNT}u, cell)]) > 0) {
        t = FLUID_CELL;
    }
    grid[gi(${GRID_TYPE}u, cell)] = t;

    let du = fromFixed(atomicLoad(&acc[gi(${ACC_DU}u, cell)]));
    let dv = fromFixed(atomicLoad(&acc[gi(${ACC_DV}u, cell)]));
    var uu = 0.0;
    var vv = 0.0;
    if (du > 0.0) { uu = fromFixed(atomicLoad(&acc[gi(${ACC_U}u, cell)])) / du; }
    if (dv > 0.0) { vv = fromFixed(atomicLoad(&acc[gi(${ACC_V}u, cell)])) / dv; }

    // No-penetration on faces touching a solid cell
    if (solid || (i > 0u && grid[gi(${GRID_S}u, cell - n)] == 0.0)) { uu = 0.0; }
    if (solid || (j > 0u && grid[gi(${GRID_S}u, cell - 1u)] == 0.0)) { vv = 0.0; }

    grid[gi(${GRID_U}u, cell)] = uu;
    grid[gi(${GRID_V}u, cell)] = vv;
    grid[gi(${GRID_PREV_U}u, cell)] = uu;
    grid[gi(${GRID_PREV_V}u, cell)] = vv;
    grid[gi(${GRID_DENSITY}u, cell)] = fromFixed(atomicLoad(&acc[gi(${ACC_DENSITY}u, cell)]));
}

override PARITY: u32 = 0u;

@compute @workgroup_size(${WG})
fn pressure(@builtin(global_invocation_id) gid: vec3u) {
    let cell = gid.x;
    let n = params.fNumY;
    if (cell >= params.numCells) { return; }
    let i = cell / n;
    let j = cell % n;
    if (i < 1u || j < 1u || i >= params.fNumX - 1u || j >= n - 1u) { return; }
    if (((i + j) & 1u) != PARITY) { return; }
    if (grid[gi(${GRID_TYPE}u, cell)] != FLUID_CELL) { return; }

    let left = cell - n;
    let right = cell + n;
    let bottom = cell - 1u;
    let top = cell + 1u;

    let sx0 = grid[gi(${GRID_S}u, left)];
    let sx1 = grid[gi(${GRID_S}u, right)];
    let sy0 = grid[gi(${GRID_S}u, bottom)];
    let sy1 = grid[gi(${GRID_S}u, top)];
    let sSum = sx0 + sx1 + sy0 + sy1;
    if (sSum == 0.0) { return; }

    let U = ${GRID_U}u;
    let V = ${GRID_V}u;
    var div = grid[gi(U, right)] - grid[gi(U, cell)] + grid[gi(V, top)] - grid[gi(V, cell)];
    if (params.restDensity > 0.0 && params.compensateDrift != 0u) {
        let compression = grid[gi(${GRID_DENSITY}u, cell)] - params.restDensity;
        if (compression > 0.0) { div -= compression; }
    }

    let p = -div / sSum * params.overRelaxation;
    grid[gi(U, cell)] -= sx0 * p;
    grid[gi(U, right)] += sx1 * p;
    grid[gi(V, cell)] -= sy0 * p;
    grid[gi(V, top)] += sy1 * p;
}

fn faceValid(a: f32, b: f32) -> f32 {
    return select(0.0, 1.0, (a != AIR_CELL || b != AIR_CELL) && !(a == SOLID_CELL && b == SOLID_CELL));
}

fn cellTypeAt(cell: u32) -> f32 { return grid[gi(${GRID_TYPE}u, cell)]; }

fn g2pComponent(p: vec2f, component: u32, current: f32) -> f32 {
    let st = velocityStencil(p, component);
    let offset = select(params.fNumY, 1u, component == 1u);
    let F = select(${GRID_U}u, ${GRID_V}u, component == 1u);
    let PF = select(${GRID_PREV_U}u, ${GRID_PREV_V}u, component == 1u);

    let w0 = faceValid(cellTypeAt(st.nr0), cellTypeAt(st.nr0 - offset)) * st.d0;
    let w1 = faceValid(cellTypeAt(st.nr1), cellTypeAt(st.nr1 - offset)) * st.d1;
    let w2 = faceValid(cellTypeAt(st.nr2), cellTypeAt(st.nr2 - offset)) * st.d2;
    let w3 = faceValid(cellTypeAt(st.nr3), cellTypeAt(st.nr3 - offset)) * st.d3;
    let total = w0 + w1 + w2 + w3;
    if (total <= 0.0) { return current; }

    let f0 = grid[gi(F, st.nr0)];
    let f1 = grid[gi(F, st.nr1)];
    let f2 = grid[gi(F, st.nr2)];
    let f3 = grid[gi(F, st.nr3)];
    let p0 = grid[gi(PF, st.nr0)];
    let p1 = grid[gi(PF, st.nr1)];
    let p2 = grid[gi(PF, st.nr2)];
    let p3 = grid[gi(PF, st.nr3)];

    let picV = (w0 * f0 + w1 * f1 + w2 * f2 + w3 * f3) / total;
    let corr = (w0 * (f0 - p0) + w1 * (f1 - p1) + w2 * (f2 - p2) + w3 * (f3 - p3)) / total;
    return (1.0 - params.flipRatio) * picV + params.flipRatio * (current + corr);
}

@compute @workgroup_size(${WG})
fn g2p(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.numParticles) { return; }
    let p = pos[i];
    var vl = vel[i];
    vl.x = g2pComponent(p, 0u, vl.x);
    vl.y = g2pComponent(p, 1u, vl.y);
    vel[i] = vl;
}

@compute @workgroup_size(${WG})
fn colors(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.numParticles) { return; }
    let p = pos[i];
    let h1 = params.fInvSpacing;
    let xi = u32(clamp(i32(floor(p.x * h1)), 1, i32(params.fNumX) - 1));
    let yi = u32(clamp(i32(floor(p.y * h1)), 1, i32(params.fNumY) - 1));
    let cell = xi * params.fNumY + yi;

    let base = vec3f(params.baseR, params.baseG, params.baseB);
    let foam = vec3f(params.foamR, params.foamG, params.foamB);
    var c = readColor(i);

    let d0 = params.restDensity;
    if (d0 > 0.0 && grid[gi(${GRID_DENSITY}u, cell)] / d0 < 0.7) {
        c = foam;
    } else {
        let t = clamp(params.foamReturnRate * params.dt, 0.0, 1.0);
        c = c + (base - c) * t;
    }
    writeColor(i, c);
}

fn sciColor(val0: f32, minVal: f32, maxVal: f32) -> vec3f {
    let val = clamp(val0, minVal, maxVal - 0.0001);
    let d = maxVal - minVal;
    let t = select((val - minVal) / d, 0.5, d == 0.0);
    let m = 0.25;
    let num = floor(t / m);
    let sv = (t - num * m) / m;
    switch (u32(num)) {
        case 0u: { return vec3f(0.0, sv, 1.0); }
        case 1u: { return vec3f(0.0, 1.0, 1.0 - sv); }
        case 2u: { return vec3f(sv, 1.0, 0.0); }
        case 3u: { return vec3f(1.0, 1.0 - sv, 0.0); }
        default: { return vec3f(1.0, 0.0, 0.0); }
    }
}

@compute @workgroup_size(${WG})
fn cellColors(@builtin(global_invocation_id) gid: vec3u) {
    let cell = gid.x;
    if (cell >= params.numCells) { return; }
    var c = vec3f(0.0);
    let t = cellTypeAt(cell);
    if (t == SOLID_CELL) {
        c = vec3f(0.5);
    } else if (t == FLUID_CELL) {
        var d = grid[gi(${GRID_DENSITY}u, cell)];
        if (params.restDensity > 0.0) { d /= params.restDensity; }
        c = sciColor(d, 0.0, 2.0);
    }
    let base = ${GRID_COLOR}u * params.numCells + 3u * cell;
    grid[base] = c.x;
    grid[base + 1u] = c.y;
    grid[base + 2u] = c.z;
}
`;

type PipelineName =
    | 'integrate'
    | 'buildBuckets'
    | 'push'
    | 'pushCommit'
    | 'walls'
    | 'p2g'
    | 'gridNormalize'
    | 'pressureRed'
    | 'pressureBlack'
    | 'g2p'
    | 'colors'
    | 'cellColors';

export class WebGPUSimulation {
    private device: GPUDevice;

    // Particle buffers (pos/color double as vertex buffers for the renderer)
    particlePosBuffer!: GPUBuffer;
    particleVelBuffer!: GPUBuffer;
    particleColorBuffer!: GPUBuffer;
    private tmpBuffer!: GPUBuffer;
    private accBuffer!: GPUBuffer;
    /** Packed grid fields; cell colors (for the renderer) start at cellColorOffset. */
    gridBuffer!: GPUBuffer;
    cellColorOffset = 0;
    private bucketBuffer!: GPUBuffer;
    private bucketCountBytes = 0;

    private paramsBuffer!: GPUBuffer;
    private paramsData = new ArrayBuffer(28 * 4);
    private paramsU32 = new Uint32Array(this.paramsData);
    private paramsF32 = new Float32Array(this.paramsData);

    private bindGroup!: GPUBindGroup;
    private pipelines!: Record<PipelineName, GPUComputePipeline>;

    // Readback of particle state into the FlipFluid arrays
    private readbackBuffer!: GPUBuffer;
    private readbackPending = false;
    private generation = 0;
    private destroyed = false;

    fNumX = 0;
    fNumY = 0;
    fNumCells = 0;
    numParticles = 0;
    maxParticles = 0;
    h = 1.0;

    constructor(device: GPUDevice) {
        this.device = device;
    }

    initialize(fluid: FlipFluid): void {
        this.fNumX = fluid.fNumX;
        this.fNumY = fluid.fNumY;
        this.fNumCells = fluid.fNumCells;
        this.numParticles = fluid.numParticles;
        this.maxParticles = fluid.maxParticles;
        this.h = fluid.h;

        // Rest density is measured once, exactly like the CPU path does on its first
        // step: classify cells (particle → grid pass) and then average the density
        // over fluid cells. This only touches CPU-side grid scratch data.
        if (fluid.particleRestDensity === 0.0) {
            fluid.transferVelocities(true, 0.0);
            fluid.updateParticleDensity();
        }

        const P = fluid.maxParticles;
        const C = fluid.fNumCells;
        const B = fluid.pNumCells;
        const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
        const vertexStorage = storage | GPUBufferUsage.VERTEX;

        this.particlePosBuffer = this.createBuffer(P * 8, vertexStorage);
        this.particleVelBuffer = this.createBuffer(P * 8, storage);
        this.particleColorBuffer = this.createBuffer(P * 12, vertexStorage);
        this.tmpBuffer = this.createBuffer(P * 5 * 4, storage);
        this.accBuffer = this.createBuffer(C * ACC_FIELDS * 4, storage);
        this.gridBuffer = this.createBuffer(C * GRID_FIELDS * 4, vertexStorage);
        this.cellColorOffset = C * GRID_COLOR * 4;
        this.bucketCountBytes = B * 4;
        this.bucketBuffer = this.createBuffer(B * (1 + BUCKET_CAPACITY) * 4, storage);

        this.paramsBuffer = this.createBuffer(
            this.paramsData.byteLength,
            GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        );

        // pos (2) + vel (2) + color (3) floats per particle
        this.readbackBuffer = this.createBuffer(P * 7 * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);

        const ordered = [
            this.paramsBuffer,
            this.particlePosBuffer,
            this.particleVelBuffer,
            this.particleColorBuffer,
            this.tmpBuffer,
            this.accBuffer,
            this.gridBuffer,
            this.bucketBuffer
        ];
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: ordered.map((_, binding) => ({
                binding,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: binding === 0 ? 'uniform' : 'storage' }
            }))
        });
        this.bindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: ordered.map((buffer, binding) => ({ binding, resource: { buffer } }))
        });

        const layout = this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
        const module = this.device.createShaderModule({ code: shaderWGSL });
        const make = (entryPoint: string, constants?: Record<string, number>) =>
            this.device.createComputePipeline({ layout, compute: { module, entryPoint, constants } });

        this.pipelines = {
            integrate: make('integrate'),
            buildBuckets: make('buildBuckets'),
            push: make('push'),
            pushCommit: make('pushCommit'),
            walls: make('walls'),
            p2g: make('p2g'),
            gridNormalize: make('gridNormalize'),
            pressureRed: make('pressure', { PARITY: 0 }),
            pressureBlack: make('pressure', { PARITY: 1 }),
            g2p: make('g2p'),
            colors: make('colors'),
            cellColors: make('cellColors')
        };

        // Initial upload: solid flags, then particles
        this.device.queue.writeBuffer(this.gridBuffer, C * GRID_S * 4, fluid.s);
        this.syncParticles(fluid, 0);
    }

    private createBuffer(size: number, usage: number): GPUBuffer {
        // Storage bindings need a 4-byte aligned, non-zero size
        return this.device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage });
    }

    /**
     * Push CPU-side particle edits to the GPU. Particles [fromIndex, numParticles)
     * are uploaded; pass 0 after a removal (which compacts the arrays).
     */
    syncParticles(fluid: FlipFluid, fromIndex: number): void {
        const n = fluid.numParticles;
        const start = Math.max(0, Math.min(fromIndex, n));
        if (n > start) {
            const q = this.device.queue;
            q.writeBuffer(this.particlePosBuffer, start * 8, fluid.particlePos, start * 2, (n - start) * 2);
            q.writeBuffer(this.particleVelBuffer, start * 8, fluid.particleVel, start * 2, (n - start) * 2);
            q.writeBuffer(this.particleColorBuffer, start * 12, fluid.particleColor, start * 3, (n - start) * 3);
        }
        this.numParticles = n;
        // Any readback already in flight predates this edit; drop it.
        this.generation++;
    }

    /** Upload particle colors only (e.g. after FlipFluid.setFluidColor reset them). */
    syncColors(fluid: FlipFluid): void {
        if (this.numParticles === 0) return;
        this.device.queue.writeBuffer(this.particleColorBuffer, 0, fluid.particleColor, 0, this.numParticles * 3);
    }

    private writeParams(
        fluid: FlipFluid,
        dt: number,
        gravityX: number,
        gravityY: number,
        damping: number,
        flipRatio: number,
        overRelaxation: number,
        compensateDrift: boolean
    ): void {
        const u = this.paramsU32;
        const f = this.paramsF32;
        u[0] = this.numParticles;
        u[1] = fluid.fNumX;
        u[2] = fluid.fNumY;
        u[3] = fluid.pNumX;
        u[4] = fluid.pNumY;
        u[5] = compensateDrift ? 1 : 0;
        u[6] = this.maxParticles;
        u[7] = this.fNumCells;
        f[8] = fluid.h;
        f[9] = fluid.fInvSpacing;
        f[10] = fluid.pInvSpacing;
        f[11] = dt;
        f[12] = gravityX;
        f[13] = gravityY;
        f[14] = damping;
        f[15] = flipRatio;
        f[16] = overRelaxation;
        f[17] = fluid.particleRadius;
        f[18] = fluid.particleRestDensity;
        f[19] = fluid.colorDiffusionCoeff;
        f[20] = fluid.foamReturnRate;
        f[21] = fluid.baseColor.r;
        f[22] = fluid.baseColor.g;
        f[23] = fluid.baseColor.b;
        f[24] = fluid.foamColor.r;
        f[25] = fluid.foamColor.g;
        f[26] = fluid.foamColor.b;
        this.device.queue.writeBuffer(this.paramsBuffer, 0, this.paramsData);
    }

    simulate(
        fluid: FlipFluid,
        dt: number,
        gravityX: number,
        gravityY: number,
        flipRatio: number,
        numPressureIters: number,
        numParticleIters: number,
        overRelaxation: number,
        compensateDrift: boolean,
        separateParticles: boolean,
        damping: number,
        computeCellColors = false
    ): void {
        if (this.destroyed) return;
        this.writeParams(fluid, dt, gravityX, gravityY, damping, flipRatio, overRelaxation, compensateDrift);

        const particleGroups = Math.ceil(this.numParticles / WG);
        const cellGroups = Math.ceil(this.fNumCells / WG);
        const encoder = this.device.createCommandEncoder();

        const run = (pass: GPUComputePassEncoder, name: PipelineName, groups: number) => {
            if (groups === 0) return;
            pass.setPipeline(this.pipelines[name]);
            pass.dispatchWorkgroups(groups);
        };

        // Integrate, separate particles (spatial hash built once per step), walls
        encoder.clearBuffer(this.bucketBuffer, 0, this.bucketCountBytes);
        {
            const pass = encoder.beginComputePass();
            pass.setBindGroup(0, this.bindGroup);
            run(pass, 'integrate', particleGroups);
            if (separateParticles) {
                run(pass, 'buildBuckets', particleGroups);
                for (let k = 0; k < numParticleIters; k++) {
                    run(pass, 'push', particleGroups);
                    run(pass, 'pushCommit', particleGroups);
                }
            }
            run(pass, 'walls', particleGroups);
            pass.end();
        }

        // Particle → grid, pressure solve, grid → particle
        encoder.clearBuffer(this.accBuffer);
        {
            const pass = encoder.beginComputePass();
            pass.setBindGroup(0, this.bindGroup);
            run(pass, 'p2g', particleGroups);
            run(pass, 'gridNormalize', cellGroups);
            for (let k = 0; k < numPressureIters; k++) {
                run(pass, 'pressureRed', cellGroups);
                run(pass, 'pressureBlack', cellGroups);
            }
            run(pass, 'g2p', particleGroups);
            run(pass, 'colors', particleGroups);
            if (computeCellColors) run(pass, 'cellColors', cellGroups);
            pass.end();
        }

        // Kick off an async readback if none is in flight
        const n = this.numParticles;
        const startReadback = !this.readbackPending && n > 0;
        if (startReadback) {
            encoder.copyBufferToBuffer(this.particlePosBuffer, 0, this.readbackBuffer, 0, n * 8);
            encoder.copyBufferToBuffer(this.particleVelBuffer, 0, this.readbackBuffer, n * 8, n * 8);
            encoder.copyBufferToBuffer(this.particleColorBuffer, 0, this.readbackBuffer, n * 16, n * 12);
        }

        this.device.queue.submit([encoder.finish()]);

        if (startReadback) this.readBack(fluid, n);
    }

    private async readBack(fluid: FlipFluid, n: number): Promise<void> {
        this.readbackPending = true;
        const gen = this.generation;
        const size = n * 28;
        try {
            await this.readbackBuffer.mapAsync(GPUMapMode.READ, 0, size);
        } catch {
            this.readbackPending = false;
            return;
        }
        if (this.destroyed) return;
        if (gen === this.generation && fluid.numParticles === n) {
            const data = new Float32Array(this.readbackBuffer.getMappedRange(0, size));
            fluid.particlePos.set(data.subarray(0, n * 2));
            fluid.particleVel.set(data.subarray(n * 2, n * 4));
            fluid.particleColor.set(data.subarray(n * 4, n * 7));
        }
        this.readbackBuffer.unmap();
        this.readbackPending = false;
    }

    destroy(): void {
        this.destroyed = true;
        [
            this.particlePosBuffer,
            this.particleVelBuffer,
            this.particleColorBuffer,
            this.tmpBuffer,
            this.accBuffer,
            this.gridBuffer,
            this.bucketBuffer,
            this.paramsBuffer,
            this.readbackBuffer
        ].forEach((buf) => buf?.destroy());
    }
}
