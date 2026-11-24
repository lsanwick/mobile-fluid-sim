import type { FlipFluid } from './FlipFluid';
import { GPUBufferUsage, GPUShaderStage, GPUMapMode } from '../gpu-types';

// ============================================================
// WebGPU Compute Simulation — GPU-accelerated FLIP pipeline
// ============================================================
// All 8 kernels run on GPU:
//   K1: Integrate particles
//   K2: Push particles apart (brute-force per-particle)
//   K3: Wall collisions
//   K4: Transfer velocity particle → grid (brute-force per-cell)
//   K5: Update particle density (brute-force per-cell)
//   K6: Pressure solver (Jacobi — ~57% of frame time)
//   K7: Transfer velocity grid → particle
//   K8: Update particle colors
// ============================================================

// ---- WGSL shader source ----

const integrateShader = `
struct Params {
    dt: f32,
    gravityX: f32,
    gravityY: f32,
    damping: f32,
    numParticles: u32,
}

@group(0) @binding(0) var<storage, read_write> particlePos: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> particleVel: array<vec2f>;
@group(0) @binding(13) var<uniform> params: Params;

@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.numParticles) { return; }

    var vel = particleVel[i];
    vel.x += params.dt * params.gravityX;
    vel.y += params.dt * params.gravityY;
    vel = vel * params.damping;

    var pos = particlePos[i];
    pos += vel * params.dt;

    particleVel[i] = vel;
    particlePos[i] = pos;
}
`;

const wallCollisionsShader = `
struct Params {
    numParticles: u32,
    fNumX: u32,
    fNumY: u32,
    h: f32,
    particleRadius: f32,
}

@group(0) @binding(0) var<storage, read_write> particlePos: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> particleVel: array<vec2f>;
@group(0) @binding(13) var<uniform> params: Params;

@compute @workgroup_size(64)
fn wallCollisions(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.numParticles) { return; }

    var pos = particlePos[i];
    var vel = particleVel[i];

    let minX = params.h + params.particleRadius;
    let maxX = f32(params.fNumX - 1u) * params.h - params.particleRadius;
    let minY = params.h + params.particleRadius;
    let maxY = f32(params.fNumY - 1u) * params.h - params.particleRadius;

    if (pos.x < minX) { pos.x = minX; vel.x = 0.0; }
    if (pos.x > maxX) { pos.x = maxX; vel.x = 0.0; }
    if (pos.y < minY) { pos.y = minY; vel.y = 0.0; }
    if (pos.y > maxY) { pos.y = maxY; vel.y = 0.0; }

    particlePos[i] = pos;
    particleVel[i] = vel;
}
`;

const transferFromGridShader = `
struct Params {
    numParticles: u32,
    fNumX: u32,
    fNumY: u32,
    h: f32,
    fInvSpacing: f32,
    flipRatio: f32,
}

@group(0) @binding(0) var<storage, read_write> particlePos: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> particleVel: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> gridU: array<f32>;
@group(0) @binding(4) var<storage, read_write> gridV: array<f32>;
@group(0) @binding(5) var<storage, read_write> prevU: array<f32>;
@group(0) @binding(6) var<storage, read_write> prevV: array<f32>;
@group(0) @binding(10) var<storage, read_write> cellType: array<f32>;
@group(0) @binding(13) var<uniform> params: Params;

fn isValid(a: f32, b: f32) -> f32 {
    return select(0.0, 1.0, (a > 0.5 || b > 0.5) && !(a > 1.5 && b > 1.5));
}

@compute @workgroup_size(256)
fn transferFromGrid(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.numParticles) { return; }

    let pos = particlePos[i];
    let h = params.h;
    let h1 = params.fInvSpacing;

    for (let component = 0u; component < 2u; component++) {
        let dx = select(0.0, 0.5 * h, component == 1u);
        let dy = select(0.5 * h, 0.0, component == 1u);
        let f = select(gridU, gridV, component == 1u);
        let prevF = select(prevU, prevV, component == 1u);

        let x0 = u32(floor((pos.x - dx) * h1));
        let tx = ((pos.x - dx) - f32(x0) * h) * h1;
        let x1 = min(x0 + 1u, params.fNumX - 2u);
        let y0 = u32(floor((pos.y - dy) * h1));
        let ty = ((pos.y - dy) - f32(y0) * h) * h1;
        let y1 = min(y0 + 1u, params.fNumY - 2u);

        let sx = 1.0 - tx;
        let sy = 1.0 - ty;
        let d0 = sx * sy;
        let d1 = tx * sy;
        let d2 = tx * ty;
        let d3 = sx * ty;

        let nr0 = x0 * params.fNumY + y0;
        let nr1 = x1 * params.fNumY + y0;
        let nr2 = x1 * params.fNumY + y1;
        let nr3 = x0 * params.fNumY + y1;

        let offset = select(params.fNumY, 1u, component == 1u);
        let n0 = nr0 - offset;
        let n1 = nr1 - offset;
        let n2 = nr2 - offset;
        let n3 = nr3 - offset;

        let valid0 = isValid(cellType[nr0], cellType[n0]);
        let valid1 = isValid(cellType[nr1], cellType[n1]);
        let valid2 = isValid(cellType[nr2], cellType[n2]);
        let valid3 = isValid(cellType[nr3], cellType[n3]);

        let totalValid = valid0 * d0 + valid1 * d1 + valid2 * d2 + valid3 * d3;

        if (totalValid > 0.0) {
            let picV = (valid0 * d0 * f[nr0] + valid1 * d1 * f[nr1] + valid2 * d2 * f[nr2] + valid3 * d3 * f[nr3]) / totalValid;
            let corr = (valid0 * d0 * (f[nr0] - prevF[nr0]) + valid1 * d1 * (f[nr1] - prevF[nr1]) + valid2 * d2 * (f[nr2] - prevF[nr2]) + valid3 * d3 * (f[nr3] - prevF[nr3])) / totalValid;

            let v = select(particleVel[i].x, particleVel[i].y, component == 1u);
            let newV = (1.0 - params.flipRatio) * picV + params.flipRatio * (v + corr);

            if (component == 0u) {
                particleVel[i].x = newV;
            } else {
                particleVel[i].y = newV;
            }
        }
    }
}
`;

const pressureSolverShader = `
struct Params {
    fNumX: u32,
    fNumY: u32,
    h: f32,
    density: f32,
    dt: f32,
    overRelaxation: f32,
    compensateDrift: f32,
    particleRestDensity: f32,
}

@group(0) @binding(3) var<storage, read_write> gridU: array<f32>;
@group(0) @binding(4) var<storage, read_write> gridV: array<f32>;
@group(0) @binding(9) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(10) var<storage, read_write> cellType: array<f32>;
@group(0) @binding(11) var<storage, read_write> density: array<f32>;
@group(0) @binding(13) var<uniform> params: Params;

const FLUID_CELL: f32 = 1.0;

@compute @workgroup_size(16, 16)
fn pressureSolver(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let j = gid.y;

    if (i >= params.fNumX - 1u || j >= params.fNumY - 1u) { return; }

    let center = i * params.fNumY + j;
    if (cellType[center] < 0.5) { return; }

    let left = (i - 1u) * params.fNumY + j;
    let right = (i + 1u) * params.fNumY + j;
    let bottom = i * params.fNumY + (j - 1u);
    let top = i * params.fNumY + (j + 1u);

    let s = cellType[left] + cellType[right] + cellType[bottom] + cellType[top];
    if (s < 0.5) { return; }

    let div = gridU[right] - gridU[center] + gridV[top] - gridV[center];

    var divFinal = div;
    if (params.compensateDrift > 0.5 && params.particleRestDensity > 0.0) {
        let compression = density[center] - params.particleRestDensity;
        if (compression > 0.0) {
            divFinal = div - compression;
        }
    }

    let p = -divFinal / s * params.overRelaxation;
    pressure[center] += params.density * params.h / params.dt * p;

    gridU[center] -= cellType[left] * p;
    gridU[right] += cellType[right] * p;
    gridV[center] -= cellType[bottom] * p;
    gridV[top] += cellType[top] * p;
}
`;

const updateColorsShader = `
struct Params {
    numParticles: u32,
}

@group(0) @binding(0) var<storage, read_write> particlePos: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> particleColor: array<vec3f>;
@group(0) @binding(11) var<storage, read_write> density: array<f32>;
@group(0) @binding(14) var<storage, read-only> baseColor: vec3f;
@group(0) @binding(15) var<storage, read-only> foamColor: vec3f;
@group(0) @binding(16) var<storage, read-only> foamReturnRate: f32;
@group(0) @binding(17) var<storage, read-only> particleRestDensity: f32;
@group(0) @binding(18) var<storage, read-only> fNumX: u32;
@group(0) @binding(19) var<storage, read-only> fNumY: u32;
@group(0) @binding(20) var<storage, read-only> fInvSpacing: f32;
@group(0) @binding(13) var<uniform> params: Params;

@compute @workgroup_size(256)
fn updateColors(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.numParticles) { return; }

    let pos = particlePos[i];
    let h1 = fInvSpacing;

    let xi = min(max(u32(floor(pos.x * h1)), 1u), params.fNumX - 1u);
    let yi = min(max(u32(floor(pos.y * h1)), 1u), params.fNumY - 1u);
    let cellIdx = xi * params.fNumY + yi;

    let applyFoam = particleRestDensity > 0.0 && density[cellIdx] / particleRestDensity < 0.7;

    if (applyFoam) {
        particleColor[i] = foamColor;
    } else {
        let c = particleColor[i];
        particleColor[i] = c + (baseColor - c) * foamReturnRate;
    }
}
`;

// ---- K2: Push particles apart (brute-force per-particle) ----

const pushParticlesShader = `
struct Params {
    numParticles: u32,
    particleRadius: f32,
    colorDiffusionCoeff: f32,
}

@group(0) @binding(0) var<storage, read_write> particlePos: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> particleColor: array<vec3f>;
@group(0) @binding(13) var<uniform> params: Params;

@compute @workgroup_size(64)
fn pushParticles(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.numParticles) { return; }

    let px = particlePos[2 * i];
    let py = particlePos[2 * i + 1];

    let minDist = 2.0 * params.particleRadius;
    let minDist2 = minDist * minDist;

    for (let j = 0u; j < params.numParticles; j++) {
        if (j == i) { continue; }

        let qx = particlePos[2 * j];
        let qy = particlePos[2 * j + 1];
        let dx = qx - px;
        let dy = qy - py;
        let d2 = dx * dx + dy * dy;

        if (d2 > minDist2 || d2 < 1e-10) { continue; }

        let d = sqrt(d2);
        let s = 0.5 * (minDist - d) / d;
        let deltaX = dx * s;
        let deltaY = dy * s;

        particlePos[2 * i] -= deltaX;
        particlePos[2 * i + 1] -= deltaY;

        // Color diffusion (asymmetric: only update owned particle)
        for (let k = 0u; k < 3u; k++) {
            let color0 = particleColor[3 * i + k];
            let color1 = particleColor[3 * j + k];
            let color = (color0 + color1) * 0.5;
            particleColor[3 * i + k] = color0 + (color - color0) * params.colorDiffusionCoeff;
        }
    }
}
`;

// ---- K4: Transfer velocity particle → grid (brute-force per-cell) ----

const transferFromParticleShader = `
struct Params {
    numParticles: u32,
    fNumX: u32,
    fNumY: u32,
    h: f32,
    fInvSpacing: f32,
}

@group(0) @binding(0) var<storage, read> particlePos: array<vec2f>;
@group(0) @binding(1) var<storage, read> particleVel: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> u: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;
@group(0) @binding(5) var<storage, read_write> prevU: array<f32>;
@group(0) @binding(6) var<storage, read_write> prevV: array<f32>;
@group(0) @binding(7) var<storage, read_write> du: array<f32>;
@group(0) @binding(8) var<storage, read_write> dv: array<f32>;
@group(0) @binding(10) var<storage, read> cellType: array<f32>;
@group(0) @binding(13) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn transferFromParticle(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let j = gid.y;

    if (i >= params.fNumX || j >= params.fNumY) { return; }

    let cellIdx = i * params.fNumY + j;

    // Copy u/v to prevU/prevV
    prevU[cellIdx] = u[cellIdx];
    prevV[cellIdx] = v[cellIdx];

    // Zero u/v/du/dv
    u[cellIdx] = 0.0;
    v[cellIdx] = 0.0;
    du[cellIdx] = 0.0;
    dv[cellIdx] = 0.0;

    // Accumulate particle velocities
    for (let p = 0u; p < params.numParticles; p++) {
        for (let component = 0u; component < 2u; component++) {
            let dx = select(0.0, 0.5 * params.h, component == 1u);
            let dy = select(0.5 * params.h, 0.0, component == 1u);

            let x0 = u32(floor((particlePos[2 * p].x - dx) * params.fInvSpacing));
            let tx = ((particlePos[2 * p].x - dx) - f32(x0) * params.h) * params.fInvSpacing;
            let x1 = min(x0 + 1u, params.fNumX - 1u);
            let y0 = u32(floor((particlePos[2 * p].y - dy) * params.fInvSpacing));
            let ty = ((particlePos[2 * p].y - dy) - f32(y0) * params.h) * params.fInvSpacing;
            let y1 = min(y0 + 1u, params.fNumY - 1u);

            let sx = 1.0 - tx;
            let sy = 1.0 - ty;

            // Weight for this cell from this particle
            let weight = 0.0;
            if (x0 == i && y0 == j) { weight = sx * sy; }
            else if (x1 == i && y0 == j) { weight = tx * sy; }
            else if (x1 == i && y1 == j) { weight = tx * ty; }
            else if (x0 == i && y1 == j) { weight = sx * ty; }

            if (weight > 0.0) {
                let pv = select(particleVel[2 * p].x, particleVel[2 * p].y, component == 1u);
                if (component == 0u) {
                    u[cellIdx] += pv * weight;
                    du[cellIdx] += weight;
                } else {
                    v[cellIdx] += pv * weight;
                    dv[cellIdx] += weight;
                }
            }
        }
    }

    // Normalize
    if (du[cellIdx] > 0.0) u[cellIdx] /= du[cellIdx];
    if (dv[cellIdx] > 0.0) v[cellIdx] /= dv[cellIdx];

    // No-penetration boundary conditions
    let solid = cellType[cellIdx] > 1.5;
    if (solid || (i > 0 && cellType[(i - 1u) * params.fNumY + j] > 1.5)) {
        u[cellIdx] = 0.0;
    }
    if (solid || (j > 0 && cellType[i * params.fNumY + (j - 1u)] > 1.5)) {
        v[cellIdx] = 0.0;
    }
}
`;

// ---- K5: Update particle density (brute-force per-cell) ----

const updateDensityShader = `
struct Params {
    numParticles: u32,
    fNumX: u32,
    fNumY: u32,
    h: f32,
    fInvSpacing: f32,
}

@group(0) @binding(0) var<storage, read> particlePos: array<vec2f>;
@group(0) @binding(11) var<storage, read_write> density: array<f32>;
@group(0) @binding(13) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn updateDensity(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let j = gid.y;

    if (i >= params.fNumX || j >= params.fNumY) { return; }

    let cellIdx = i * params.fNumY + j;
    density[cellIdx] = 0.0;

    let h = params.h;
    let h1 = params.fInvSpacing;
    let h2 = 0.5 * h;

    for (let p = 0u; p < params.numParticles; p++) {
        let px = particlePos[2 * p];
        let py = particlePos[2 * p + 1];

        let cx = max(h, min(f32(params.fNumX - 1u) * h, px));
        let cy = max(h, min(f32(params.fNumY - 1u) * h, py));

        let x0 = u32(floor((cx - h2) * h1));
        let tx = ((cx - h2) - f32(x0) * h) * h1;
        let x1 = min(x0 + 1u, params.fNumX - 1u);
        let y0 = u32(floor((cy - h2) * h1));
        let ty = ((cy - h2) - f32(y0) * h) * h1;
        let y1 = min(y0 + 1u, params.fNumY - 1u);

        let sx = 1.0 - tx;
        let sy = 1.0 - ty;

        // Weight for this cell from this particle
        let weight = 0.0;
        if (x0 == i && y0 == j) { weight = sx * sy; }
        else if (x1 == i && y0 == j) { weight = tx * sy; }
        else if (x1 == i && y1 == j) { weight = tx * ty; }
        else if (x0 == i && y1 == j) { weight = sx * ty; }

        density[cellIdx] += weight;
    }
}
`;

// ---- Rest density reduction (computed on GPU after K5, before K6) ----

const computeRestDensityShader = `
struct Params {
    fNumX: u32,
    fNumY: u32,
}

@group(0) @binding(11) var<storage, read> density: array<f32>;
@group(0) @binding(10) var<storage, read> cellType: array<f32>;
@group(0) @binding(21) var<storage, read_write> restDensityResult: array<f32>;
@group(0) @binding(13) var<uniform> params: Params;

@compute @workgroup_size(1)
fn computeRestDensity(@builtin(global_invocation_id) gid: vec3u) {
    let sum = 0.0;
    let count = 0u;

    for (let i = 0u; i < params.fNumX; i++) {
        for (let j = 0u; j < params.fNumY; j++) {
            let cellIdx = i * params.fNumY + j;
            if (cellType[cellIdx] == 1.0) {
                sum += density[cellIdx];
                count += 1u;
            }
        }
    }

    if (count > 0u) {
        restDensityResult[0] = sum / f32(count);
    }
}
`;

// ---- Parameter buffer layout ----
// Uniform buffer (64 bytes, aligned):
//   offset 0:  dt (f32)
//   offset 4:  gravityX (f32)
//   offset 8:  gravityY (f32)
//   offset 12: damping (f32)
//   offset 16: numParticles (u32)
//   offset 20: fNumX (u32)
//   offset 24: fNumY (u32)
//   offset 28: h (f32)
//   offset 32: fInvSpacing (f32)
//   offset 36: flipRatio (f32)
//   offset 40: density (f32)
//   offset 44: overRelaxation (f32)
//   offset 48: compensateDrift (f32)
//   offset 52: particleRestDensity (f32)
//   offset 56: numPressureIters (u32)
//   offset 60: numParticleIters (u32)
//   offset 64: separateParticles (u32)
//   padding to 128 bytes

interface GPUBufferDesc {
    buffer: GPUBuffer;
    usage: number;
    size: number;
}

export class WebGPUSimulation {
    private device: GPUDevice;

    // GPU buffers
    private particlePosBuffer!: GPUBuffer;
    private particleVelBuffer!: GPUBuffer;
    private particleColorBuffer!: GPUBuffer;
    private gridUBuffer!: GPUBuffer;
    private gridVBuffer!: GPUBuffer;
    private prevUBuffer!: GPUBuffer;
    private prevVBuffer!: GPUBuffer;
    private duBuffer!: GPUBuffer;
    private dvBuffer!: GPUBuffer;
    private pressureBuffer!: GPUBuffer;
    private cellTypeBuffer!: GPUBuffer;
    private densityBuffer!: GPUBuffer;
    private cellColorBuffer!: GPUBuffer;

    // Params uniform buffer
    private paramsBuffer!: GPUBuffer;

    // Bind group
    private bindGroup!: GPUBindGroup;

    // Pipelines
    private integratePipeline!: GPUComputePipeline;
    private pushParticlesPipeline!: GPUComputePipeline;
    private wallCollisionsPipeline!: GPUComputePipeline;
    private transferFromParticlePipeline!: GPUComputePipeline;
    private updateDensityPipeline!: GPUComputePipeline;
    private computeRestDensityPipeline!: GPUComputePipeline;
    private pressureSolverPipeline!: GPUComputePipeline;
    private transferFromGridPipeline!: GPUComputePipeline;
    private updateColorsPipeline!: GPUComputePipeline;

    // Rest density result buffer (GPU-computed, used by K6)
    private restDensityResultBuffer!: GPUBuffer;

    // Simulation dimensions
    private fNumX = 0;
    private fNumY = 0;
    private fNumCells = 0;
    private numParticles = 0;
    private h = 1.0;
    private fInvSpacing = 0;

    // CPU-side data for reading back
    private particlePosData!: Float32Array;
    private particleVelData!: Float32Array;
    private particleColorData!: Float32Array;
    private gridUData!: Float32Array;
    private gridVData!: Float32Array;
    private cellColorData!: Float32Array;

    // Readback buffers
    private posReadbackBuffer!: GPUBuffer;
    private velReadbackBuffer!: GPUBuffer;
    private colorReadbackBuffer!: GPUBuffer;
    private gridUReadbackBuffer!: GPUBuffer;
    private gridVReadbackBuffer!: GPUBuffer;
    private cellColorReadbackBuffer!: GPUBuffer;

    // Pending readbacks
    private pendingReadbacks: Map<GPUBuffer, { data: Uint8Array; resolve: (value: ArrayBuffer) => void }>;

    constructor(device: GPUDevice) {
        this.device = device;
        this.pendingReadbacks = new Map();
    }

    initialize(
        fluid: FlipFluid,
        gravityX: number,
        gravityY: number,
        damping: number
    ): void {
        this.fNumX = fluid.fNumX;
        this.fNumY = fluid.fNumY;
        this.fNumCells = fluid.fNumCells;
        this.numParticles = fluid.numParticles;
        this.h = fluid.h;
        this.fInvSpacing = fluid.fInvSpacing;

        const numParticles = fluid.maxParticles;
        const fNumX = fluid.fNumX;
        const fNumY = fluid.fNumY;

        // Create GPU buffers
        this.particlePosBuffer = this.createBuffer(
            numParticles * 2 * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX
        );
        this.particleVelBuffer = this.createBuffer(
            numParticles * 2 * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );
        this.particleColorBuffer = this.createBuffer(
            numParticles * 3 * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX
        );
        this.gridUBuffer = this.createBuffer(
            fNumX * fNumY * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );
        this.gridVBuffer = this.createBuffer(
            fNumX * fNumY * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );
        this.prevUBuffer = this.createBuffer(
            fNumX * fNumY * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );
        this.prevVBuffer = this.createBuffer(
            fNumX * fNumY * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );
        this.duBuffer = this.createBuffer(
            fNumX * fNumY * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );
        this.dvBuffer = this.createBuffer(
            fNumX * fNumY * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );
        this.pressureBuffer = this.createBuffer(
            fNumX * fNumY * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );
        this.cellTypeBuffer = this.createBuffer(
            fNumX * fNumY * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );
        this.densityBuffer = this.createBuffer(
            fNumX * fNumY * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );
        this.cellColorBuffer = this.createBuffer(
            fNumX * fNumY * 3 * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX
        );

        // Rest density result buffer (single f32)
        this.restDensityResultBuffer = this.createBuffer(
            16,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );

        // Params uniform buffer (256 bytes, aligned)
        this.paramsBuffer = this.createBuffer(
            256,
            GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        );

        // Readback buffers
        this.posReadbackBuffer = this.createBuffer(
            numParticles * 2 * 4,
            GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        );
        this.velReadbackBuffer = this.createBuffer(
            numParticles * 2 * 4,
            GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        );
        this.colorReadbackBuffer = this.createBuffer(
            numParticles * 3 * 4,
            GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        );
        this.gridUReadbackBuffer = this.createBuffer(
            fNumX * fNumY * 4,
            GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        );
        this.gridVReadbackBuffer = this.createBuffer(
            fNumX * fNumY * 4,
            GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        );
        this.cellColorReadbackBuffer = this.createBuffer(
            fNumX * fNumY * 3 * 4,
            GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        );

        // Create bind group layout
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // particlePos
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // particleVel
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // particleColor
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // gridU
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // gridV
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // prevU
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // prevV
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // du
                { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // dv
                { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // pressure
                { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // cellType
                { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // density
                { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // cellColor
                { binding: 13, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }, // params
                { binding: 14, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // baseColor
                { binding: 15, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // foamColor
                { binding: 16, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // foamReturnRate
                { binding: 17, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // particleRestDensity
                { binding: 18, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // fNumX
                { binding: 19, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // fNumY
                { binding: 20, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // fInvSpacing
                { binding: 21, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // restDensityResult
            ]
        });

        // Create read-only buffers for color params
        const baseColorData = new Float32Array([fluid.baseColor.r, fluid.baseColor.g, fluid.baseColor.b, 0]);
        const baseColorBuffer = this.createBuffer(
            16,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );
        this.device.queue.writeBuffer(baseColorBuffer, 0, baseColorData);

        const foamColorData = new Float32Array([fluid.foamColor.r, fluid.foamColor.g, fluid.foamColor.b, 0]);
        const foamColorBuffer = this.createBuffer(
            16,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );
        this.device.queue.writeBuffer(foamColorBuffer, 0, foamColorData);

        const foamReturnRateBuffer = this.createBuffer(
            16,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );
        this.device.queue.writeBuffer(foamReturnRateBuffer, 0, new Float32Array([fluid.foamReturnRate]));

        const restDensityBuffer = this.createBuffer(
            16,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );
        this.device.queue.writeBuffer(restDensityBuffer, 0, new Float32Array([fluid.particleRestDensity]));

        const fNumXBuffer = this.createBuffer(
            16,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );
        this.device.queue.writeBuffer(fNumXBuffer, 0, new Uint32Array([fNumX]));

        const fNumYBuffer = this.createBuffer(
            16,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );
        this.device.queue.writeBuffer(fNumYBuffer, 0, new Uint32Array([fNumY]));

        const fInvSpacingBuffer = this.createBuffer(
            16,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );
        this.device.queue.writeBuffer(fInvSpacingBuffer, 0, new Float32Array([fluid.fInvSpacing]));

        // Create bind group
        this.bindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.particlePosBuffer } },
                { binding: 1, resource: { buffer: this.particleVelBuffer } },
                { binding: 2, resource: { buffer: this.particleColorBuffer } },
                { binding: 3, resource: { buffer: this.gridUBuffer } },
                { binding: 4, resource: { buffer: this.gridVBuffer } },
                { binding: 5, resource: { buffer: this.prevUBuffer } },
                { binding: 6, resource: { buffer: this.prevVBuffer } },
                { binding: 7, resource: { buffer: this.duBuffer } },
                { binding: 8, resource: { buffer: this.dvBuffer } },
                { binding: 9, resource: { buffer: this.pressureBuffer } },
                { binding: 10, resource: { buffer: this.cellTypeBuffer } },
                { binding: 11, resource: { buffer: this.densityBuffer } },
                { binding: 12, resource: { buffer: this.cellColorBuffer } },
                { binding: 13, resource: { buffer: this.paramsBuffer } },
                { binding: 14, resource: { buffer: baseColorBuffer } },
                { binding: 15, resource: { buffer: foamColorBuffer } },
                { binding: 16, resource: { buffer: foamReturnRateBuffer } },
                { binding: 17, resource: { buffer: restDensityBuffer } },
                { binding: 18, resource: { buffer: fNumXBuffer } },
                { binding: 19, resource: { buffer: fNumYBuffer } },
                { binding: 20, resource: { buffer: fInvSpacingBuffer } },
                { binding: 21, resource: { buffer: this.restDensityResultBuffer } },
            ]
        });

        // Create compute pipelines
        this.integratePipeline = this.device.createComputePipeline({
            layout: this.createComputePipelineLayout([0, 1, 13]),
            compute: { module: this.device.createShaderModule({ code: integrateShader }), entryPoint: 'integrate' }
        });

        this.wallCollisionsPipeline = this.device.createComputePipeline({
            layout: this.createComputePipelineLayout([0, 1, 13]),
            compute: { module: this.device.createShaderModule({ code: wallCollisionsShader }), entryPoint: 'wallCollisions' }
        });

        this.pressureSolverPipeline = this.device.createComputePipeline({
            layout: this.createComputePipelineLayout([3, 4, 9, 10, 11, 13]),
            compute: { module: this.device.createShaderModule({ code: pressureSolverShader }), entryPoint: 'pressureSolver' }
        });

        this.transferFromGridPipeline = this.device.createComputePipeline({
            layout: this.createComputePipelineLayout([0, 1, 3, 4, 5, 6, 10, 13]),
            compute: { module: this.device.createShaderModule({ code: transferFromGridShader }), entryPoint: 'transferFromGrid' }
        });

        this.updateColorsPipeline = this.device.createComputePipeline({
            layout: this.createComputePipelineLayout([0, 2, 11, 13, 14, 15, 16, 17, 18, 19, 20]),
            compute: { module: this.device.createShaderModule({ code: updateColorsShader }), entryPoint: 'updateColors' }
        });

        this.pushParticlesPipeline = this.device.createComputePipeline({
            layout: this.createComputePipelineLayout([0, 2, 13]),
            compute: { module: this.device.createShaderModule({ code: pushParticlesShader }), entryPoint: 'pushParticles' }
        });

        this.transferFromParticlePipeline = this.device.createComputePipeline({
            layout: this.createComputePipelineLayout([0, 1, 3, 4, 5, 6, 7, 8, 10, 13]),
            compute: { module: this.device.createShaderModule({ code: transferFromParticleShader }), entryPoint: 'transferFromParticle' }
        });

        this.updateDensityPipeline = this.device.createComputePipeline({
            layout: this.createComputePipelineLayout([0, 11, 13]),
            compute: { module: this.device.createShaderModule({ code: updateDensityShader }), entryPoint: 'updateDensity' }
        });

        this.computeRestDensityPipeline = this.device.createComputePipeline({
            layout: this.createComputePipelineLayout([10, 11, 13, 21]),
            compute: { module: this.device.createShaderModule({ code: computeRestDensityShader }), entryPoint: 'computeRestDensity' }
        });

        // Initialize buffers with CPU data
        this.uploadToGPU(fluid);
    }

    private createBuffer(size: number, usage: number): GPUBuffer {
        return this.device.createBuffer({ size: Math.max(size, 1), usage });
    }

    private createComputePipelineLayout(bindings: number[]): GPUPipelineLayout {
        // Build a minimal bind group layout for the specific pipeline
        const entries: GPUBindGroupLayoutEntry[] = [];
        for (const binding of bindings) {
            if (binding >= 0 && binding <= 12) {
                entries.push({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } });
            } else if (binding === 13) {
                entries.push({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } });
            } else {
                entries.push({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } });
            }
        }
        return this.device.createPipelineLayout({ bindGroupLayouts: [this.device.createBindGroupLayout({ entries })] });
    }

    private uploadToGPU(fluid: FlipFluid): void {
        const numParticles = fluid.maxParticles;

        // Upload particle data
        this.device.queue.writeBuffer(this.particlePosBuffer, 0, fluid.particlePos, 0, 2 * numParticles * 4);
        this.device.queue.writeBuffer(this.particleVelBuffer, 0, fluid.particleVel, 0, 2 * numParticles * 4);
        this.device.queue.writeBuffer(this.particleColorBuffer, 0, fluid.particleColor, 0, 3 * numParticles * 4);

        // Upload grid data
        this.device.queue.writeBuffer(this.gridUBuffer, 0, fluid.u);
        this.device.queue.writeBuffer(this.gridVBuffer, 0, fluid.v);
        this.device.queue.writeBuffer(this.prevUBuffer, 0, fluid.prevU);
        this.device.queue.writeBuffer(this.prevVBuffer, 0, fluid.prevV);
        this.device.queue.writeBuffer(this.duBuffer, 0, fluid.du);
        this.device.queue.writeBuffer(this.dvBuffer, 0, fluid.dv);
        this.device.queue.writeBuffer(this.pressureBuffer, 0, fluid.p);
        this.device.queue.writeBuffer(this.cellTypeBuffer, 0, fluid.cellType);
        this.device.queue.writeBuffer(this.densityBuffer, 0, fluid.particleDensity);
        this.device.queue.writeBuffer(this.cellColorBuffer, 0, fluid.cellColor);
    }

    private writeParams(
        dt: number,
        gravityX: number,
        gravityY: number,
        damping: number,
        flipRatio: number,
        numPressureIters: number,
        numParticleIters: number,
        overRelaxation: number,
        compensateDrift: boolean,
        particleRestDensity: number,
        particleRadius: number,
        colorDiffusionCoeff: number,
        pNumX: number,
        pNumY: number,
        pInvSpacing: number
    ): void {
        const params = new Float32Array(64); // 64 × 4 = 256 bytes
        // K1 params
        params[0] = dt;
        params[1] = gravityX;
        params[2] = gravityY;
        params[3] = damping;
        // K3 params (overlap with K1, same struct)
        params[4] = this.numParticles;
        params[5] = this.fNumX;
        params[6] = this.fNumY;
        params[7] = this.h;
        // K6 params (pressure solver)
        params[8] = 1000.0; // density (default)
        params[9] = dt;
        params[10] = overRelaxation;
        params[11] = compensateDrift ? 1.0 : 0.0;
        params[12] = particleRestDensity;
        // K7 params (transfer from grid)
        params[13] = this.fInvSpacing;
        params[14] = flipRatio;
        // K8 params
        params[15] = this.numParticles;
        // Extra
        params[16] = numPressureIters;
        params[17] = numParticleIters;
        params[18] = 0.0; // separateParticles (bool as f32)
        // K2 params
        params[19] = particleRadius;
        params[20] = colorDiffusionCoeff;
        params[21] = pNumX;
        params[22] = pNumY;
        params[23] = pInvSpacing;

        this.device.queue.writeBuffer(this.paramsBuffer, 0, params);
    }

    // ---- Simulation step ----

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
        damping: number
    ): void {
        const encoder = this.device.createCommandEncoder();

        // Upload CPU data that changed (particle-to-grid transfer, density, cell type)
        this.uploadToGPU(fluid);

        // Write params
        this.writeParams(
            dt, gravityX, gravityY, damping,
            flipRatio, numPressureIters, numParticleIters,
            overRelaxation, compensateDrift, fluid.particleRestDensity,
            fluid.particleRadius, fluid.colorDiffusionCoeff,
            fluid.pNumX, fluid.pNumY, fluid.pInvSpacing
        );

        // K1: Integrate particles
        {
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.integratePipeline);
            pass.setBindGroup(0, this.bindGroup);
            pass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
            pass.end();
        }

        // K2: Push particles apart (GPU)
        if (separateParticles) {
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.pushParticlesPipeline);
            pass.setBindGroup(0, this.bindGroup);
            pass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
            pass.end();
        }

        // K3: Wall collisions
        {
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.wallCollisionsPipeline);
            pass.setBindGroup(0, this.bindGroup);
            pass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
            pass.end();
        }

        // K4: Transfer velocity particle → grid (GPU)
        {
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.transferFromParticlePipeline);
            pass.setBindGroup(0, this.bindGroup);
            pass.dispatchWorkgroups(
                Math.ceil(this.fNumX / 16),
                Math.ceil(this.fNumY / 16),
                1
            );
            pass.end();
        }

        // K5: Update particle density (GPU)
        {
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.updateDensityPipeline);
            pass.setBindGroup(0, this.bindGroup);
            pass.dispatchWorkgroups(
                Math.ceil(this.fNumX / 16),
                Math.ceil(this.fNumY / 16),
                1
            );
            pass.end();
        }

        // Compute rest density from density buffer (GPU reduction)
        {
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.computeRestDensityPipeline);
            pass.setBindGroup(0, this.bindGroup);
            pass.dispatchWorkgroups(1);
            pass.end();
        }

        // K6: Pressure solver (Jacobi iterations on GPU)
        for (let iter = 0; iter < numPressureIters; iter++) {
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.pressureSolverPipeline);
            pass.setBindGroup(0, this.bindGroup);
            pass.dispatchWorkgroups(
                Math.ceil((this.fNumX - 1) / 16),
                Math.ceil((this.fNumY - 1) / 16),
                1
            );
            pass.end();
        }

        // K7: Transfer velocity grid → particle
        {
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.transferFromGridPipeline);
            pass.setBindGroup(0, this.bindGroup);
            pass.dispatchWorkgroups(Math.ceil(this.numParticles / 256));
            pass.end();
        }

        // K8: Update particle colors
        {
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.updateColorsPipeline);
            pass.setBindGroup(0, this.bindGroup);
            pass.dispatchWorkgroups(Math.ceil(this.numParticles / 256));
            pass.end();
        }

        // Copy GPU data back to readback buffers
        this.copyToReadback(encoder);

        this.device.queue.submit([encoder.finish()]);
    }

    private copyToReadback(encoder: GPUCommandEncoder): void {
        encoder.copyBufferToBuffer(this.particlePosBuffer, 0, this.posReadbackBuffer, 0, this.numParticles * 2 * 4);
        encoder.copyBufferToBuffer(this.particleVelBuffer, 0, this.velReadbackBuffer, 0, this.numParticles * 2 * 4);
        encoder.copyBufferToBuffer(this.particleColorBuffer, 0, this.colorReadbackBuffer, 0, this.numParticles * 3 * 4);
        encoder.copyBufferToBuffer(this.gridUBuffer, 0, this.gridUReadbackBuffer, 0, this.fNumX * this.fNumY * 4);
        encoder.copyBufferToBuffer(this.gridVBuffer, 0, this.gridVReadbackBuffer, 0, this.fNumX * this.fNumY * 4);
        encoder.copyBufferToBuffer(this.cellColorBuffer, 0, this.cellColorReadbackBuffer, 0, this.fNumX * this.fNumY * 3 * 4);
    }

    // ---- Readback ----

    async readParticlePos(): Promise<Float32Array> {
        return this.mapBuffer(this.posReadbackBuffer, this.numParticles * 2 * 4);
    }

    async readParticleVel(): Promise<Float32Array> {
        return this.mapBuffer(this.velReadbackBuffer, this.numParticles * 2 * 4);
    }

    async readParticleColor(): Promise<Float32Array> {
        return this.mapBuffer(this.colorReadbackBuffer, this.numParticles * 3 * 4);
    }

    async readGridU(): Promise<Float32Array> {
        return this.mapBuffer(this.gridUReadbackBuffer, this.fNumX * this.fNumY * 4);
    }

    async readGridV(): Promise<Float32Array> {
        return this.mapBuffer(this.gridVReadbackBuffer, this.fNumX * this.fNumY * 4);
    }

    async readCellColor(): Promise<Float32Array> {
        return this.mapBuffer(this.cellColorReadbackBuffer, this.fNumX * this.fNumY * 3 * 4);
    }

    private async mapBuffer(buffer: GPUBuffer, size: number): Promise<Float32Array> {
        await buffer.mapAsync(GPUMapMode.READ, 0, size);
        const arrayBuffer = buffer.getMappedRange(0, size);
        const result = new Float32Array(arrayBuffer);
        buffer.unmap();
        return result;
    }

    // ---- Cleanup ----

    destroy(): void {
        [
            this.particlePosBuffer, this.particleVelBuffer, this.particleColorBuffer,
            this.gridUBuffer, this.gridVBuffer, this.prevUBuffer, this.prevVBuffer,
            this.duBuffer, this.dvBuffer, this.pressureBuffer, this.cellTypeBuffer,
            this.densityBuffer, this.cellColorBuffer, this.paramsBuffer,
            this.restDensityResultBuffer,
            this.posReadbackBuffer, this.velReadbackBuffer, this.colorReadbackBuffer,
            this.gridUReadbackBuffer, this.gridVReadbackBuffer, this.cellColorReadbackBuffer
        ].forEach((buf) => buf?.destroy());
    }
}
