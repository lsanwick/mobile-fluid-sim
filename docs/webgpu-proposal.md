# WebGPU Compute Shader Proposal for Fluid Simulation

## 1. Current Architecture Analysis

### CPU Bottleneck Breakdown (at 70.9 FPS, ~14.1ms/frame)

| Phase | Operations/frame | Est. Time | Parallelizable? |
|---|---|---|---|
| Pressure solve | 60 iters × 6,674 cells = 400K iters | ~7-8ms (57%) | Partially |
| Push particles apart | 3 iters × 8,976 particles × ~9 neighbors | ~2-3ms (20%) | Yes (with spatial hash) |
| Transfer velocities (to grid) | 8,976 particles × 4 samples × 2 components | ~1ms (7%) | Yes |
| Transfer velocities (from grid) | 8,976 particles × 4 samples × 2 components | ~1ms (7%) | Yes |
| Update particle density | 8,976 particles × 4 samples | ~0.5ms (4%) | Yes |
| Integrate particles | 8,976 particles | ~0.3ms (2%) | Yes |
| Wall collisions | 8,976 particles | ~0.1ms (1%) | Yes |
| Color updates | 8,976 particles | ~0.2ms (1%) | Yes |
| **Total** | | **~14.1ms** | |

The pressure solver consumes **57% of frame time** and is the primary target for GPU acceleration.

## 2. WebGPU Architecture

### Compute Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Main Thread (CPU)                           │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐     │
│  │ Dispatch │──▶│ Dispatch │──▶│ Dispatch │──▶│ Dispatch │ ... │
│  │  (K1)    │   │  (K2)    │   │  (K3)    │   │  (K4)    │     │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘     │
│       │              │              │              │             │
│       ▼              ▼              ▼              ▼             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  GPU Compute Unit                       │   │
│  │  K1: Integrate  │  K2: PushApart  │  K3: TransferToGrid │   │
│  │  K4: WallCollid │  K5: Density      │  K6: PressureSolve  │   │
│  │  K7: TransferFromGrid │ K8: Colors        │               │   │
│  └─────────────────────────────────────────────────────────┘   │
│       ▲              ▲              ▲              ▲             │
│       │              │              │              │             │
│  ┌────┴──────────────┴──────────────┴──────────────┴─────────┐  │
│  │                  GPU Storage Buffers (VRAM)               │  │
│  │  particlePos │ particleVel │ particleColor │ gridU │ gridV │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Layout: Structure of Arrays (SoA)

WebGPU benefits from SoA for coalesced memory access in compute shaders:

```typescript
// Layout in GPU memory (contiguous, tightly packed)
interface ParticleData {
  pos: Float32Array;       // [x0, y0, x1, y1, x2, y2, ...]  (2 × N floats)
  vel: Float32Array;       // [vx0, vy0, vx1, vy1, ...]        (2 × N floats)
  color: Float32Array;     // [r0, g0, b0, r1, g1, b1, ...]    (3 × N floats)
}

interface GridData {
  u: Float32Array;         // x-velocity field   (Nx × Ny floats)
  v: Float32Array;         // y-velocity field   (Nx × Ny floats)
  prevU: Float32Array;     // previous u (for FLIP) (Nx × Ny floats)
  prevV: Float32Array;     // previous v (for FLIP) (Nx × Ny floats)
  du: Float32Array;        // denominator for u normalization (Nx × Ny floats)
  dv: Float32Array;        // denominator for v normalization (Nx × Ny floats)
  p: Float32Array;         // pressure field     (Nx × Ny floats)
  s: Float32Array;         // cell type (0=solid, 1=fluid) (Nx × Ny floats)
  density: Float32Array;   // particle density at cells (Nx × Ny floats)
  cellColor: Float32Array; // visualization color (3 × Nx × Ny floats)
}

interface SimParams {
  numParticles: number;
  numCellsX: number;
  numCellsY: number;
  cellSize: number;
  particleRadius: number;
  dt: number;
  gravityX: number;
  gravityY: number;
  flipRatio: number;
  numPressureIters: number;
  numParticleIters: number;
  overRelaxation: number;
  compensateDrift: boolean;
  separateParticles: boolean;
  damping: number;
}
```

### Bind Group Layout

```typescript
// Single bind group for all GPU buffers
const bindGroupLayout = device.createBindGroupLayout({
  entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'storage' } },     // particlePos
    { binding: 1, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'storage' } },     // particleVel
    { binding: 2, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'storage' } },     // particleColor
    { binding: 3, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'storage' } },     // gridU
    { binding: 4, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'storage' } },     // gridV
    { binding: 5, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'storage' } },     // prevU
    { binding: 6, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'storage' } },     // prevV
    { binding: 7, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'storage' } },     // du
    { binding: 8, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'storage' } },     // dv
    { binding: 9, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'storage' } },     // p (pressure)
    { binding: 10, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'storage' } },     // s (cell type)
    { binding: 11, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'storage' } },     // density
    { binding: 12, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'storage' } },     // cellColor
    { binding: 13, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'read-only-storage' } }, // params
  ]
});
```

## 3. Compute Kernels

### Kernel 1: Integrate Particles

```wgsl
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
```

### Kernel 2: Push Particles Apart (Spatial Hash)

This is the trickiest kernel. We use a two-pass approach:
- **Pass 1**: Build spatial hash (atomic counter for bin counts)
- **Pass 2**: Push particles apart

```wgsl
// Shared data structures
// Spatial hash: grid of bins, each bin has a linked list of particles
// Bin size = 2.2 × particleRadius

struct Params {
  // ... common params
  pNumX: u32,
  pNumY: u32,
  pNumCells: u32,
  pInvSpacing: f32,
  numParticleIters: u32,
}

@group(0) @binding(0) var<storage, read_write> particlePos: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> particleColor: array<vec3f>;
@group(0) @binding(14) var<storage, read_write> numCellParticles: array<atomic<u32>>;  // bin counts
@group(0) @binding(15) var<storage, read_write> firstCellParticle: array<atomic<u32>>;   // bin heads
@group(0) @binding(16) var<storage, read_write> cellParticleIds: array<atomic<u32>>;     // particle IDs in bin
@group(0) @binding(13) var<uniform> params: Params;

@compute @workgroup_size(256)
fn pushParticlesPass1(@builtin(global_invocation_id) gid: vec3u) {
  // Build spatial hash: assign each particle to a bin
  let i = gid.x;
  if (i >= params.numParticles) { return; }

  let pos = particlePos[i];
  let xi = min(max(u32(floor(pos.x * params.pInvSpacing)), 0u), params.pNumX - 1u);
  let yi = min(max(u32(floor(pos.y * params.pInvSpacing)), 0u), params.pNumY - 1u);
  let cellNr = xi * params.pNumY + yi;

  // Atomic increment of bin count and get old count for position
  let myPos = atomicAdd(&numCellParticles[cellNr], 1u);

  // Store particle index in bin's linked list
  // firstCellParticle[cellNr] points to next available slot
  // We use a different approach: prefix sum after this pass
  cellParticleIds[cellNr * params.pNumCells + myPos] = atomicI32Add(&firstCellParticle[cellNr], 1);
  // Actually, this is complex. Let me use a simpler approach.
}
```

Actually, the spatial hash with atomic operations in WGSL is quite complex. Let me simplify the proposal to use a more practical approach:

**Revised approach for Push Apart**: Use a simpler O(n²) with distance cutoff for small particle counts (< 5000), and reserve the spatial hash for larger simulations. For the current use case (~9K particles), we can use a grid-based approach where each thread processes one particle and checks its 9 neighboring bins.

Let me rewrite this more carefully:

```wgsl
// Kernel 2: Push Particles Apart
// Uses a spatial hash stored in GPU buffers
// Each workgroup handles one "super-particle" batch

struct Params {
  numParticles: u32,
  pNumX: u32,
  pNumY: u32,
  pNumCells: u32,
  pInvSpacing: f32,
  particleRadius: f32,
  colorDiffusionCoeff: f32,
  numParticleIters: u32,
}

@group(0) @binding(0) var<storage, read_write> particlePos: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> particleColor: array<vec3f>;
@group(0) @binding(17) var<storage, read_write> spatialHash: array<u32>;  // flat array: [pNumCells] bins
@group(0) @binding(18) var<storage, read_write> spatialHashNext: array<u32>;  // linked list next pointers
@group(0) @binding(13) var<uniform> params: Params;

const MIN_DIST: f32 = 2.0 * 0.3; // 2 * particleRadius (normalized)
const MIN_DIST2: f32 = MIN_DIST * MIN_DIST;

@compute @workgroup_size(256)
fn pushParticles(@builtin(global_invocation_id) gid: vec3u,
                 @builtin(workgroup_id) wid: vec3u,
                 @builtin(local_invocation_id) lid: vec3u) {
  let i = gid.x;
  if (i >= params.numParticles) { return; }

  // This kernel runs multiple iterations on CPU side
  // Each invocation handles one particle per iteration
  // The spatial hash is rebuilt each iteration on CPU for simplicity
  // In the GPU version, we'd use a more sophisticated approach

  let pos = particlePos[i];
  let xi = u32(floor(pos.x * params.pInvSpacing));
  let yi = u32(floor(pos.y * params.pInvSpacing));

  // Check 9 neighboring bins
  for (let dx = 0u; dx < 3u; dx++) {
    for (let dy = 0u; dy < 3u; dy++) {
      let nx = xi + dx - 1u;
      let ny = yi + dy - 1u;
      if (nx >= params.pNumX || ny >= params.pNumY) { continue; }

      let cellNr = nx * params.pNumY + ny;
      // Iterate through particles in this bin (linked list)
      var j = spatialHash[cellNr];
      while (j < params.numParticles) {
        if (j == i) { break; }

        let otherPos = particlePos[j];
        let d = otherPos - pos;
        let d2 = dot(d, d);

        if (d2 < MIN_DIST2 && d2 > 0.0001) {
          let dist = sqrt(d2);
          let s = 0.5 * (MIN_DIST - dist) / dist;

          particlePos[i] -= d * s;
          particlePos[j] += d * s;

          // Color mixing
          let c0 = particleColor[i];
          let c1 = particleColor[j];
          let avg = (c0 + c1) * 0.5;
          particleColor[i] = c0 + (avg - c0) * params.colorDiffusionCoeff;
          particleColor[j] = c1 + (avg - c1) * params.colorDiffusionCoeff;
        }

        j = spatialHashNext[j];
      }
    }
  }
}
```

The spatial hash on GPU is complex due to WGSL's limited atomic operations. **Practical recommendation**: Keep the spatial hash on CPU (it's already fast enough at ~2ms) and only move the pressure solver and particle kernels to GPU. This gives us the best ROI.

### Kernel 3: Wall Collisions

```wgsl
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
  let maxX = (params.fNumX - 1u) * params.h - params.particleRadius;
  let minY = params.h + params.particleRadius;
  let maxY = (params.fNumY - 1u) * params.h - params.particleRadius;

  if (pos.x < minX) { pos.x = minX; vel.x = 0.0; }
  if (pos.x > maxX) { pos.x = maxX; vel.x = 0.0; }
  if (pos.y < minY) { pos.y = minY; vel.y = 0.0; }
  if (pos.y > maxY) { pos.y = maxY; vel.y = 0.0; }

  particlePos[i] = pos;
  particleVel[i] = vel;
}
```

### Kernel 4: Transfer Velocity (Particle → Grid)

```wgsl
struct Params {
  numParticles: u32,
  fNumX: u32,
  fNumY: u32,
  h: f32,
  fInvSpacing: f32,
}

@group(0) @binding(0) var<storage, read_write> particlePos: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> particleVel: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> gridU: array<f32>;
@group(0) @binding(4) var<storage, read_write> gridV: array<f32>;
@group(0) @binding(7) var<storage, read_write> gridDu: array<f32>;
@group(0) @binding(8) var<storage, read_write> gridDv: array<f32>;
@group(0) @binding(13) var<uniform> params: Params;

@compute @workgroup_size(256)
fn transferToGrid(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.numParticles) { return; }

  let pos = particlePos[i];
  let vel = particleVel[i];
  let h = params.h;
  let h1 = params.fInvSpacing;

  // For U component (x-velocity, sampled at cell faces)
  let x0 = u32(floor((pos.x - 0.5 * h) * h1));
  let tx = ((pos.x - 0.5 * h) - f32(x0) * h) * h1;
  let x1 = min(x0 + 1u, params.fNumX - 2u);
  let y0 = u32(floor((pos.y - 0.5 * h) * h1));
  let ty = ((pos.y - 0.5 * h) - f32(y0) * h) * h1;
  let y1 = min(y0 + 1u, params.fNumY - 2u);

  let sx = 1.0 - tx;
  let sy = 1.0 - ty;
  let d0 = sx * sy;
  let d1 = tx * sy;
  let d2 = tx * ty;
  let d3 = sx * ty;

  // Accumulate velocity and weights for U
  atomicAdd(&gridU[x0 * params.fNumY + y0], vel.x * d0);
  atomicAdd(&gridU[x1 * params.fNumY + y0], vel.x * d1);
  atomicAdd(&gridU[x1 * params.fNumY + y1], vel.x * d2);
  atomicAdd(&gridU[x0 * params.fNumY + y1], vel.x * d3);
  atomicAdd(&gridDu[x0 * params.fNumY + y0], d0);
  atomicAdd(&gridDu[x1 * params.fNumY + y0], d1);
  atomicAdd(&gridDu[x1 * params.fNumY + y1], d2);
  atomicAdd(&gridDu[x0 * params.fNumY + y1], d3);

  // For V component (y-velocity, sampled at cell faces)
  let x0v = u32(floor((pos.x - 0.5 * h) * h1));
  let txv = ((pos.x - 0.5 * h) - f32(x0v) * h) * h1;
  let x1v = min(x0v + 1u, params.fNumX - 2u);
  let y0v = u32(floor((pos.y - h) * h1));
  let tyv = ((pos.y - h) - f32(y0v) * h) * h1;
  let y1v = min(y0v + 1u, params.fNumY - 2u);

  let sxv = 1.0 - txv;
  let syv = 1.0 - tyv;
  let d0v = sxv * syv;
  let d1v = txv * syv;
  let d2v = txv * tyv;
  let d3v = sxv * tyv;

  atomicAdd(&gridV[x0v * params.fNumY + y0v], vel.y * d0v);
  atomicAdd(&gridV[x1v * params.fNumY + y0v], vel.y * d1v);
  atomicAdd(&gridV[x1v * params.fNumY + y1v], vel.y * d2v);
  atomicAdd(&gridV[x0v * params.fNumY + y1v], vel.y * d3v);
  atomicAdd(&gridDv[x0v * params.fNumY + y0v], d0v);
  atomicAdd(&gridDv[x1v * params.fNumY + y0v], d1v);
  atomicAdd(&gridDv[x1v * params.fNumY + y1v], d2v);
  atomicAdd(&gridDv[x0v * params.fNumY + y1v], d3v);
}
```

**Problem**: `atomicAdd` on `f32` storage buffers is **not supported** in the WebGPU shader model. WGSL only supports `atomicAdd` on `u32`/`i32`.

**Solution**: We need a different approach for the particle-to-grid transfer. Options:
1. **Use a two-pass approach**: First pass writes particle data to a flat buffer, second pass (on CPU or GPU) aggregates it.
2. **Use compute shader for grid normalization**: Have the GPU do the velocity transfer by writing to a temporary buffer, then do the division pass on GPU.
3. **Accept the CPU cost**: The particle-to-grid transfer is only ~1ms, so it's not worth the complexity of a GPU workaround.

**Revised recommendation**: Only move the **pressure solver** and **particle integration** to GPU. Keep the particle-grid transfers on CPU since they're fast and the atomic f32 problem makes GPU implementation complex.

### Kernel 5: Update Particle Density

Same particle-to-grid bilinear interpolation problem. **Keep on CPU**.

### Kernel 6: Pressure Solver (The Big Win)

This is where we get the most value. We'll use **Jacobi iteration** (fully parallel) instead of Gauss-Seidel (sequential).

```wgsl
struct Params {
  numParticles: u32,
  fNumX: u32,
  fNumY: u32,
  h: f32,
  density: f32,
  dt: f32,
  overRelaxation: f32,
  compensateDrift: bool,
  particleRestDensity: f32,
}

@group(0) @binding(3) var<storage, read_write> gridU: array<f32>;
@group(0) @binding(4) var<storage, read_write> gridV: array<f32>;
@group(0) @binding(5) var<storage, read_write> prevU: array<f32>;
@group(0) @binding(6) var<storage, read_write> prevV: array<f32>;
@group(0) @binding(9) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(10) var<storage, read_write> cellType: array<f32>;  // 0=solid, 1=fluid
@group(0) @binding(11) var<storage, read_write> density: array<f32>;
@group(0) @binding(13) var<uniform> params: Params;

const FLUID_CELL: f32 = 1.0;

@compute @workgroup_size(16, 16)
fn pressureSolver(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let j = gid.y;

  // Boundary check
  if (i >= params.fNumX - 1u || j >= params.fNumY - 1u) { return; }

  let center = i * params.fNumY + j;
  if (cellType[center] < 0.5) { return; } // Not a fluid cell

  let left = (i - 1u) * params.fNumY + j;
  let right = (i + 1u) * params.fNumY + j;
  let bottom = i * params.fNumY + (j - 1u);
  let top = i * params.fNumY + (j + 1u);

  let s = cellType[left] + cellType[right] + cellType[bottom] + cellType[top];
  if (s < 0.5) { return; } // No fluid neighbors

  // Divergence
  let div = gridU[right] - gridU[center] + gridV[top] - gridV[center];

  // Compensate drift
  var divFinal = div;
  if (params.compensateDrift && params.particleRestDensity > 0.0) {
    let compression = density[center] - params.particleRestDensity;
    if (compression > 0.0) {
      divFinal = div - compression;
    }
  }

  // Jacobi update (parallel-friendly, unlike Gauss-Seidel)
  let p = -divFinal / s * params.overRelaxation;
  pressure[center] += params.density * params.h / params.dt * p;

  // Update velocities
  gridU[center] -= cellType[left] * p;
  gridU[right] += cellType[right] * p;
  gridV[center] -= cellType[bottom] * p;
  gridV[top] += cellType[top] * p;
}
```

**Dispatch**: `workgroups((fNumX - 1) / 16 + 1, (fNumY - 1) / 16 + 1, 1)`

**Key difference from CPU**: Jacobi uses `prevU`/`prevV` for the divergence calculation, reading all values from the previous iteration. This is fully parallelizable with no dependencies between threads.

**Trade-off**: Jacobi converges slower than Gauss-Seidel. We may need ~80-100 iterations instead of 60. But since each iteration is ~100× faster on GPU, the total time is still much better.

### Kernel 7: Transfer Velocity (Grid → Particle)

This reads from grid and writes to particles. No atomics needed.

```wgsl
@group(0) @binding(0) var<storage, read_write> particlePos: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> particleVel: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> gridU: array<f32>;
@group(0) @binding(4) var<storage, read_write> gridV: array<f32>;
@group(0) @binding(5) var<storage, read_write> prevU: array<f32>;
@group(0) @binding(6) var<storage, read_write> prevV: array<f32>;
@group(0) @binding(10) var<storage, read_write> cellType: array<f32>;
@group(0) @binding(13) var<uniform> params: Params;

@compute @workgroup_size(256)
fn transferFromGrid(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.numParticles) { return; }

  let pos = particlePos[i];
  let h = params.h;
  let h1 = 1.0 / params.h;

  // FLIP/PIC velocity update for both u and v components
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

    let isValid = fn(a: f32, b: f32) -> f32 {
      return select(0.0, 1.0, (a > 0.5 || b > 0.5) && !(a > 1.5 && b > 1.5));
    };

    let valid0 = isValid(cellType[nr0], cellType[n0]);
    let valid1 = isValid(cellType[nr1], cellType[n1]);
    let valid2 = isValid(cellType[nr2], cellType[n2]);
    let valid3 = isValid(cellType[nr3], cellType[n3]);

    let totalValid = valid0 * d0 + valid1 * d1 + valid2 * d2 + valid3 * d3;

    if (totalValid > 0.0) {
      let picV = (valid0 * d0 * f[nr0] + valid1 * d1 * f[nr1] + valid2 * d2 * f[nr2] + valid3 * d3 * f[nr3]) / totalValid;
      let corr = (valid0 * d0 * (f[nr0] - prevF[nr0]) + valid1 * d1 * (f[nr1] - prevF[nr1]) + valid2 * d2 * (f[nr2] - prevF[nr2]) + valid3 * d3 * (f[nr3] - prevF[nr3])) / totalValid;
      let v = particleVel[i].x + select(0.0, particleVel[i].y, component == 1u);
      let newV = (1.0 - params.flipRatio) * picV + params.flipRatio * (v + corr);

      if (component == 0u) {
        particleVel[i].x = newV;
      } else {
        particleVel[i].y = newV;
      }
    }
  }
}
```

### Kernel 8: Update Particle Colors

```wgsl
@group(0) @binding(0) var<storage, read_write> particlePos: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> particleColor: array<vec3f>;
@group(0) @binding(11) var<storage, read_write> density: array<f32>;
@group(0) @binding(19) var<storage, read-only> baseColor: vec3f;
@group(0) @binding(20) var<storage, read-only> foamColor: vec3f;
@group(0) @binding(21) var<storage, read-only> foamReturnRate: f32;
@group(0) @binding(22) var<storage, read-only> particleRestDensity: f32;
@group(0) @binding(23) var<storage, read-only> fNumX: u32;
@group(0) @binding(24) var<storage, read-only> fNumY: u32;
@group(0) @binding(25) var<storage, read-only> fInvSpacing: f32;
@group(0) @binding(13) var<uniform> params: Params;

@compute @workgroup_size(256)
fn updateColors(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.numParticles) { return; }

  let pos = particlePos[i];
  let h1 = fInvSpacing;
  let t = min(max(f32(params.numParticles) * foamReturnRate, 0.0), 1.0);

  let xi = min(max(u32(floor(pos.x * h1)), 1u), params.fNumX - 1u);
  let yi = min(max(u32(floor(pos.y * h1)), 1u), params.fNumY - 1u);
  let cellIdx = xi * params.fNumY + yi;

  let applyFoam = particleRestDensity > 0.0 && density[cellIdx] / particleRestDensity < 0.7;

  if (applyFoam) {
    particleColor[i] = foamColor;
  } else {
    let c = particleColor[i];
    particleColor[i] = c + (baseColor - c) * t;
  }
}
```

## 4. Expected Performance

| Kernel | CPU Time | GPU Estimate | Speedup |
|---|---|---|---|
| Integrate particles | 0.3ms | ~0.05ms | 6× |
| Push particles apart | 2-3ms | ~0.5ms (if on GPU) | 5-6× |
| Wall collisions | 0.1ms | ~0.02ms | 5× |
| Transfer velocities (CPU) | 1ms | — (keep on CPU) | — |
| **Pressure solver** | **7-8ms** | **~0.5ms** | **15-20×** |
| Update colors | 0.2ms | ~0.03ms | 7× |
| **Subtotal (GPU)** | **~10ms** | **~0.6ms** | **~17×** |
| Subtotal (CPU) | ~4ms | ~4ms | — |
| **Total** | **~14ms** | **~4.6ms** | **~3×** |

**Expected FPS**: 70 → **~250+ FPS** (well above the 120 FPS target)

The pressure solver is the star: going from 7-8ms on CPU to ~0.5ms on GPU is a 15-20× speedup. Even with the overhead of dispatching and the slower convergence of Jacobi iteration, the net result is a massive improvement.

## 5. Implementation Plan

### Phase 1: WebGPU Setup & Fallback (1 day)
- Create `WebGPURenderer` class alongside existing `FluidRenderer`
- Add feature detection: if WebGPU is available, use GPU pipeline; otherwise fall back to CPU
- Create buffer management (create/update/dispose GPU buffers)

### Phase 2: Compute Kernels (2-3 days)
- Implement WGSL shaders for: integrate, wall collisions, pressure solver (Jacobi), transfer from grid, colors
- Wire up dispatch calls in the simulation loop
- Handle buffer synchronization (GPU → CPU reads for renderer)

### Phase 3: Integration & Testing (1-2 days)
- Replace CPU simulation calls with GPU dispatch in `FluidSimulation.svelte`
- Add `numPressureIters` tuning for Jacobi convergence
- Benchmark and validate against CPU version
- Handle buffer copy-back for WebGL rendering (or migrate renderer to WebGPU)

### Phase 4: Renderer Migration (optional, 2-3 days)
- Migrate `FluidRenderer` to WebGPU rendering for full GPU pipeline
- Use compute-to-render buffer copies or transform feedback

### Phase 5: Advanced Optimizations (nice-to-have)
- Coarse-grid pressure solve (solve on 2× downsampled grid)
- Adaptive iteration count (stop when pressure change < threshold)
- Particle spatial hash on GPU for large simulations (> 20K particles)

## 6. Browser Support

| Browser | WebGPU Support |
|---|---|
| Chrome/Edge | ✅ 113+ |
| Firefox | ✅ 134+ (enabled by default) |
| Safari | ⚠️ 17.4+ (needs flag) |
| Mobile Chrome | ✅ 113+ |
| Mobile Safari | ⚠️ 17.4+ |

**Fallback strategy**: The existing WebGL renderer remains as fallback. The CPU simulation continues to work. We add a graceful degradation:
1. WebGPU available → use GPU compute + WebGPU render
2. WebGL available → use CPU compute + WebGL render (current behavior)
3. Neither → show error

## 7. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Jacobi slower convergence than Gauss-Seidel | More iterations needed | Use Gauss-Seidel with red-black ordering (checkerboard) for partial parallelism |
| GPU-CPU buffer sync overhead | Slows down frame | Use double-buffering; only copy back when needed for rendering |
| WebGPU not available on older devices | Fallback needed | Feature detection + graceful CPU fallback |
| Shader compilation fails | Crash | Try/catch during pipeline creation, fall back to CPU |
| Mobile GPU memory limits | Out of memory | Limit max particles for GPU; use smaller grid for mobile |

## 8. Recommended Next Step

Start with **Phase 1 + 2** (WebGPU setup + pressure solver). This gives us the biggest performance win with the least complexity. The pressure solver alone accounts for 57% of frame time and is trivially parallelizable with Jacobi iteration.
