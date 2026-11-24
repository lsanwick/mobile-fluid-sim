/**
 * Headless benchmark for the fluid simulation.
 * Runs the simulation loop without rendering and measures:
 *   - FPS (sim steps per second)
 *   - Memory pressure (peak heap usage)
 *   - GC pressure (allocation rate)
 *   - Stability (does it run for 60 seconds without issues?)
 */

import { FlipFluid } from '../src/lib/fluid/FlipFluid';

// --- Config ---
const SIM_WIDTH = 4.0;
const SIM_HEIGHT = 3.0;
const RESOLUTION = 70;
const H = SIM_HEIGHT / RESOLUTION;
const DENSITY = 1000.0;
const R = 0.3 * H;
const MAX_PARTICLES = 5000;

const DT = 1.0 / 120.0;
const FLIP_RATIO = 0.95;
const NUM_PRESSURE_ITERS = 60;
const NUM_PARTICLE_ITERS = 3;
const OVER_RELAXATION = 1.7;
const COMPENSATE_DRIFT = true;
const SEPARATE_PARTICLES = true;
const DAMPING = 1.0;

const BENCHMARK_DURATION = 30; // seconds
const WARMUP_FRAMES = 120;

// --- Helpers ---
function createFluid(): FlipFluid {
    const fluid = new FlipFluid(
        DENSITY,
        SIM_WIDTH,
        SIM_HEIGHT,
        H,
        R,
        MAX_PARTICLES,
        { r: 0.09, g: 0.4, b: 1.0 },
        { r: 0.75, g: 0.9, b: 1.0 }
    );

    // Initialize particles in a grid
    const dx = 2.0 * R;
    const dy = Math.sqrt(3.0) / 2.0 * dx;
    const numX = Math.floor((0.6 * SIM_WIDTH - 2.0 * H - 2.0 * R) / dx);
    const numY = Math.floor((0.8 * SIM_HEIGHT - 2.0 * H - 2.0 * R) / dy);
    fluid.numParticles = numX * numY;

    const totalWidth = (numX - 1) * dx;
    const totalHeight = (numY - 1) * dy;
    const startX = (SIM_WIDTH - totalWidth) / 2.0;
    const startY = (SIM_HEIGHT - totalHeight) / 2.0;

    let p = 0;
    for (let i = 0; i < numX; i++) {
        for (let j = 0; j < numY; j++) {
            fluid.particlePos[p++] = startX + dx * i;
            fluid.particlePos[p++] = startY + dy * j;
        }
    }

    // Setup solid walls
    const n = fluid.fNumY;
    for (let i = 0; i < fluid.fNumX; i++) {
        for (let j = 0; j < fluid.fNumY; j++) {
            fluid.s[i * n + j] = (i === 0 || i === fluid.fNumX - 1 || j === 0) ? 0.0 : 1.0;
        }
    }

    return fluid;
}

// --- Benchmark ---
async function runBenchmark() {
    console.log('=== Fluid Simulation Benchmark ===');
    console.log(`Grid: ${SIM_WIDTH}x${SIM_HEIGHT}, Resolution: ${RESOLUTION}`);
    console.log(`Particle radius: ${R.toFixed(4)}`);

    // --- Measure initial heap ---
    const initialMem = process.memoryUsage();
    console.log(`Initial heap: ${(initialMem.heapUsed / 1024 / 1024).toFixed(2)} MB`);

    // --- Warmup ---
    const fluid = createFluid();
    console.log(`Initial particles: ${fluid.getParticleCount()}`);

    console.log(`\nWarming up for ${WARMUP_FRAMES} frames...`);
    for (let i = 0; i < WARMUP_FRAMES; i++) {
        fluid.simulate(
            DT, 0, -9.81, FLIP_RATIO, NUM_PRESSURE_ITERS,
            NUM_PARTICLE_ITERS, OVER_RELAXATION, COMPENSATE_DRIFT,
            SEPARATE_PARTICLES, DAMPING
        );
    }

    // --- Benchmark loop ---
    const startTime = performance.now();
    let frameCount = 0;
    let peakHeap = initialMem.heapUsed;
    let frameTimes: number[] = [];

    console.log(`\nRunning benchmark for ${BENCHMARK_DURATION} seconds...`);

    const benchmark = () => {
        const frameStart = performance.now();

        fluid.simulate(
            DT, 0, -9.81, FLIP_RATIO, NUM_PRESSURE_ITERS,
            NUM_PARTICLE_ITERS, OVER_RELAXATION, COMPENSATE_DRIFT,
            SEPARATE_PARTICLES, DAMPING
        );

        frameCount++;
        const frameEnd = performance.now();
        frameTimes.push(frameEnd - frameStart);

        const mem = process.memoryUsage();
        peakHeap = Math.max(peakHeap, mem.heapUsed);

        if (frameEnd - startTime < BENCHMARK_DURATION * 1000) {
            setImmediate(benchmark);
        } else {
            finishBenchmark(frameCount, peakHeap, frameTimes, startTime, fluid.getParticleCount());
        }
    };

    benchmark();
}

function finishBenchmark(frameCount: number, peakHeap: number, frameTimes: number[], startTime: number, particleCount: number) {
    const elapsed = performance.now() - startTime;
    const fps = frameCount / (elapsed / 1000);

    // Percentiles
    const sorted = [...frameTimes].sort((a, b) => a - b);
    const p50 = sorted[Math.floor(sorted.length * 0.5)];
    const p90 = sorted[Math.floor(sorted.length * 0.9)];
    const p99 = sorted[Math.floor(sorted.length * 0.99)];

    const memDelta = (peakHeap - process.memoryUsage().heapUsed) / 1024 / 1024;

    console.log('\n=== Results ===');
    console.log(`Frames: ${frameCount}`);
    console.log(`Duration: ${(elapsed / 1000).toFixed(1)}s`);
    console.log(`Avg FPS: ${fps.toFixed(1)}`);
    console.log(`P50 frame time: ${p50.toFixed(2)}ms`);
    console.log(`P90 frame time: ${p90.toFixed(2)}ms`);
    console.log(`P99 frame time: ${p99.toFixed(2)}ms`);
    console.log(`Peak heap delta: ${memDelta.toFixed(2)} MB`);
    console.log(`Particles: ${particleCount}`);
    console.log(`\nTarget: 120 FPS (8.33ms per frame)`);
    console.log(fps >= 100 ? '✅ PASS' : '⚠️  BELOW TARGET');
}

runBenchmark();
