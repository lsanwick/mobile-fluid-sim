/**
 * Memory leak test: creates and destroys fluid instances repeatedly.
 * If there's a leak, heap usage will grow monotonically.
 */

import { FlipFluid } from '../src/lib/fluid/FlipFluid';

const SIM_WIDTH = 4.0;
const SIM_HEIGHT = 3.0;
const RESOLUTION = 70;
const H = SIM_HEIGHT / RESOLUTION;
const DENSITY = 1000.0;
const R = 0.3 * H;
const MAX_PARTICLES = 5000;

function createFluid(): FlipFluid {
    const fluid = new FlipFluid(
        DENSITY, SIM_WIDTH, SIM_HEIGHT, H, R, MAX_PARTICLES,
        { r: 0.09, g: 0.4, b: 1.0 },
        { r: 0.75, g: 0.9, b: 1.0 }
    );

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

    const n = fluid.fNumY;
    for (let i = 0; i < fluid.fNumX; i++) {
        for (let j = 0; j < fluid.fNumY; j++) {
            fluid.s[i * n + j] = (i === 0 || i === fluid.fNumX - 1 || j === 0) ? 0.0 : 1.0;
        }
    }

    return fluid;
}

async function runMemoryTest() {
    console.log('=== Memory Leak Test ===');
    console.log(`Grid: ${SIM_WIDTH}x${SIM_HEIGHT}, Resolution: ${RESOLUTION}`);
    console.log(`Max particles: ${MAX_PARTICLES}`);

    const phases = [
        { name: 'Initial', iterations: 0 },
        { name: '100 creations', iterations: 100 },
        { name: '500 creations', iterations: 500 },
        { name: '1000 creations', iterations: 1000 },
        { name: '2000 creations', iterations: 2000 },
    ];

    let leaked = 0;

    for (const phase of phases) {
        const before = process.memoryUsage();
        const heapBefore = before.heapUsed;

        for (let i = 0; i < phase.iterations; i++) {
            const fluid = createFluid();
            // Run a few simulation steps to exercise all code paths
            for (let s = 0; s < 10; s++) {
                fluid.simulate(
                    1 / 120, 0, -9.81, 0.95, 60, 3, 1.7, true, true, 1.0
                );
            }
            // Explicitly null out references
            fluid.particlePos = null as any;
            fluid.particleColor = null as any;
            fluid.particleVel = null as any;
            fluid.particleDensity = null as any;
            fluid.u = null as any;
            fluid.v = null as any;
            fluid.du = null as any;
            fluid.dv = null as any;
            fluid.prevU = null as any;
            fluid.prevV = null as any;
            fluid.p = null as any;
            fluid.s = null as any;
            fluid.cellType = null as any;
            fluid.cellColor = null as any;
            fluid.numCellParticles = null as any;
            fluid.firstCellParticle = null as any;
            fluid.cellParticleIds = null as any;
        }

        // Force GC if available
        if ((globalThis as any).gc) {
            (globalThis as any).gc();
        }

        const after = process.memoryUsage();
        const heapAfter = after.heapUsed;
        const delta = (heapAfter - heapBefore) / 1024 / 1024;

        console.log(`${phase.name}: heap delta = ${delta.toFixed(2)} MB (total: ${(heapAfter / 1024 / 1024).toFixed(2)} MB)`);

        leaked += delta;
    }

    console.log(`\nTotal leaked: ${leaked.toFixed(2)} MB over ${phases[phases.length - 1].iterations} iterations`);
    console.log(leaked < 50 ? '✅ No significant leak' : '⚠️  Possible leak detected');
}

runMemoryTest();
