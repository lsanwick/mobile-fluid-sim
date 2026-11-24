/**
 * Compare different pressure iteration counts.
 */

import { FlipFluid } from '../src/lib/fluid/FlipFluid';

const SIM_WIDTH = 4.0;
const SIM_HEIGHT = 3.0;
const RESOLUTION = 70;
const H = SIM_HEIGHT / RESOLUTION;
const DENSITY = 1000.0;
const R = 0.3 * H;

function createFluid(): FlipFluid {
    const fluid = new FlipFluid(
        DENSITY, SIM_WIDTH, SIM_HEIGHT, H, R, 5000,
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

async function bench(iters: number, label: string) {
    const fluid = createFluid();
    const WARMUP = 120;
    const BENCH = 30;
    const DT = 1 / 120;

    for (let i = 0; i < WARMUP; i++) {
        fluid.simulate(DT, 0, -9.81, 0.95, iters, 3, 1.7, true, true, 1.0);
    }

    const start = performance.now();
    let count = 0;
    const loop = () => {
        fluid.simulate(DT, 0, -9.81, 0.95, iters, 3, 1.7, true, true, 1.0);
        count++;
        const now = performance.now();
        if (now - start < BENCH * 1000) setImmediate(loop);
        else console.log(`${label}: ${count} frames in ${(now - start).toFixed(0)}ms = ${(count / ((now - start) / 1000)).toFixed(1)} FPS`);
    };
    setImmediate(loop);
}

console.log('=== Pressure Iteration Comparison ===');
await bench(60, '60 iters (current)');
await bench(40, '40 iters');
await bench(30, '30 iters');
await bench(20, '20 iters');
await bench(15, '15 iters');
await bench(10, '10 iters');
