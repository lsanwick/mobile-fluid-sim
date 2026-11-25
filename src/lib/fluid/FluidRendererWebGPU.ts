import type { FlipFluid } from './FlipFluid';
import { GPUShaderStage, GPUBufferUsage } from '../gpu-types';

export interface RenderConfig {
    showParticles: boolean;
    showGrid: boolean;
    simWidth: number;
    simHeight: number;
}

// Buffers provided by WebGPUSimulation (must have VERTEX usage)
export interface SimulationBuffers {
    particlePosBuffer: GPUBuffer;
    particleColorBuffer: GPUBuffer;
    cellColorBuffer: GPUBuffer;
    cellColorOffset: number;
    fNumX: number;
    fNumY: number;
    fNumCells: number;
    h: number;
    numParticles: number;
}

// WebGPU has no point sprites (points are always 1px), so every particle /
// grid cell is drawn as an instanced quad and the disk is cut in the fragment shader.
const pointShaderWGSL = /* wgsl */ `
struct Uniforms {
    domainSize: vec2f,
    halfSize: f32,
    drawDisk: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec3f,
    @location(1) local: vec2f,
};

const CORNERS = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
    vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
);

@vertex
fn vs(
    @builtin(vertex_index) vi: u32,
    @location(0) center: vec2f,
    @location(1) color: vec3f
) -> VertexOutput {
    let corner = CORNERS[vi];
    let p = center + corner * uniforms.halfSize;
    var out: VertexOutput;
    out.position = vec4f(p * (2.0 / uniforms.domainSize) - 1.0, 0.0, 1.0);
    out.color = color;
    out.local = corner;
    return out;
}

@fragment
fn fs(in: VertexOutput) -> @location(0) vec4f {
    if (uniforms.drawDisk > 0.5) {
        // Same falloff as the WebGL shader (gl_PointCoord - 0.5 == local * 0.5)
        let r2 = 0.25 * dot(in.local, in.local);
        if (r2 > 0.25) {
            discard;
        }
        let alpha = (1.0 - smoothstep(0.15, 0.25, r2)) * 0.8;
        return vec4f(in.color * alpha, alpha);
    }
    return vec4f(in.color, 1.0);
}
`;

export class FluidRendererWebGPU {
    private device: GPUDevice;
    private context: GPUCanvasContext;
    private format: GPUTextureFormat;
    private pipeline: GPURenderPipeline;

    private simBuffers: SimulationBuffers | null = null;

    private particleUniforms: GPUBuffer;
    private particleBindGroup: GPUBindGroup;
    private gridUniforms: GPUBuffer;
    private gridBindGroup: GPUBindGroup;

    // Cell centers for grid display (built lazily)
    private gridCenterBuffer: GPUBuffer | null = null;

    private canvasWidth = 0;
    private canvasHeight = 0;

    constructor(canvas: HTMLCanvasElement, device: GPUDevice) {
        this.device = device;

        const context = canvas.getContext('webgpu');
        if (!context) {
            throw new Error('WebGPU context not supported');
        }
        this.context = context;
        this.format = navigator.gpu.getPreferredCanvasFormat() as GPUTextureFormat;
        context.configure({ device, format: this.format, alphaMode: 'premultiplied' });

        const module = device.createShaderModule({ code: pointShaderWGSL });

        const bindGroupLayout = device.createBindGroupLayout({
            entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }]
        });

        this.pipeline = device.createRenderPipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            vertex: {
                module,
                entryPoint: 'vs',
                buffers: [
                    {
                        arrayStride: 2 * 4,
                        stepMode: 'instance',
                        attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }]
                    },
                    {
                        arrayStride: 3 * 4,
                        stepMode: 'instance',
                        attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x3' }]
                    }
                ]
            },
            fragment: {
                module,
                entryPoint: 'fs',
                targets: [
                    {
                        format: this.format,
                        // Premultiplied-alpha "over"
                        blend: {
                            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }
                        }
                    }
                ]
            },
            primitive: { topology: 'triangle-list' }
        });

        const makeUniforms = () =>
            device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.particleUniforms = makeUniforms();
        this.gridUniforms = makeUniforms();
        this.particleBindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [{ binding: 0, resource: { buffer: this.particleUniforms } }]
        });
        this.gridBindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [{ binding: 0, resource: { buffer: this.gridUniforms } }]
        });

        this.canvasWidth = canvas.width;
        this.canvasHeight = canvas.height;
    }

    setSimulationBuffers(buffers: SimulationBuffers): void {
        this.simBuffers = buffers;
    }

    render(fluid: FlipFluid, config: RenderConfig): void {
        if (this.canvasWidth === 0 || this.canvasHeight === 0) return;

        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
            colorAttachments: [
                {
                    // The swap-chain texture changes every frame
                    view: this.context.getCurrentTexture().createView(),
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    loadOp: 'clear',
                    storeOp: 'store'
                }
            ]
        });

        if (this.simBuffers) {
            pass.setPipeline(this.pipeline);
            if (config.showGrid) this.renderGrid(pass, fluid, config);
            if (config.showParticles) this.renderParticles(pass, fluid, config);
        }

        pass.end();
        this.device.queue.submit([encoder.finish()]);
    }

    private renderGrid(pass: GPURenderPassEncoder, fluid: FlipFluid, config: RenderConfig): void {
        const sb = this.simBuffers!;
        if (sb.fNumCells === 0) return;

        if (!this.gridCenterBuffer) {
            const centers = new Float32Array(2 * sb.fNumCells);
            let p = 0;
            for (let i = 0; i < sb.fNumX; i++) {
                for (let j = 0; j < sb.fNumY; j++) {
                    centers[p++] = (i + 0.5) * sb.h;
                    centers[p++] = (j + 0.5) * sb.h;
                }
            }
            this.gridCenterBuffer = this.device.createBuffer({
                size: centers.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
            });
            this.device.queue.writeBuffer(this.gridCenterBuffer, 0, centers);
        }

        this.writeUniforms(this.gridUniforms, config, 0.45 * fluid.h, 0.0);
        pass.setBindGroup(0, this.gridBindGroup);
        pass.setVertexBuffer(0, this.gridCenterBuffer);
        pass.setVertexBuffer(1, sb.cellColorBuffer, sb.cellColorOffset);
        pass.draw(6, sb.fNumCells, 0, 0);
    }

    private renderParticles(pass: GPURenderPassEncoder, fluid: FlipFluid, config: RenderConfig): void {
        const sb = this.simBuffers!;
        if (sb.numParticles === 0) return;

        this.writeUniforms(this.particleUniforms, config, fluid.particleRadius, 1.0);
        pass.setBindGroup(0, this.particleBindGroup);
        pass.setVertexBuffer(0, sb.particlePosBuffer);
        pass.setVertexBuffer(1, sb.particleColorBuffer);
        pass.draw(6, sb.numParticles, 0, 0);
    }

    private writeUniforms(buffer: GPUBuffer, config: RenderConfig, halfSize: number, drawDisk: number): void {
        this.device.queue.writeBuffer(
            buffer,
            0,
            new Float32Array([config.simWidth, config.simHeight, halfSize, drawDisk])
        );
    }

    resize(width: number, height: number): void {
        this.canvasWidth = width;
        this.canvasHeight = height;
    }

    destroy(): void {
        this.particleUniforms.destroy();
        this.gridUniforms.destroy();
        this.gridCenterBuffer?.destroy();
        this.context.unconfigure();
    }
}
