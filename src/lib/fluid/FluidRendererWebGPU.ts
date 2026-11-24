import type { FlipFluid } from './FlipFluid';
import { GPUShaderStage, GPUBufferUsage } from '../gpu-types';

export interface RenderConfig {
    showParticles: boolean;
    showGrid: boolean;
    simWidth: number;
    simHeight: number;
}

// Buffers provided by WebGPUSimulation (must have STORAGE | COPY_SRC | VERTEX usage)
export interface SimulationBuffers {
    particlePosBuffer: GPUBuffer;
    particleColorBuffer: GPUBuffer;
    cellColorBuffer: GPUBuffer;
    fNumX: number;
    fNumY: number;
    fNumCells: number;
    h: number;
    numParticles: number;
    maxParticles: number;
}

export class FluidRendererWebGPU {
    private device: GPUDevice;
    private context: GPUCanvasContext;
    private format: GPUTextureFormat;
    private swapChain: GPUTextureView;

    // Pipeline state
    private pointPipeline!: GPURenderPipeline;
    private pointPipelineBindGroup!: GPUBindGroup;

    // Vertex buffers (provided by simulation)
    private simBuffers: SimulationBuffers | null = null;

    // Uniform buffer
    private uniformBuffer: GPUBuffer;
    private uniformBindGroup: GPUBindGroup;

    // Canvas dimensions
    private canvasWidth = 0;
    private canvasHeight = 0;

    constructor(canvas: HTMLCanvasElement, device: GPUDevice) {
        this.device = device;

        const context = canvas.getContext('webgpu');
        if (!context) {
            throw new Error('WebGPU context not supported');
        }
        this.context = context;

        const format = navigator.gpu.getPreferredCanvasFormat() as GPUTextureFormat;
        this.format = format;

        context.configure({
            device,
            format,
            alphaMode: 'premultiplied'
        });
        this.swapChain = context.getCurrentTexture().createView();

        // Create shaders
        const pointShaderModule = device.createShaderModule({
            code: pointVertexShaderWGSL + pointFragmentShaderWGSL
        });

        // Create bind group layout for point rendering
        const pointBindGroupLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: { type: 'uniform' }
                }
            ]
        });

        // Create point rendering pipeline
        this.pointPipeline = device.createRenderPipeline({
            layout: device.createPipelineLayout({
                bindGroupLayouts: [pointBindGroupLayout]
            }),
            vertex: {
                module: pointShaderModule,
                entryPoint: 'pointVertex',
                buffers: [
                    {
                        arrayStride: 2 * 4,
                        attributes: [
                            {
                                shaderLocation: 0,
                                offset: 0,
                                format: 'float2'
                            }
                        ]
                    },
                    {
                        arrayStride: 3 * 4,
                        attributes: [
                            {
                                shaderLocation: 1,
                                offset: 0,
                                format: 'float3'
                            }
                        ]
                    }
                ]
            },
            fragment: {
                module: pointShaderModule,
                entryPoint: 'pointFragment',
                targets: [
                    {
                        format,
                        blend: {
                            color: {
                                srcFactor: 'src-alpha',
                                dstFactor: 'one-minus-src-alpha',
                                operation: 'add'
                            },
                            alpha: {
                                srcFactor: 'src-alpha',
                                dstFactor: 'one-minus-src-alpha',
                                operation: 'add'
                            }
                        }
                    }
                ]
            },
            primitive: {
                topology: 'point-list'
            }
        });

        // Create uniform buffer and bind group
        this.uniformBuffer = device.createBuffer({
            size: 4 * 4, // 4 floats
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.uniformBindGroup = device.createBindGroup({
            layout: pointBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.uniformBuffer
                    }
                }
            ]
        });
    }

    // Set the simulation buffers to render from
    setSimulationBuffers(buffers: SimulationBuffers): void {
        this.simBuffers = buffers;
    }

    render(fluid: FlipFluid, config: RenderConfig): void {
        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
            colorAttachments: [
                {
                    view: this.swapChain,
                    clearValue: { r: 0.02, g: 0.1, b: 0.2, a: 1.0 },
                    loadOp: 'clear',
                    storeOp: 'store'
                }
            ]
        });

        // Render grid cells
        if (config.showGrid && this.simBuffers) {
            this.renderGrid(pass, fluid, config);
        }

        // Render particles
        if (config.showParticles && this.simBuffers) {
            this.renderParticles(pass, fluid, config);
        }

        pass.end();
        this.device.queue.submit([encoder.finish()]);
    }

    private renderGrid(pass: GPURenderPassEncoder, fluid: FlipFluid, config: RenderConfig): void {
        if (!this.simBuffers) return;

        const numCells = this.simBuffers.fNumCells;
        if (numCells === 0) return;

        // Upload color data (grid colors change every frame)
        this.device.queue.writeBuffer(this.simBuffers.cellColorBuffer, 0, fluid.cellColor);

        // Set uniform: domain size, point size, draw disk (0 = square)
        const pointSize = 0.9 * fluid.h / config.simWidth * this.canvasWidth;
        this.uploadUniforms(config.simWidth, config.simHeight, pointSize, 0.0);

        pass.setPipeline(this.pointPipeline);
        pass.setBindGroup(0, this.uniformBindGroup);
        pass.setVertexBuffer(0, this.simBuffers.particlePosBuffer);
        pass.setVertexBuffer(1, this.simBuffers.cellColorBuffer);
        pass.draw(numCells, 1, 0, 0);
    }

    private renderParticles(pass: GPURenderPassEncoder, fluid: FlipFluid, config: RenderConfig): void {
        if (!this.simBuffers) return;

        const numParticles = this.simBuffers.numParticles;
        if (numParticles === 0) return;

        // Set uniform: domain size, point size, draw disk (1 = disk)
        const pointSize = 2.0 * fluid.particleRadius / config.simWidth * this.canvasWidth;
        this.uploadUniforms(config.simWidth, config.simHeight, pointSize, 1.0);

        pass.setPipeline(this.pointPipeline);
        pass.setBindGroup(0, this.uniformBindGroup);
        pass.setVertexBuffer(0, this.simBuffers.particlePosBuffer);
        pass.setVertexBuffer(1, this.simBuffers.particleColorBuffer);
        pass.draw(numParticles, 1, 0, 0);
    }

    private uploadUniforms(domainX: number, domainY: number, pointSize: number, drawDisk: number): void {
        const uniformData = new Float32Array([domainX, domainY, pointSize, drawDisk]);
        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);
    }

    resize(width: number, height: number): void {
        this.canvasWidth = width;
        this.canvasHeight = height;
        this.context.configure({
            device: this.device,
            format: this.format,
            alphaMode: 'premultiplied'
        });
    }

    destroy(): void {
        this.uniformBuffer.destroy();
        this.context.unconfigure();
    }
}

// WGSL shaders

const pointVertexShaderWGSL = `
struct Uniforms {
    domainSize: vec2f,
    pointSize: f32,
    drawDisk: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct PointVertexInput {
    @location(0) position: vec2f,
    @location(1) color: vec3f,
};

struct PointVertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec3f,
    @location(1) drawDisk: f32,
};

@vertex
fn pointVertex(input: PointVertexInput) -> PointVertexOutput {
    var output: PointVertexOutput;
    let screenTransform = vec2f(2.0 / uniforms.domainSize.x, 2.0 / uniforms.domainSize.y);
    output.position = vec4f(input.position * screenTransform + vec2f(-1.0, -1.0), 0.0, 1.0);
    output.position.y *= -1.0;
    output.color = input.color;
    output.drawDisk = uniforms.drawDisk;
    return output;
}
`;

const pointFragmentShaderWGSL = `
@fragment
fn pointFragment(@location(0) color: vec3f, @location(1) drawDisk: f32) -> @location(0) vec4f {
    if (drawDisk > 0.5) {
        let r2 = dot(gl_PointCoord - vec2f(0.5), gl_PointCoord - vec2f(0.5));
        if (r2 > 0.25) {
            discard;
        }
        let alpha = 1.0 - smoothstep(0.15, 0.25, r2);
        return vec4f(color, alpha * 0.8);
    }
    return vec4f(color, 1.0);
}
`;
