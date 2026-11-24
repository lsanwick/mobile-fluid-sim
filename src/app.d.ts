// See https://svelte.dev/docs/kit/types#app.d.ts
// for information about these interfaces
declare global {
	namespace App {
		// interface Error {}
		// interface Locals {}
		// interface PageData {}
		// interface PageState {}
		// interface Platform {}
	}

	// WebGPU type definitions
	interface GPUBufferUsage {
		COPY_SRC: number;
		COPY_DST: number;
		INDEX: number;
		VERTEX: number;
		UNIFORM: number;
		STORAGE: number;
		INDIRECT: number;
		QUERY_RESOLVE: number;
	}

	interface GPUTextureUsage {
		TRANSFER_SRC: number;
		TRANSFER_DST: number;
		SAMPLED: number;
		STORE: number;
		OUTPUT_ATTACHMENT: number;
		READ_ONLY_BINDING: number;
		COPY_SRC: number;
		COPY_DST: number;
	}

	interface GPUShaderStage {
		VERTEX: number;
		FRAGMENT: number;
		COMPUTE: number;
	}

	interface GPUColorDict {
		r: number;
		g: number;
		b: number;
		a: number;
	}

	interface GPUExtent3DDict {
		width: number;
		height: number;
		depthOrArrayLayers: number;
	}

	interface GPUOrigin3DDict {
		x: number;
		y: number;
		z: number;
	}

	interface GPUBindGroupEntry {
		binding: number;
		resource: GPUSampler | GPUTextureView | GPUBufferBinding;
	}

	interface GPUBindGroupDescriptor {
		layout: GPUBindGroupLayout;
		entries: GPUBindGroupEntry[];
	}

	interface GPUBindGroup {
		readonly label: string;
	}

	interface GPUBindGroupLayoutEntry {
		binding: number;
		visibility: number;
		buffer?: {
			type?: 'uniform' | 'storage' | 'read-only-storage';
			hasDynamicOffset?: boolean;
			minBindingSize?: number;
		};
		sampler?: { type?: 'filtering' | 'non-filtering' | 'comparison' };
		texture?: {
			sampleType?: 'float' | 'unfilterable-float' | 'depth' | 'sint' | 'uint';
			multisampled?: boolean;
			viewFormatCount?: number;
			viewFormats?: GPUTextureFormat[];
		};
		storageTexture?: {
			access?: 'write-only';
			format: GPUTextureFormat;
			viewFormatCount?: number;
			viewFormats?: GPUTextureFormat[];
		};
	}

	interface GPUBindGroupLayoutDescriptor {
		entries: GPUBindGroupLayoutEntry[];
	}

	interface GPUPipelineLayoutDescriptor {
		bindGroupLayouts: GPUBindGroupLayout[];
	}

	interface GPUComputePipelineDescriptor {
		layout: GPUPipelineLayout | 'auto';
		compute: {
			module: GPUShaderModule;
			entryPoint: string;
		};
	}

	interface GPUBuffer {
		readonly size: number;
		readonly mapped: boolean;
		readonly mappedAtCreation: boolean;
		readonly label: string;
		mapAsync(offset: number, size: number, mode?: number): Promise<void>;
		getMappedRange(offset?: number, size?: number): ArrayBuffer;
		unmap(): void;
		destroy(): void;
	}

	interface GPUBufferBinding {
		buffer: GPUBuffer;
		offset?: number;
		size?: number;
	}

	interface GPURenderPassDescriptor {
		colorAttachments: GPURenderPassColorAttachment[];
		depthStencilAttachment?: GPURenderPassDepthStencilAttachment;
		occlusionQuerySet?: GPUQuerySet;
		label?: string;
	}

	interface GPURenderPassColorAttachment {
		view: GPUTextureView;
		depthSlice?: number;
		resolveTarget?: GPUTextureView;
		clearValue?: GPUColorDict;
		loadOp: 'clear' | 'load';
		storeOp: 'store' | 'discard';
	}

	interface GPURenderPassDepthStencilAttachment {
		view: GPUTextureView;
		depthClearValue?: number;
		depthLoadOp?: 'clear' | 'load';
		depthStoreOp?: 'store' | 'discard';
		depthReadOnly?: boolean;
		stencilClearValue?: number;
		stencilLoadOp?: 'clear' | 'load';
		stencilStoreOp?: 'store' | 'discard';
		stencilReadOnly?: boolean;
	}

	interface GPUCommandEncoderDescriptor {
		label?: string;
	}

	interface GPUCommandBufferDescriptor {
		label?: string;
	}

	interface GPUQueueDescriptor {
		label?: string;
	}

	interface GPUShaderModuleDescriptor {
		code: string;
		label?: string;
	}

	interface GPUTextureDescriptor {
		size: number | number[] | GPUExtent3DDict;
		mipLevelCount?: number;
		sampleCount?: number;
		dimension?: '1d' | '2d' | '3d';
		format: GPUTextureFormat;
		usage: number;
		viewFormats?: GPUTextureFormat[];
		label?: string;
	}

	interface GPUTextureViewDescriptor {
		dimension?: '1d' | '2d' | '3d' | 'cube';
		format: GPUTextureFormat;
		baseMipLevel?: number;
		mipLevelCount?: number;
		baseArrayLayer?: number;
		arrayLayerCount?: number;
		aspect?: 'depth-only' | 'stencil-only' | 'all';
		label?: string;
	}

	interface GPUComputePassDescriptor {
		label?: string;
		timestampWrites?: GPUComputePassTimestampWrites;
	}

	interface GPUComputePassTimestampWrites {
		querySet: GPUQuerySet;
		beginningOccurrenceOfPerStageQueries?: boolean;
		endingOccurrenceOfPerStageQueries?: boolean;
	}

	interface GPUQuerySetDescriptor {
		type: 'occlusion' | 'timestamp' | 'resolve';
		count: number;
		label?: string;
	}

	interface GPURequestAdapterOptions {
		forceFallbackAdapter?: boolean;
		powerPreference?: 'low-power' | 'high-performance';
	}

	interface GPUAdapterInfo {
		vendor: string;
		device: string;
		architecture?: string;
		deviceType?: 'independent-gpu' | 'integrated-gpu' | 'cpu' | 'unknown';
	}

	interface GPURequestDeviceDescriptor {
		label?: string;
		requiredFeatures?: GPUFeatureName[];
		requiredLimits?: Record<string, number>;
		defaultQueue?: GPUQueueDescriptor;
	}

	type GPUFeatureName = 'depth-clip-control' | 'depth32float-stencil8' | 'timestamp-query' | 'texture-compression-bc' | 'texture-compression-etc2' | 'texture-compression-astc' | 'indirect-first-instance' | 'shader-f16' | 'rg11b10ufloat-render-format' | 'bgra8unorm-storage' | 'float32-filterable' | 'float32-blendable' | 'depth24plus' | 'depth24plus-stencil8' | 'texture-compression-astc-sliced-3d' | 'subgroups' | 'subgroups-extension' | 'core-features-and-features' | 'core-features-and-limits' | 'core-features-and-features-and-limits' | 'core-features-and-features-and-limits-and-extensions' | 'core-features-and-features-and-limits-and-extensions-and-internal';

	interface GPU {
		requestAdapter(options?: GPURequestAdapterOptions): Promise<GPUAdapter | null>;
		getPreferredCanvasFormat(): GPUTextureFormat;
	}

	interface GPUAdapter {
		requestDevice(descriptor?: GPURequestDeviceDescriptor): Promise<GPUDevice>;
		requestAdapterInfo(peekOnly?: boolean): Promise<GPUAdapterInfo>;
		readonly features: GPUFeatureSet;
		readonly limits: GPUSize64Dict;
	}

	interface GPUDevice extends EventTarget {
		readonly features: GPUFeatureSet;
		readonly limits: GPUSize64Dict;
		readonly queue: GPUQueue;
		createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer;
		createTexture(descriptor: GPUTextureDescriptor): GPUTexture;
		createShaderModule(descriptor: GPUShaderModuleDescriptor): GPUShaderModule;
		createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor): GPUBindGroupLayout;
		createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor): GPUPipelineLayout;
		createComputePipeline(descriptor: GPUComputePipelineDescriptor): GPUComputePipeline;
		createRenderPipeline(descriptor: GPURenderPipelineDescriptor): GPURenderPipeline;
		createBindGroup(descriptor: GPUBindGroupDescriptor): GPUBindGroup;
		createCommandEncoder(descriptor?: GPUCommandEncoderDescriptor): GPUCommandEncoder;
		createQuerySet(descriptor: GPUQuerySetDescriptor): GPUQuerySet;
		destroy(): void;
	}

	interface GPUBufferDescriptor {
		size: number;
		usage: number;
		mappedAtCreation?: boolean;
		label?: string;
	}

	interface GPUQueue {
		submit(commandBuffers: GPUCommandBuffer[]): void;
		writeBuffer(buffer: GPUBuffer, bufferOffset: number, data: ArrayBufferView, dataOffset?: number, size?: number): void;
	}

	interface GPUCommandEncoder {
		beginRenderPass(descriptor: GPURenderPassDescriptor): GPURenderPassEncoder;
		beginComputePass(descriptor?: GPUComputePassDescriptor): GPUComputePassEncoder;
		copyBufferToBuffer(
			source: GPUBuffer,
			sourceOffset: number,
			destination: GPUBuffer,
			destinationOffset: number,
			size: number
		): void;
		end(): GPUCommandBuffer;
		finish(): GPUCommandBuffer;
	}

	interface GPURenderPassEncoder {
		setPipeline(pipeline: GPURenderPipeline): void;
		setBindGroup(index: number, bindGroup: GPUBindGroup, dynamicOffsets?: Uint32Array[]): void;
		setVertexBuffer(index: number, buffer: GPUBuffer, offset?: number, size?: number): void;
		setIndexBuffer(buffer: GPUBuffer, indexFormat: GPUIndexFormat, offset?: number, size?: number): void;
		draw(
			vertexCount: number,
			instanceCount?: number,
			firstVertex?: number,
			firstInstance?: number
		): void;
		drawIndexed(
			indexCount: number,
			instanceCount?: number,
			firstIndex?: number,
			bias?: number,
			firstInstance?: number
		): void;
		end(): void;
	}

	interface GPUComputePassEncoder {
		setPipeline(pipeline: GPUComputePipeline): void;
		setBindGroup(index: number, bindGroup: GPUBindGroup, dynamicOffsets?: Uint32Array[]): void;
		dispatchWorkgroups(x: number, y?: number, z?: number): void;
		end(): void;
	}

	interface GPURenderPipeline {
		getBindGroupLayout(index: number): GPUBindGroupLayout;
	}

	interface GPUComputePipeline {
		getBindGroupLayout(index: number): GPUBindGroupLayout;
	}

	type GPUTextureFormat =
		| 'r8unorm'
		| 'r8snorm'
		| 'r8uint'
		| 'r8sint'
		| 'r16uint'
		| 'r16sint'
		| 'r16float'
		| 'rg8unorm'
		| 'rg8snorm'
		| 'rg8uint'
		| 'rg8sint'
		| 'r32uint'
		| 'r32sint'
		| 'r32float'
		| 'rg16uint'
		| 'rg16sint'
		| 'rg16float'
		| 'rgba8unorm'
		| 'rgba8unorm-srgb'
		| 'rgba8snorm'
		| 'rgba8uint'
		| 'rgba8sint'
		| 'bgra8unorm'
		| 'bgra8unorm-srgb'
		| 'rgb9e5ufloat'
		| 'rgb10a2unorm'
		| 'rg11b10ufloat'
		| 'single-channel'
		| 'three-channel'
		| 'four-channel'
		| 'rgb10a2uint'
		| 'r32uint'
		| 'r32sint'
		| 'r32float'
		| 'rg32uint'
		| 'rg32sint'
		| 'rg32float'
		| 'rgba32uint'
		| 'rgba32sint'
		| 'rgba32float'
		| 'stencil8'
		| 'depth16unorm'
		| 'depth24plus'
		| 'depth24plus-stencil8'
		| 'depth32float'
		| 'depth32float-stencil8'
		| 'bc1-rgba-unorm'
		| 'bc1-rgba-unorm-srgb'
		| 'bc2-rgba-unorm'
		| 'bc2-rgba-unorm-srgb'
		| 'bc3-rgba-unorm'
		| 'bc3-rgba-unorm-srgb'
		| 'bc4-r-unorm'
		| 'bc4-r-snorm'
		| 'bc5-rg-unorm'
		| 'bc5-rg-snorm'
		| 'bc6h-rgb-ufloat'
		| 'bc6h-rgb-float'
		| 'bc7-rgba-unorm'
		| 'bc7-rgba-unorm-srgb'
		| 'etc2-rgb8unorm'
		| 'etc2-rgb8unorm-srgb'
		| 'etc2-rgb8a1unorm'
		| 'etc2-rgb8a1unorm-srgb'
		| 'etc2-rgba8unorm'
		| 'etc2-rgba8unorm-srgb'
		| 'eac-r11unorm'
		| 'eac-r11snorm'
		| 'eac-rg11unorm'
		| 'eac-rg11snorm'
		| 'astc-4x4-unorm'
		| 'astc-4x4-unorm-srgb'
		| 'astc-5x4-unorm'
		| 'astc-5x4-unorm-srgb'
		| 'astc-5x5-unorm'
		| 'astc-5x5-unorm-srgb'
		| 'astc-6x5-unorm'
		| 'astc-6x5-unorm-srgb'
		| 'astc-6x6-unorm'
		| 'astc-6x6-unorm-srgb'
		| 'astc-8x5-unorm'
		| 'astc-8x5-unorm-srgb'
		| 'astc-8x6-unorm'
		| 'astc-8x6-unorm-srgb'
		| 'astc-8x7-unorm'
		| 'astc-8x7-unorm-srgb'
		| 'astc-8x8-unorm'
		| 'astc-8x8-unorm-srgb'
		| 'astc-10x5-unorm'
		| 'astc-10x5-unorm-srgb'
		| 'astc-10x6-unorm'
		| 'astc-10x6-unorm-srgb'
		| 'astc-10x7-unorm'
		| 'astc-10x7-unorm-srgb'
		| 'astc-10x8-unorm'
		| 'astc-10x8-unorm-srgb'
		| 'astc-10x10-unorm'
		| 'astc-10x10-unorm-srgb'
		| 'astc-12x10-unorm'
		| 'astc-12x10-unorm-srgb'
		| 'astc-12x12-unorm'
		| 'astc-12x12-unorm-srgb';

	interface GPUBlendComponent {
		operation?: 'add' | 'subtract' | 'reverse-subtract' | 'min' | 'max';
		srcFactor?:
			| 'zero'
			| 'one'
			| 'src'
			| 'one-minus-src'
			| 'src-alpha'
			| 'one-minus-src-alpha'
			| 'dst'
			| 'one-minus-dst'
			| 'dst-alpha'
			| 'one-minus-dst-alpha'
			| 'src-alpha-saturated'
			| 'constant'
			| 'one-minus-constant';
		dstFactor?:
			| 'zero'
			| 'one'
			| 'src'
			| 'one-minus-src'
			| 'src-alpha'
			| 'one-minus-src-alpha'
			| 'dst'
			| 'one-minus-dst'
			| 'dst-alpha'
			| 'one-minus-dst-alpha'
			| 'src-alpha-saturated'
			| 'constant'
			| 'one-minus-constant';
	}

	interface GPUBlendState {
		color: GPUBlendComponent;
		alpha: GPUBlendComponent;
	}

	type GPUIndexFormat = 'uint16' | 'uint32';

	interface GPUPrimitiveTopology {
		topology: 'point-list' | 'line-list' | 'line-strip' | 'triangle-list' | 'triangle-strip';
		stripIndexFormat?: GPUIndexFormat;
		frontFace?: 'cw' | 'ccw';
		cullMode?: 'none' | 'front' | 'back';
	}

	interface GPUVertexState {
		module: GPUShaderModule;
		entryPoint: string;
		buffers?: GPUVertexBufferLayout[];
	}

	interface GPUFragmentState {
		module: GPUShaderModule;
		entryPoint: string;
		targets: GPUColorTargetState[];
	}

	interface GPUColorTargetState {
		format: GPUTextureFormat;
		blend?: GPUBlendState;
		writeMask?: number;
	}

	interface GPUVertexBufferLayout {
		arrayStride: number;
		attributes: GPUVertexAttribute[];
		mode?: 'vertex' | 'instance';
	}

	interface GPUVertexAttribute {
		format:
			| 'float1'
			| 'float2'
			| 'float3'
			| 'float4'
			| 'uint1'
			| 'uint2'
			| 'uint3'
			| 'uint4'
			| 'sint1'
			| 'sint2'
			| 'sint3'
			| 'sint4'
			| 'float16x2'
			| 'float16x4';
		offset: number;
		shaderLocation: number;
	}

	interface GPUPipelineLayout {
		readonly label: string;
	}

	interface GPUBindGroupLayout {
		readonly label: string;
	}

	interface GPUShaderModule {
		readonly label: string;
	}

	interface GPUQuerySet {
		readonly label: string;
		destroy(): void;
	}

	interface GPUSize64Dict {
		[key: string]: number;
	}

	interface GPUFeatureSet {
		hasFeature(feature: GPUFeatureName): boolean;
		readonly size: number;
	}

	interface GPURenderPipelineDescriptor {
		layout: GPUPipelineLayout | 'auto';
		vertex: GPUVertexState;
		primitive?: GPUPrimitiveTopology;
		depthStencil?: GPUDepthStencilState;
		multisample?: GPUMultisampleState;
		fragment?: GPUFragmentState;
	}

	interface GPUDepthStencilState {
		format: GPUTextureFormat;
		depthWriteEnabled?: boolean;
		depthCompare?: 'never' | 'less' | 'less-equal' | 'greater' | 'greater-equal' | 'equal' | 'not-equal' | 'always';
		stencilFront?: GPUStencilFaceState;
		stencilBack?: GPUStencilFaceState;
		stencilReadMask?: number;
		stencilWriteMask?: number;
		depthBias?: number;
		depthBiasSlopeScale?: number;
		depthBiasClamp?: number;
	}

	interface GPUStencilFaceState {
		compare?: 'never' | 'less' | 'less-equal' | 'greater' | 'greater-equal' | 'equal' | 'not-equal' | 'always';
		failOp?: 'keep' | 'zero' | 'replace' | 'increment-clamp' | 'decrement-clamp' | 'invert';
		depthFailOp?: 'keep' | 'zero' | 'replace' | 'increment-clamp' | 'decrement-clamp' | 'invert';
		passOp?: 'keep' | 'zero' | 'replace' | 'increment-clamp' | 'decrement-clamp' | 'invert';
	}

	interface GPUMultisampleState {
		count?: number;
		mask?: number;
		alphaToCoverageEnabled?: boolean;
	}

	interface GPUSampler {
		readonly label: string;
		destroy(): void;
	}

	interface GPUSamplerDescriptor {
		addressModeU?: 'clamp-to-edge' | 'repeat' | 'mirrored-repeat';
		addressModeV?: 'clamp-to-edge' | 'repeat' | 'mirrored-repeat';
		addressModeW?: 'clamp-to-edge' | 'repeat' | 'mirrored-repeat';
		magFilter?: 'nearest' | 'linear';
		minFilter?: 'nearest' | 'linear';
		mipmapFilter?: 'nearest' | 'linear';
		lodMinClamp?: number;
		lodMaxClamp?: number;
		compare?: 'never' | 'less' | 'less-equal' | 'greater' | 'greater-equal' | 'equal' | 'not-equal' | 'always';
		maxAnisotropy?: number;
		label?: string;
	}

	interface GPUTexture {
		readonly label: string;
		createView(descriptor?: GPUTextureViewDescriptor): GPUTextureView;
		destroy(): void;
		readonly width: number;
		readonly height: number;
		readonly depthOrArrayLayers: number;
		readonly mipLevelCount: number;
		readonly sampleCount: number;
		readonly dimension: '1d' | '2d' | '3d';
		readonly format: GPUTextureFormat;
	}

	interface GPUTextureView {
		readonly label: string;
	}

	interface GPUCommandBuffer {
		readonly label: string;
	}

	interface GPUCanvasContext {
		getCurrentTexture(): GPUTexture;
		configure(config: {
			device: GPUDevice;
			format: GPUTextureFormat;
			usage?: number;
			viewFormats?: GPUTextureFormat[];
			alphaMode?: 'opaque' | 'premultiplied';
		}): void;
		unconfigure(): void;
	}

	interface HTMLCanvasElement {
		getContext(contextId: 'webgpu'): GPUCanvasContext | null;
	}

	interface Navigator {
		gpu: GPU;
	}
}

export {};
