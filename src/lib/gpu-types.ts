// GPU constant objects (used as values, not types)
export const GPUBufferUsage = {
	COPY_SRC: 0x0001,
	COPY_DST: 0x0002,
	INDEX: 0x0004,
	VERTEX: 0x0008,
	UNIFORM: 0x0010,
	STORAGE: 0x0020,
	INDIRECT: 0x0040,
	QUERY_RESOLVE: 0x0080,
	MAP_READ: 0x1000,
	MAP_WRITE: 0x2000,
} as const;

export const GPUTextureUsage = {
	TRANSFER_SRC: 0x0001,
	TRANSFER_DST: 0x0002,
	SAMPLED: 0x0004,
	STORE: 0x0008,
	OUTPUT_ATTACHMENT: 0x0010,
	READ_ONLY_BINDING: 0x0020,
	COPY_SRC: 0x0040,
	COPY_DST: 0x0080,
} as const;

export const GPUShaderStage = {
	VERTEX: 0x1,
	FRAGMENT: 0x2,
	COMPUTE: 0x4,
} as const;

export const GPUMapMode = {
	READ: 0x1,
	WRITE: 0x2,
} as const;
