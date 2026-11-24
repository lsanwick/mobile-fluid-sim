<script lang="ts">
	import { onMount, onDestroy } from 'svelte';

	import {
		setupFluidScene,
		FluidRenderer,
		FluidRendererWebGPU,
		WebGPUSimulation,
		type RenderConfig
	} from '$lib/fluid';
	import type { FlipFluid } from '$lib/fluid';

	let {
		gravity = { x: 0, y: -9.81 },
		resolution = 70,
		fluidColor = { r: 0.09, g: 0.4, b: 1.0 },
		foamColor = { r: 0.75, g: 0.9, b: 1.0 },
		colorDiffusionCoeff = 0.0008,
		foamReturnRate = 0.5,
		onclick,
		useWebGPU: propUseWebGPU = false
	}: {
		gravity?: { x: number; y: number };
		resolution?: number;
		angle?: number;
		fluidColor?: { r: number; g: number; b: number };
		foamColor?: { r: number; g: number; b: number };
		colorDiffusionCoeff?: number;
		foamReturnRate?: number;
		onclick?: () => void;
		useWebGPU?: boolean;
	} = $props();

	let canvas: HTMLCanvasElement;
	let fluid: FlipFluid;

	// WebGL pipeline (CPU simulation + WebGL render)
	let webglRenderer: FluidRenderer | null = null;

	// WebGPU pipeline (GPU simulation + WebGPU render)
	let gpuDevice: GPUDevice | null = null;
	let gpuSimulation: WebGPUSimulation | null = null;
	let gpuRenderer: FluidRendererWebGPU | null = null;
	let gpuInitialized = $state(false);

	// Animation
	let animationId: number;

	let simHeight = 3.0;
	let simWidth = 4.0;

	const dt = 1.0 / 120.0;
	const flipRatio = 0.95;
	const numPressureIters = 60;
	const numParticleIters = 3;
	const overRelaxation = 1.7;
	const compensateDrift = true;
	const separateParticles = true;
	const showParticles = true;
	const showGrid = false;
	const damping = 1.0;

	// Particle count controls
	const relWaterWidth = 0.6;
	const relWaterHeight = 0.8;

	// Interaction state
	let holdTimer: ReturnType<typeof setTimeout> | null = null;
	let holdInterval: ReturnType<typeof setInterval> | null = null;
	let isHolding = false;
	let holdPosition = { x: 0, y: 0 };

	// FPS tracking
	let frameCount = 0;
	let lastFpsUpdate = 0;
	let fps = $state(0);
	let fpsSmooth = $state(0);

	const TAP_RADIUS = 0.15;
	const HOLD_THRESHOLD = 300;

	// WebGPU toggle state
	let webgpuEnabled = $state(propUseWebGPU);
	let webgpuSupported = $state(false);
	let webgpuAvailable = $state(false);

	// Check WebGPU support
	$effect(() => {
		if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
			webgpuAvailable = true;
		}
	});

	// Sync internal state with prop
	$effect(() => {
		webgpuEnabled = propUseWebGPU;
	});

	async function initWebGPU() {
		if (!canvas || !webgpuAvailable) return;

		try {
			const adapter = await navigator.gpu.requestAdapter();
			if (!adapter) {
				console.warn('WebGPU: No adapter found');
				webgpuSupported = false;
				return;
			}

			gpuDevice = await adapter.requestDevice();
			if (!gpuDevice) {
				console.warn('WebGPU: Failed to request device');
				webgpuSupported = false;
				return;
			}

			gpuRenderer = new FluidRendererWebGPU(canvas, gpuDevice);
			gpuSimulation = new WebGPUSimulation(gpuDevice);

			gpuInitialized = true;
			webgpuSupported = true;
			console.log('WebGPU pipeline initialized');
		} catch (err) {
			console.warn('WebGPU initialization failed, falling back to WebGL:', err);
			webgpuSupported = false;
			gpuInitialized = false;
		}
	}

	function initWebGL() {
		webglRenderer = new FluidRenderer(canvas);
		gpuInitialized = false;
	}

	function resizeCanvas() {
		if (!canvas) return;

		const rect = canvas.getBoundingClientRect();
		const devicePixelRatio = window.devicePixelRatio || 1;

		canvas.width = rect.width * devicePixelRatio;
		canvas.height = rect.height * devicePixelRatio;

		// Update simulation dimensions to maintain aspect ratio
		const cScale = canvas.height / simHeight;
		simWidth = canvas.width / cScale;

		if (webglRenderer) {
			webglRenderer.resize(canvas.width, canvas.height);
		}
		if (gpuRenderer) {
			gpuRenderer.resize(canvas.width, canvas.height);
		}
	}

	function simulateCPU() {
		if (!fluid) return;

		fluid.simulate(
			dt,
			gravity.x,
			gravity.y,
			flipRatio,
			numPressureIters,
			numParticleIters,
			overRelaxation,
			compensateDrift,
			separateParticles,
			damping
		);
	}

	function simulateGPU() {
		if (!fluid || !gpuSimulation) return;

		gpuSimulation.simulate(
			fluid,
			dt,
			gravity.x,
			gravity.y,
			flipRatio,
			numPressureIters,
			numParticleIters,
			overRelaxation,
			compensateDrift,
			separateParticles,
			damping
		);
	}

	function screenToSim(screenX: number, screenY: number): { x: number; y: number } {
		if (!canvas) return { x: 0, y: 0 };
		const rect = canvas.getBoundingClientRect();
		const scaleX = simWidth / rect.width;
		const scaleY = simHeight / rect.height;
		return {
			x: (screenX - rect.left) * scaleX,
			y: simHeight - (screenY - rect.top) * scaleY
		};
	}

	function startHoldAction(add: boolean) {
		if (!fluid) return;
		holdInterval = setInterval(() => {
			if (!fluid) return;
			if (add) {
				fluid.addParticlesAt(holdPosition.x, holdPosition.y, TAP_RADIUS * 0.5);
			} else {
				fluid.removeParticlesNear(holdPosition.x, holdPosition.y, TAP_RADIUS * 0.5);
			}
		}, 30);
	}

	function stopHoldAction() {
		if (holdInterval) {
			clearInterval(holdInterval);
			holdInterval = null;
		}
	}

	function handlePointerDown(e: PointerEvent) {
		// Don't interfere with the toggle checkbox
		if ((e.target as HTMLElement).closest('.gpu-toggle')) return;

		if (!fluid) return;
		e.preventDefault();
		isHolding = false;

		const simPos = screenToSim(e.clientX, e.clientY);
		holdPosition = { x: simPos.x, y: simPos.y };

		const hasFluid = fluid.hasParticlesNear(simPos.x, simPos.y, TAP_RADIUS);
		const adding = !hasFluid;

		holdTimer = setTimeout(() => {
			isHolding = true;
			if (adding) {
				fluid.addParticlesAt(simPos.x, simPos.y, TAP_RADIUS * 0.5);
			} else {
				fluid.removeParticlesNear(simPos.x, simPos.y, TAP_RADIUS * 0.7);
			}
			startHoldAction(adding);
		}, HOLD_THRESHOLD);
	}

	function handlePointerMove(e: PointerEvent) {
		if (!fluid || !isHolding) return;
		const simPos = screenToSim(e.clientX, e.clientY);
		holdPosition = { x: simPos.x, y: simPos.y };
	}

	function handlePointerUp(e: PointerEvent) {
		if (!fluid) return;
		e.preventDefault();

		stopHoldAction();

		if (holdTimer) {
			clearTimeout(holdTimer);
			holdTimer = null;
		}

		isHolding = false;
	}

	function renderCPU() {
		if (!fluid || !webglRenderer) return;

		const config: RenderConfig = {
			showParticles,
			showGrid,
			simWidth,
			simHeight
		};

		webglRenderer.render(fluid, config);
	}

	function renderGPU() {
		if (!fluid || !gpuRenderer || !gpuSimulation) return;

		const config: RenderConfig = {
			showParticles,
			showGrid,
			simWidth,
			simHeight
		};

		gpuRenderer.setSimulationBuffers({
			particlePosBuffer: gpuSimulation['particlePosBuffer'],
			particleColorBuffer: gpuSimulation['particleColorBuffer'],
			cellColorBuffer: gpuSimulation['cellColorBuffer'],
			fNumX: gpuSimulation['fNumX'],
			fNumY: gpuSimulation['fNumY'],
			fNumCells: gpuSimulation['fNumCells'],
			h: gpuSimulation['h'],
			numParticles: gpuSimulation['numParticles'],
			maxParticles: gpuSimulation['particlePosBuffer'].size / 8
		});

		gpuRenderer.render(fluid, config);
	}

	function update() {
		const now = performance.now();
		frameCount++;

		if (lastFpsUpdate === 0) {
			lastFpsUpdate = now;
		} else {
			const elapsed = now - lastFpsUpdate;
			if (elapsed >= 500) {
				fps = Math.round((frameCount * 1000) / elapsed);
				fpsSmooth = Math.round(fpsSmooth * 0.7 + fps * 0.3);
				frameCount = 0;
				lastFpsUpdate = now;
			}
		}

		if (webgpuEnabled && gpuInitialized) {
			simulateGPU();
			renderGPU();
		} else {
			simulateCPU();
			renderCPU();
		}
		animationId = requestAnimationFrame(update);
	}

	async function onMountInit() {
		resizeCanvas();

		// Initialize fluid simulation
		fluid = setupFluidScene(
			simWidth,
			simHeight,
			resolution,
			relWaterWidth,
			relWaterHeight,
			fluidColor,
			foamColor,
			colorDiffusionCoeff,
			foamReturnRate
		);

		// Initialize renderers
		initWebGL();

		// Initialize WebGPU if requested
		if (webgpuEnabled) {
			await initWebGPU();
			if (!gpuInitialized) {
				// Fallback to WebGL if WebGPU failed
				webgpuEnabled = false;
			}
		}

		// Initial color is already set via constructor, keep setter for consistency
		if (fluid) {
			fluid.setFluidColor(fluidColor);
			fluid.setFoamColor(foamColor);
			fluid.setColorDiffusionCoeff(colorDiffusionCoeff);
			fluid.setFoamReturnRate(foamReturnRate);
		}

		// Handle window resize
		const handleResize = () => {
			resizeCanvas();
		};
		window.addEventListener('resize', handleResize);

		// Touch/click interaction handlers
		document.addEventListener('pointerdown', handlePointerDown, { passive: false });
		document.addEventListener('pointermove', handlePointerMove);
		document.addEventListener('pointerup', handlePointerUp, { passive: false });
		document.addEventListener('pointercancel', handlePointerUp, { passive: false });

		// Start animation loop
		update();

		return () => {
			window.removeEventListener('resize', handleResize);
			if (animationId) {
				cancelAnimationFrame(animationId);
			}
			stopHoldAction();
			if (holdTimer) {
				clearTimeout(holdTimer);
			}
			document.removeEventListener('pointerdown', handlePointerDown);
			document.removeEventListener('pointermove', handlePointerMove);
			document.removeEventListener('pointerup', handlePointerUp);
			document.removeEventListener('pointercancel', handlePointerUp);

			// Cleanup WebGL
			webglRenderer = null;

			// Cleanup WebGPU resources
			if (gpuRenderer) {
				gpuRenderer.destroy();
				gpuRenderer = null;
			}
			if (gpuSimulation) {
				gpuSimulation.destroy();
				gpuSimulation = null;
			}
			if (gpuDevice) {
				gpuDevice.destroy();
				gpuDevice = null;
			}
		};
	}

	onMount(onMountInit);

	// Watch for color changes and update fluid
	$effect(() => {
		if (fluid) {
			fluid.setFluidColor(fluidColor);
		}
	});

	$effect(() => {
		if (fluid) {
			fluid.setFoamColor(foamColor);
		}
	});

	$effect(() => {
		if (fluid) {
			fluid.setColorDiffusionCoeff(colorDiffusionCoeff);
		}
	});

	$effect(() => {
		if (fluid) {
			fluid.setFoamReturnRate(foamReturnRate);
		}
	});

	function toggleWebGPU() {
		if (!webgpuAvailable) return;

		if (webgpuEnabled) {
			// Switching from WebGPU to WebGL
			if (gpuRenderer) {
				gpuRenderer.destroy();
				gpuRenderer = null;
			}
			if (gpuSimulation) {
				gpuSimulation.destroy();
				gpuSimulation = null;
			}
			if (gpuDevice) {
				gpuDevice.destroy();
				gpuDevice = null;
			}
			gpuInitialized = false;
			webgpuEnabled = false;
			console.log('Switched to WebGL pipeline');
		} else {
			// Switching from WebGL to WebGPU
			if (webgpuSupported) {
				initWebGPU();
				webgpuEnabled = true;
				console.log('Switched to WebGPU pipeline');
			} else {
				console.warn('WebGPU not supported on this device');
			}
		}
	}
</script>

<div class="relative h-full w-full">
	<!-- WebGPU Toggle + FPS - Top Left -->
	{#if webgpuAvailable}
		<div class="gpu-toggle absolute left-2 top-2 z-30 flex flex-col gap-1">
			<label
				class="flex cursor-pointer items-center gap-1.5 rounded-md bg-black/70 px-2.5 py-1.5 text-xs font-medium text-gray-200 backdrop-blur-sm transition-colors hover:bg-black/85"
				onclick={(e) => {
					e.preventDefault();
					e.stopPropagation();
					toggleWebGPU();
				}}
			>
				<input
					type="checkbox"
					checked={webgpuEnabled}
					class="h-3.5 w-3.5 cursor-pointer rounded border-gray-500 bg-transparent text-blue-500 focus:ring-1 focus:ring-blue-500"
					onclick={(e) => e.stopPropagation()}
					onchange={(e) => {
						e.stopPropagation();
						e.preventDefault();
						toggleWebGPU();
					}}
					disabled={!webgpuAvailable}
				/>
				<svg
					xmlns="http://www.w3.org/2000/svg"
					width="14"
					height="14"
					viewBox="0 0 24 24"
					fill="none"
					stroke="currentColor"
					stroke-width="2"
					stroke-linecap="round"
					stroke-linejoin="round"
					class="text-blue-400"
				>
					<polyline points="16 18 22 12 16 6" />
					<polyline points="8 6 2 12 8 18" />
				</svg>
				<span class="select-none">WebGPU</span>
			</label>
			<div class="rounded-md bg-black/70 px-2 py-1 text-xs font-mono font-medium text-green-400 backdrop-blur-sm">
				{fpsSmooth} FPS
			</div>
		</div>
	{:else}
		<!-- FPS only (no WebGPU available) -->
		<div class="pointer-events-none absolute left-2 top-2 z-30">
			<div class="rounded-md bg-black/70 px-2 py-1 text-xs font-mono font-medium text-green-400 backdrop-blur-sm">
				{fpsSmooth} FPS
			</div>
		</div>
	{/if}

	<canvas bind:this={canvas} class="absolute inset-0 z-10 h-full w-full" style="touch-action: none;"></canvas>
</div>
