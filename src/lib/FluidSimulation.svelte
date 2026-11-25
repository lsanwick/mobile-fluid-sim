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
		damping = 1.0,
		overRelaxation = 1.7,
		showParticles = true,
		showGrid = false,
		particleSize = 0.03,
		onclick
	}: {
		gravity?: { x: number; y: number };
		resolution?: number;
		angle?: number;
		fluidColor?: { r: number; g: number; b: number };
		foamColor?: { r: number; g: number; b: number };
		colorDiffusionCoeff?: number;
		foamReturnRate?: number;
		damping?: number;
		overRelaxation?: number;
		showParticles?: boolean;
		showGrid?: boolean;
		particleSize?: number;
		onclick?: () => void;
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
	const compensateDrift = true;
	const separateParticles = true;

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

	// WebGPU availability check
	let webgpuAvailable = $state(false);

	$effect(() => {
		if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
			webgpuAvailable = true;
		}
	});

	async function initWebGPU() {
		if (!canvas || !webgpuAvailable) return false;

		try {
			const adapter = await navigator.gpu.requestAdapter();
			if (!adapter) {
				console.warn('WebGPU: No adapter found');
				return false;
			}

			gpuDevice = await adapter.requestDevice();
			gpuDevice.lost.then((info) => console.warn('WebGPU device lost:', info.message));
			gpuDevice.addEventListener('uncapturederror', (e: Event) =>
				console.error('WebGPU error:', (e as GPUUncapturedErrorEvent).error.message)
			);

			// Catch shader/pipeline validation failures so we can fall back to WebGL
			gpuDevice.pushErrorScope('validation');
			gpuRenderer = new FluidRendererWebGPU(canvas, gpuDevice);
			gpuSimulation = new WebGPUSimulation(gpuDevice);
			gpuSimulation.initialize(fluid);
			const error = await gpuDevice.popErrorScope();
			if (error) throw new Error(error.message);

			gpuInitialized = true;
			console.log('WebGPU pipeline initialized');
			return true;
		} catch (err) {
			console.warn('WebGPU initialization failed, falling back to WebGL:', err);
			gpuRenderer?.destroy();
			gpuSimulation?.destroy();
			gpuDevice?.destroy();
			gpuRenderer = null;
			gpuSimulation = null;
			gpuDevice = null;
			return false;
		}
	}

	function initWebGL() {
		// A canvas that already handed out a 'webgpu' context can't provide 'webgl',
		// so swap in a fresh one if the WebGPU attempt got that far.
		if (canvas.getContext('webgl') === null) {
			const fresh = canvas.cloneNode() as HTMLCanvasElement;
			canvas.replaceWith(fresh);
			canvas = fresh;
			resizeCanvas();
		}
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

	function simulatePush() {
		if (!fluid) return;
		fluid.pushAt(holdPosition.x, holdPosition.y, 0.3, 2.0);
	}

	function simulateGrab() {
		if (!fluid) return;
		fluid.grabAt(holdPosition.x, holdPosition.y, 0.2);
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
			damping,
			showGrid
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

	// Particle edits happen on the CPU copy (kept fresh by the GPU readback)
	// and are then pushed to the GPU when that pipeline is active.
	function addParticles(x: number, y: number, radius: number) {
		const before = fluid.numParticles;
		fluid.addParticlesAt(x, y, radius);
		if (gpuSimulation && fluid.numParticles !== before) {
			gpuSimulation.syncParticles(fluid, before);
		}
	}

	function removeParticles(x: number, y: number, radius: number) {
		const removed = fluid.removeParticlesNear(x, y, radius);
		if (gpuSimulation && removed > 0) {
			gpuSimulation.syncParticles(fluid, 0);
		}
	}

	function startHoldAction(add: boolean) {
		if (!fluid) return;
		holdInterval = setInterval(() => {
			if (!fluid) return;
			if (add) {
				addParticles(holdPosition.x, holdPosition.y, TAP_RADIUS * 0.5);
			} else {
				removeParticles(holdPosition.x, holdPosition.y, TAP_RADIUS * 0.5);
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
				addParticles(simPos.x, simPos.y, TAP_RADIUS * 0.5);
			} else {
				removeParticles(simPos.x, simPos.y, TAP_RADIUS * 0.7);
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
			particlePosBuffer: gpuSimulation.particlePosBuffer,
			particleColorBuffer: gpuSimulation.particleColorBuffer,
			cellColorBuffer: gpuSimulation.gridBuffer,
			cellColorOffset: gpuSimulation.cellColorOffset,
			fNumX: gpuSimulation.fNumX,
			fNumY: gpuSimulation.fNumY,
			fNumCells: gpuSimulation.fNumCells,
			h: gpuSimulation.h,
			numParticles: gpuSimulation.numParticles
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

		try {
			if (gpuInitialized) {
				simulateGPU();
				renderGPU();
			} else {
				simulateCPU();
				renderCPU();
			}
		} catch (err) {
			console.error('update error:', err);
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

		// Try WebGPU first, fall back to WebGL
		const success = await initWebGPU();
		if (!success) {
			initWebGL();
			console.log('Running with WebGL fallback');
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
			gpuSimulation?.syncColors(fluid);
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


</script>

<div class="relative h-full w-full">
	<!-- FPS Counter - Top Left -->
	<div class="pointer-events-none absolute left-2 top-2 z-30">
		<div class="rounded-md bg-black/70 px-2 py-1 text-xs font-mono font-medium text-green-400 backdrop-blur-sm">
			{fpsSmooth} FPS
		</div>
	</div>

	<canvas bind:this={canvas} class="absolute inset-0 z-10 h-full w-full" style="touch-action: none;"></canvas>
</div>
