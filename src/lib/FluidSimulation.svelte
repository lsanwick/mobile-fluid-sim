<script lang="ts">
	import { onMount, onDestroy } from 'svelte';

	import { setupFluidScene, FluidRenderer } from '$lib/fluid';
	import type { FlipFluid } from '$lib/fluid';

	let {
		gravity = { x: 0, y: -9.81 },
		resolution = 70,
		fluidColor = { r: 0.09, g: 0.4, b: 1.0 },
		foamColor = { r: 0.75, g: 0.9, b: 1.0 },
		colorDiffusionCoeff = 0.0008,
		foamReturnRate = 0.5,
		onclick
	}: {
		gravity?: { x: number; y: number };
		resolution?: number;
		angle?: number;
		fluidColor?: { r: number; g: number; b: number };
		foamColor?: { r: number; g: number; b: number };
		colorDiffusionCoeff?: number;
		foamReturnRate?: number;
		onclick?: () => void;
	} = $props();

	let canvas: HTMLCanvasElement;
	let fluid: FlipFluid;
	let renderer: FluidRenderer;
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
	const relWaterWidth = 0.6; // Water width as fraction of tank (0.1 to 1.0)
	const relWaterHeight = 0.8; // Water height as fraction of tank (0.1 to 1.0)

	// Interaction state
	let holdTimer: ReturnType<typeof setTimeout> | null = null;
	let holdInterval: ReturnType<typeof setInterval> | null = null;
	let isHolding = false;
	let holdPosition = { x: 0, y: 0 };

	const TAP_RADIUS = 0.15;
	const HOLD_THRESHOLD = 300;

	function resizeCanvas() {
		if (!canvas) return;

		const rect = canvas.getBoundingClientRect();
		const devicePixelRatio = window.devicePixelRatio || 1;

		canvas.width = rect.width * devicePixelRatio;
		canvas.height = rect.height * devicePixelRatio;

		// Update simulation dimensions to maintain aspect ratio
		const cScale = canvas.height / simHeight;
		simWidth = canvas.width / cScale;

		if (renderer) {
			renderer.resize(canvas.width, canvas.height);
		}
	}

	function simulate() {
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

	function screenToSim(screenX: number, screenY: number): { x: number; y: number } {
		if (!canvas) return { x: 0, y: 0 };
		const rect = canvas.getBoundingClientRect();
		const scaleX = simWidth / rect.width;
		const scaleY = simHeight / rect.height;
		const result = {
			x: (screenX - rect.left) * scaleX,
			y: simHeight - (screenY - rect.top) * scaleY
		};
		console.log('[FluidSim] screenToSim:', { screenX, screenY, rectLeft: rect.left, rectTop: rect.top, rectWidth: rect.width, rectHeight: rect.height, result });
		return result;
	}

	function startHoldAction(add: boolean) {
		if (!fluid) return;
		console.log('[FluidSim] startHoldAction add:', add, 'at:', holdPosition);
		holdInterval = setInterval(() => {
			if (!fluid) return;
			if (add) {
				console.log('[FluidSim] hold interval add at:', holdPosition);
				fluid.addParticlesAt(holdPosition.x, holdPosition.y, TAP_RADIUS * 0.5);
			} else {
				console.log('[FluidSim] hold interval remove at:', holdPosition);
				fluid.removeParticlesNear(holdPosition.x, holdPosition.y, TAP_RADIUS * 0.5);
			}
		}, 30);
	}

	function stopHoldAction() {
		console.log('[FluidSim] stopHoldAction');
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
				fluid.addParticlesAt(simPos.x, simPos.y, TAP_RADIUS * 0.5);
			} else {
				fluid.removeParticlesNear(simPos.x, simPos.y, TAP_RADIUS * 0.5);
			}
			startHoldAction(adding);
		}, HOLD_THRESHOLD);
	}

	function handlePointerMove(e: PointerEvent) {
		if (!fluid) return;
		const simPos = screenToSim(e.clientX, e.clientY);
		holdPosition = simPos;
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

	function render() {
		if (!fluid || !renderer) return;

		renderer.render(fluid, {
			showParticles,
			showGrid,
			simWidth,
			simHeight
		});
	}

	function update() {
		simulate();
		render();
		animationId = requestAnimationFrame(update);
	}

	onMount(() => {
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
		renderer = new FluidRenderer(canvas);

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
		};
	});

	// Watch for color changes and update fluid (supports live changes later)
	$effect(() => {
		if (fluid) {
			fluid.setFluidColor(fluidColor);
		}
	});

	// Watch for foam color changes
	$effect(() => {
		if (fluid) {
			fluid.setFoamColor(foamColor);
		}
	});

	// Watch for diffusion coefficient changes
	$effect(() => {
		if (fluid) {
			fluid.setColorDiffusionCoeff(colorDiffusionCoeff);
		}
	});

	// Watch for foam return rate changes
	$effect(() => {
		if (fluid) {
			fluid.setFoamReturnRate(foamReturnRate);
		}
	});
</script>

<canvas bind:this={canvas} class="absolute inset-0 z-10 h-full w-full" style="touch-action: none;"></canvas>
