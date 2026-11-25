<script lang="ts">
	import Loader2 from '@lucide/svelte/icons/loader-2';
	import Smartphone from '@lucide/svelte/icons/smartphone';

	import { onMount, onDestroy } from 'svelte';
	import { Tween } from 'svelte/motion';
	import { cubicOut } from 'svelte/easing';

	import { browser } from '$app/environment';
	import FluidSimulation from '$lib/FluidSimulation.svelte';
	import OverflowMenu from '$lib/OverflowMenu.svelte';
	import PopupInfo from '$lib/PopupInfo.svelte';

	const MAX_GRAVITY = 9.81;

	type AppState =
		| 'loading'
		| 'needs-permission'
		| 'ready'
		| 'denied'
		| 'no-motion'
		| 'not-supported';

	const fluidTypes = [
		{
			fluidColor: { r: 0.09, g: 0.4, b: 1.0 },
			foamColor: { r: 0.75, g: 0.9, b: 1.0 },
			colorDiffusionCoeff: 0.0008,
			foamReturnRate: 0.5
		},
		{
			fluidColor: { r: 0.0, g: 0.7, b: 0.8 },
			foamColor: { r: 0.6, g: 0.95, b: 0.9 },
			colorDiffusionCoeff: 0.0012,
			foamReturnRate: 0.6
		},
		{
			fluidColor: { r: 1.0, g: 0.4, b: 0.1 },
			foamColor: { r: 1.0, g: 0.8, b: 0.6 },
			colorDiffusionCoeff: 0.0004,
			foamReturnRate: 0.3
		},
		{
			fluidColor: { r: 0.5, g: 0.2, b: 0.9 },
			foamColor: { r: 0.8, g: 0.7, b: 1.0 },
			colorDiffusionCoeff: 0.001,
			foamReturnRate: 0.7
		},
		{
			fluidColor: { r: 0.1, g: 0.6, b: 0.4 },
			foamColor: { r: 0.7, g: 0.95, b: 0.8 },
			colorDiffusionCoeff: 0.0015,
			foamReturnRate: 0.4
		},
		{
			fluidColor: { r: 0.9, g: 0.5, b: 0.6 },
			foamColor: { r: 1.0, g: 0.85, b: 0.9 },
			colorDiffusionCoeff: 0.0006,
			foamReturnRate: 0.8
		},
		{
			fluidColor: { r: 0.3, g: 0.7, b: 0.9 },
			foamColor: { r: 0.9, g: 0.95, b: 1.0 },
			colorDiffusionCoeff: 0.0009,
			foamReturnRate: 0.5
		},
		{
			fluidColor: { r: 0.9, g: 0.7, b: 0.2 },
			foamColor: { r: 1.0, g: 0.9, b: 0.7 },
			colorDiffusionCoeff: 0.0005,
			foamReturnRate: 0.2
		}
	];

	let currentFluidIndex = $state(0);

	let angle: number | undefined = $state(0);
	let gravity = $state({ x: 0, y: -MAX_GRAVITY });
	let gravityStrength = $state(MAX_GRAVITY);
	let fluidColor = new Tween(fluidTypes[0].fluidColor, {
		duration: 500,
		easing: cubicOut
	});
	let foamColor = new Tween(fluidTypes[0].foamColor, {
		duration: 500,
		easing: cubicOut
	});
	let resolution = $state(70);
	let colorDiffusionCoeff: number = $state(fluidTypes[0].colorDiffusionCoeff);
	let foamReturnRate: number = $state(fluidTypes[0].foamReturnRate);
	let damping = $state(1.0);
	let overRelaxation = $state(1.7);
	let showParticles = $state(true);
	let showGrid = $state(false);
	let particleSize = $state(0.03);

	let appState: AppState = $state('loading');

	// Shake detection
	let lastShakeTime = 0;
	let lastAcceleration = { x: 0, y: 0, z: 0 };
	let shakeThreshold = 15;
	let shakeTimeThreshold = 600;

	const requestPermission = async () => {
		console.log('requestPermission called, browser=', browser);
		if (!browser) {
			console.log('returning early: not browser');
			return;
		}

		// Debug: show we're entering the function
		let dbg = document.getElementById('__page-debug');
		if (!dbg) {
			dbg = document.createElement('div');
			dbg.id = '__page-debug';
			dbg.style.cssText = 'position:fixed;top:40px;left:10px;background:#000;color:#f0f;font:11px monospace;padding:8px;z-index:9999;white-space:pre';
			document.body.appendChild(dbg);
		}
		dbg.textContent = 'requestPermission called\n';

		if (
			'DeviceOrientationEvent' in window &&
			typeof (DeviceOrientationEvent as any).requestPermission === 'function'
		) {
			dbg.textContent += 'iOS permission path\n';
			// iOS 13+ permission request
			try {
				// Start both requests synchronously within the tap: awaiting the first
				// can consume the user activation and make iOS reject the second.
				dbg.textContent += 'requesting orientation permission...\n';
				const orientationPromise = (DeviceOrientationEvent as any).requestPermission();
				dbg.textContent += 'orientation promise created, requesting motion permission...\n';
				const motionPromise =
					'DeviceMotionEvent' in window &&
					typeof (DeviceMotionEvent as any).requestPermission === 'function'
						? (DeviceMotionEvent as any).requestPermission()
						: Promise.resolve('granted');
				dbg.textContent += 'motion promise created, waiting...\n';
				const [orientationResponse, motionResponse] = await Promise.all([
					orientationPromise,
					motionPromise
				]);
				dbg.textContent += 'orientation=' + orientationResponse + ' motion=' + motionResponse + '\n';

				if (orientationResponse === 'granted' && motionResponse === 'granted') {
					dbg.textContent += 'GRANTED, setting ready\n';
					startListening();
					appState = 'ready';
				} else {
					dbg.textContent += 'DENIED\n';
					appState = 'denied';
				}
			} catch (error) {
				dbg.textContent += 'ERROR: ' + error + '\n';
				console.error('Error requesting device motion/orientation permission:', error);
				appState = 'denied';
			}
		} else if ('DeviceOrientationEvent' in window) {
			dbg.textContent += 'non-iOS path\n';
			startListening();
			appState = 'ready';
		} else {
			dbg.textContent += 'not supported\n';
			appState = 'not-supported';
		}
	};

	const startListening = () => {
		if (!browser) return;
		window.addEventListener('deviceorientation', onOrientationChange);
		window.addEventListener('devicemotion', onDeviceMotion);
	};

	const onDeviceMotion = (event: DeviceMotionEvent) => {
		if (!event.accelerationIncludingGravity) return;

		const acceleration = event.accelerationIncludingGravity;
		const x = acceleration.x || 0;
		const y = acceleration.y || 0;
		const z = acceleration.z || 0;

		// Calculate the magnitude of acceleration change
		const deltaX = Math.abs(x - lastAcceleration.x);
		const deltaY = Math.abs(y - lastAcceleration.y);
		const deltaZ = Math.abs(z - lastAcceleration.z);

		const totalDelta = deltaX + deltaY + deltaZ;
		const currentTime = Date.now();

		// Check if shake threshold is exceeded and enough time has passed
		if (totalDelta > shakeThreshold && currentTime - lastShakeTime > shakeTimeThreshold) {
			onShake();
			lastShakeTime = currentTime;
		}

		// Update last acceleration values
		lastAcceleration = { x, y, z };
	};

	const onOrientationChange = (event: DeviceOrientationEvent) => {
		if (event.beta !== null && event.gamma !== null) {
			const beta = event.beta;
			const gamma = event.gamma;

			const betaRad = beta * (Math.PI / 180);
			const gammaRad = gamma * (Math.PI / 180);

			const cosBeta = Math.cos(betaRad);
			const sinBeta = Math.sin(betaRad);
			const sinGamma = Math.sin(gammaRad);

			const gx = sinGamma * cosBeta;
			const gy = -sinBeta;

			gravity.x = gravityStrength * Math.max(-1, Math.min(1, gx));
			gravity.y = gravityStrength * Math.max(-1, Math.min(1, gy));
		}
	};

	onMount(async () => {
		if (!browser) return;

		if (!('DeviceOrientationEvent' in window)) {
			gravity = { x: 0, y: -gravityStrength };
			angle = 0;
			appState = 'not-supported';
			return;
		}

		// iOS 13+ only delivers motion events after requestPermission() is called
		// from a user gesture, so show the "Enable Motion Sensors" button there.
		if (typeof (DeviceOrientationEvent as any).requestPermission === 'function') {
			appState = 'needs-permission';
			return;
		}

		appState = 'ready';
		startListening();
	});

	onDestroy(() => {
		if (browser && window) {
			window.removeEventListener('deviceorientation', onOrientationChange);
			window.removeEventListener('devicemotion', onDeviceMotion);
		}
	});

	const onShake = () => {
		currentFluidIndex = (currentFluidIndex + 1) % fluidTypes.length;
		const newFluid = fluidTypes[currentFluidIndex];

		// Tween only the colors
		fluidColor.target = newFluid.fluidColor;
		foamColor.target = newFluid.foamColor;

		// Update other properties immediately
		colorDiffusionCoeff = newFluid.colorDiffusionCoeff;
		foamReturnRate = newFluid.foamReturnRate;
	};
</script>

<div
	class="relative flex h-screen flex-col items-center justify-center bg-gradient-to-b from-gray-900 via-gray-800 to-gray-950"
>
	<OverflowMenu
		bind:gravity
		bind:resolution
		bind:fluidColor
		bind:foamColor
		bind:colorDiffusionCoeff
		bind:foamReturnRate
		bind:damping
		bind:overRelaxation
		bind:showParticles
		bind:showGrid
		bind:particleSize
		bind:gravityStrength
	/>
	{#if appState === 'loading'}
		<div class="text-center">
			<Loader2 class="mx-auto mb-4 h-12 w-12 animate-spin text-blue-400" />
			<h1 class="mb-2 text-2xl font-bold text-white">Fluid Simulation</h1>
			<p class="text-gray-300">Initializing...</p>
		</div>
	{:else if appState === 'needs-permission'}
		<div class="text-center">
			<Smartphone class="mx-auto mb-6 h-16 w-16 text-blue-400" />
			<h1 class="mb-4 text-2xl font-bold text-white">Motion Sensors Required</h1>
			<p class="mb-6 max-w-sm text-gray-300">
				This fluid simulation responds to your device's tilt and orientation. It also detects shake
				gestures to change fluid colors! Please grant permission to access motion sensors for the
				best experience.
			</p>
			<button
				onclick={() => { console.log('test click direct'); appState = 'ready'; }}
				class="rounded-lg bg-green-600 px-8 py-3 font-semibold text-white shadow-lg transition-colors hover:bg-green-700"
			>
				Test: Skip to Ready
			</button>
			<button
				onclick={requestPermission}
				class="rounded-lg bg-blue-500 px-8 py-3 font-semibold text-white shadow-lg transition-colors hover:bg-blue-600 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900 focus:outline-none"
			>
				Enable Motion Sensors
			</button>
		</div>
	{:else if appState === 'denied'}
		<div class="text-center">
			<div class="mb-6 text-6xl">🚫</div>
			<h2 class="mb-4 text-xl font-semibold text-white">Permission Denied</h2>
			<p class="max-w-sm text-center text-gray-300">
				Motion sensor access was denied. You can still use the simulation, but it won't respond to
				device tilt. To enable this feature, please allow motion access in your browser settings.
			</p>
			<button
				onclick={requestPermission}
				class="mt-4 rounded-lg bg-gray-600 px-6 py-2 text-white transition-colors hover:bg-gray-500"
			>
				Try Again
			</button>
			<button
				onclick={() => (appState = 'no-motion')}
				class="mt-4 ml-2 rounded-lg bg-blue-500 px-6 py-2 font-semibold text-white transition-colors hover:bg-blue-600"
			>
				Continue Without Tilt
			</button>
		</div>
	{:else}
		<FluidSimulation
			{gravity}
			{resolution}
			fluidColor={fluidColor.current}
			foamColor={foamColor.current}
			{colorDiffusionCoeff}
			{foamReturnRate}
			{damping}
			{overRelaxation}
			{showParticles}
			{showGrid}
			particleSize={particleSize}
		/>

		{#if appState === 'ready'}
			<PopupInfo />
		{/if}
	{/if}
</div>
