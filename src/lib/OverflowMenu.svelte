<script lang="ts">
	import { cubicOut } from 'svelte/easing';
	import { Tween } from 'svelte/motion';
	import { Menu, Plus, Minus, ArrowUpRight, Hand, ChevronDown, ChevronUp } from '@lucide/svelte/icons';

	let {
		gravity = $bindable({ x: 0, y: -9.81 }),
		resolution = $bindable(70),
		fluidColor = $bindable({ r: 0.09, g: 0.4, b: 1.0 }),
		foamColor = $bindable({ r: 0.75, g: 0.9, b: 1.0 }),
		colorDiffusionCoeff = $bindable(0.0008),
		foamReturnRate = $bindable(0.5),
		damping = $bindable(1.0),
		overRelaxation = $bindable(1.7),
		showParticles = $bindable(true),
		showGrid = $bindable(false),
		particleSize = $bindable(0.03),
		gravityStrength = $bindable(9.81)
	} = $props();

	let menuOpen = $state(false);
	let settingsExpanded = $state(false);
	let activeMode = $state<'add' | 'remove' | 'push' | 'grab'>('add');
	let menuWidth = $state(0);

	const modeColors: Record<string, string> = {
		add: 'bg-blue-500',
		remove: 'bg-red-500',
		push: 'bg-amber-500',
		grab: 'bg-purple-500'
	};

	const modeLabels: Record<string, string> = {
		add: 'Add',
		remove: 'Remove',
		push: 'Push',
		grab: 'Grab'
	};

	function toggleMenu() {
		menuOpen = !menuOpen;
		if (!menuOpen) {
			settingsExpanded = false;
		}
	}

	function setMode(mode: 'add' | 'remove' | 'push' | 'grab') {
		activeMode = mode;
	}

	function presetColor(r: number, g: number, b: number) {
		fluidColor = { r, g, b };
		const foamR = Math.min(1, r + 0.3);
		const foamG = Math.min(1, g + 0.15);
		const foamB = Math.min(1, b + 0.15);
		foamColor = { r: foamR, g: foamG, b: foamB };
	}
</script>

<div class="fixed right-4 top-4 z-50 flex flex-col items-end">
	<!-- Overflow Menu Button -->
	<button
		onclick={toggleMenu}
		class="flex h-11 w-11 items-center justify-center rounded-full bg-gray-800/80 text-white shadow-lg backdrop-blur-md transition-all hover:bg-gray-700/90 hover:scale-105 focus:outline-none focus:ring-2 focus:ring-white/20"
		aria-label="Open menu"
	>
		<Menu class="h-5 w-5" />
	</button>

	<!-- Expanded Panel -->
	{#if menuOpen}
		<div
			class="mt-2 w-[340px] overflow-hidden rounded-2xl bg-gray-900/95 shadow-2xl backdrop-blur-xl"
			style="transition: height 0.35s {cubicOut}"
		>
			<!-- Compact Mode: Horizontal Pill -->
			{#if !settingsExpanded}
				<div class="p-3">
					<!-- Mode buttons in a pill -->
					<div class="flex items-center justify-center rounded-xl bg-gray-800/70 p-1.5">
						{#each ['add', 'remove', 'push', 'grab'] as mode}
							<button
								onclick={() => setMode(mode)}
								class="flex h-9 flex-1 items-center justify-center rounded-lg px-2 text-xs font-semibold text-white transition-all"
								class:bg-blue-500={activeMode === mode && mode === 'add'}
								class:bg-red-500={activeMode === mode && mode === 'remove'}
								class:bg-amber-500={activeMode === mode && mode === 'push'}
								class:bg-purple-500={activeMode === mode && mode === 'grab'}
								class:bg-transparent={activeMode !== mode}
								class:hover:bg-gray-700={activeMode !== mode}
								class:text-white={activeMode !== mode}
								class:text-black={activeMode === mode}
							>
								{#if mode === 'add'}
									<Plus class="h-4 w-4" />
								{:else if mode === 'remove'}
									<Minus class="h-4 w-4" />
								{:else if mode === 'push'}
									<ArrowUpRight class="h-4 w-4" />
								{:else}
									<Hand class="h-4 w-4" />
								{/if}
							</button>
							{#if mode !== 'grab'}
								<div class="h-5 w-px bg-gray-600/60" />
							{/if}
						{/each}
					</div>

					<!-- View More Settings button -->
					<button
						onclick={() => (settingsExpanded = true)}
						class="mt-3 flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 py-2.5 text-sm font-semibold text-white shadow-lg transition-all hover:from-blue-500 hover:to-indigo-500"
					>
						<ChevronDown class="h-4 w-4" />
						View More Settings
					</button>
				</div>
			{:else}
				<!-- Expanded Settings Panel -->
				<div class="max-h-[80vh] overflow-y-auto p-4" style="scrollbar-width: thin; scrollbar-color: #4b5563 transparent;">
					<!-- Header -->
					<div class="mb-4 flex items-center justify-between">
						<h2 class="text-lg font-bold text-white">Simulation Settings</h2>
						<button
							onclick={() => (settingsExpanded = false)}
							class="rounded-lg p-1.5 text-gray-400 transition-colors hover:bg-gray-800 hover:text-white"
							aria-label="Collapse settings"
						>
							<ChevronUp class="h-5 w-5" />
						</button>
					</div>

					<!-- Mode selector (compact) -->
					<div class="mb-5 flex items-center justify-center gap-1 rounded-xl bg-gray-800/70 p-1">
						{#each ['add', 'remove', 'push', 'grab'] as mode}
							<button
								onclick={() => setMode(mode)}
								class="flex h-9 flex-1 items-center justify-center rounded-lg px-2 text-xs font-semibold text-white transition-all"
								class:bg-blue-500={activeMode === mode && mode === 'add'}
								class:bg-red-500={activeMode === mode && mode === 'remove'}
								class:bg-amber-500={activeMode === mode && mode === 'push'}
								class:bg-purple-500={activeMode === mode && mode === 'grab'}
								class:bg-transparent={activeMode !== mode}
								class:hover:bg-gray-700={activeMode !== mode}
							>
								{#if mode === 'add'}
									<Plus class="h-4 w-4" />
								{:else if mode === 'remove'}
									<Minus class="h-4 w-4" />
								{:else if mode === 'push'}
									<ArrowUpRight class="h-4 w-4" />
								{:else}
									<Hand class="h-4 w-4" />
								{/if}
							</button>
							{#if mode !== 'grab'}
								<div class="h-5 w-px bg-gray-600/60" />
							{/if}
						{/each}
					</div>

					<!-- Fluid Color Presets -->
					<div class="mb-5">
						<label class="mb-2 block text-xs font-medium uppercase tracking-wider text-gray-400">
							Fluid Color
						</label>
						<div class="grid grid-cols-4 gap-2">
							<button onclick={() => presetColor(0.09, 0.4, 1.0)} class="h-9 rounded-lg bg-blue-600 transition-transform hover:scale-105" />
							<button onclick={() => presetColor(0.0, 0.7, 0.8)} class="h-9 rounded-lg bg-cyan-500 transition-transform hover:scale-105" />
							<button onclick={() => presetColor(1.0, 0.4, 0.1)} class="h-9 rounded-lg bg-orange-500 transition-transform hover:scale-105" />
							<button onclick={() => presetColor(0.5, 0.2, 0.9)} class="h-9 rounded-lg bg-purple-600 transition-transform hover:scale-105" />
							<button onclick={() => presetColor(0.1, 0.6, 0.4)} class="h-9 rounded-lg bg-emerald-500 transition-transform hover:scale-105" />
							<button onclick={() => presetColor(0.9, 0.5, 0.6)} class="h-9 rounded-lg bg-pink-500 transition-transform hover:scale-105" />
							<button onclick={() => presetColor(0.3, 0.7, 0.9)} class="h-9 rounded-lg bg-sky-500 transition-transform hover:scale-105" />
							<button onclick={() => presetColor(0.9, 0.7, 0.2)} class="h-9 rounded-lg bg-yellow-500 transition-transform hover:scale-105" />
						</div>
					</div>

					<!-- Sliders -->
					<div class="space-y-4">
						<!-- Resolution -->
						<div>
							<div class="mb-1 flex justify-between">
								<label class="text-xs font-medium text-gray-300">Resolution</label>
								<span class="text-xs text-gray-400">{resolution}</span>
							</div>
							<input
								type="range"
								min="20"
								max="120"
								step="1"
								bind:value={resolution}
								class="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-gray-700 accent-blue-500"
							/>
						</div>

						<!-- Gravity Strength -->
						<div>
							<div class="mb-1 flex justify-between">
								<label class="text-xs font-medium text-gray-300">Gravity Strength</label>
								<span class="text-xs text-gray-400">{gravityStrength.toFixed(1)}</span>
							</div>
							<input
								type="range"
								min="0"
								max="20"
								step="0.1"
								bind:value={gravityStrength}
								oninput={() => {
									const angle = 0;
									gravity = {
										x: gravityStrength * Math.sin(angle),
										y: -gravityStrength
									};
								}}
								class="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-gray-700 accent-blue-500"
							/>
						</div>

						<!-- Viscosity (Damping) -->
						<div>
							<div class="mb-1 flex justify-between">
								<label class="text-xs font-medium text-gray-300">Viscosity</label>
								<span class="text-xs text-gray-400">{damping.toFixed(2)}</span>
							</div>
							<input
								type="range"
								min="0.1"
								max="2"
								step="0.01"
								bind:value={damping}
								class="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-gray-700 accent-blue-500"
							/>
						</div>

						<!-- Particle Size -->
						<div>
							<div class="mb-1 flex justify-between">
								<label class="text-xs font-medium text-gray-300">Particle Size</label>
								<span class="text-xs text-gray-400">{particleSize.toFixed(3)}</span>
							</div>
							<input
								type="range"
								min="0.01"
								max="0.1"
								step="0.001"
								bind:value={particleSize}
								class="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-gray-700 accent-blue-500"
							/>
						</div>

						<!-- Color Diffusion -->
						<div>
							<div class="mb-1 flex justify-between">
								<label class="text-xs font-medium text-gray-300">Color Diffusion</label>
								<span class="text-xs text-gray-400">{colorDiffusionCoeff.toExponential(1)}</span>
							</div>
							<input
								type="range"
								min="0.0001"
								max="0.003"
								step="0.0001"
								bind:value={colorDiffusionCoeff}
								class="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-gray-700 accent-blue-500"
							/>
						</div>

						<!-- Foam Return Rate -->
						<div>
							<div class="mb-1 flex justify-between">
								<label class="text-xs font-medium text-gray-300">Foam Return Rate</label>
								<span class="text-xs text-gray-400">{foamReturnRate.toFixed(2)}</span>
							</div>
							<input
								type="range"
								min="0"
								max="1"
								step="0.01"
								bind:value={foamReturnRate}
								class="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-gray-700 accent-blue-500"
							/>
						</div>

						<!-- Over-Relaxation -->
						<div>
							<div class="mb-1 flex justify-between">
								<label class="text-xs font-medium text-gray-300">Over-Relaxation</label>
								<span class="text-xs text-gray-400">{overRelaxation.toFixed(2)}</span>
							</div>
							<input
								type="range"
								min="1"
								max="3"
								step="0.01"
								bind:value={overRelaxation}
								class="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-gray-700 accent-blue-500"
							/>
						</div>
					</div>

					<!-- Toggles -->
					<div class="mt-5 space-y-3">
						<div class="flex items-center justify-between">
							<span class="text-sm text-gray-300">Show Particles</span>
							<button
								onclick={() => (showParticles = !showParticles)}
								class="relative h-6 w-11 rounded-full transition-colors"
								class:bg-blue-600={showParticles}
								class:bg-gray-700={!showParticles}
								aria-label="Toggle particles"
							>
								<div
									class="absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition-transform"
									class:translate-x-5={showParticles}
									class:translate-x-0={!showParticles}
								/>
							</button>
						</div>

						<div class="flex items-center justify-between">
							<span class="text-sm text-gray-300">Show Grid</span>
							<button
								onclick={() => (showGrid = !showGrid)}
								class="relative h-6 w-11 rounded-full transition-colors"
								class:bg-blue-600={showGrid}
								class:bg-gray-700={!showGrid}
								aria-label="Toggle grid"
							>
								<div
									class="absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition-transform"
									class:translate-x-5={showGrid}
									class:translate-x-0={!showGrid}
								>
								</div>
							</button>
						</div>
					</div>

					<!-- Reset button -->
					<button
						onclick={() => {
							resolution = 70;
							gravityStrength = 9.81;
							gravity = { x: 0, y: -9.81 };
							damping = 1.0;
							overRelaxation = 1.7;
							colorDiffusionCoeff = 0.0008;
							foamReturnRate = 0.5;
							showParticles = true;
							showGrid = false;
							particleSize = 0.03;
							presetColor(0.09, 0.4, 1.0);
						}}
						class="mt-6 w-full rounded-xl bg-gray-800 py-2.5 text-sm font-medium text-gray-300 transition-colors hover:bg-gray-700"
					>
						Reset to Defaults
					</button>
				</div>
			{/if}
		</div>
	{/if}
</div>

<style>
	input[type='range']::-webkit-slider-thumb {
		-webkit-appearance: none;
		appearance: none;
		width: 16px;
		height: 16px;
		border-radius: 50%;
		background: #3b82f6;
		cursor: pointer;
		box-shadow: 0 0 6px rgba(59, 130, 246, 0.5);
	}

	input[type='range']::-moz-range-thumb {
		width: 16px;
		height: 16px;
		border-radius: 50%;
		background: #3b82f6;
		cursor: pointer;
		border: none;
		box-shadow: 0 0 6px rgba(59, 130, 246, 0.5);
	}

	input[type='range']::-webkit-slider-runnable-track {
		height: 6px;
		border-radius: 999px;
		background: #374151;
	}

	input[type='range']::-moz-range-track {
		height: 6px;
		border-radius: 999px;
		background: #374151;
	}
</style>
