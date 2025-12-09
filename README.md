ScalingCases is a curated collection of reproducible Python codes and data for all confinement and scaling regimes (Px and Py), including diagnostics, phase diagrams, and publication ready figures for polymer physics research.

Overview
ScalingCases provides a unified repository of scripts, data, and figures covering multiple cases of scaling behavior for confined polymers, including variations in system size, tether position, confinement strength, persistence length, boundary layers, and σ–scaling.
Features
	Reproducible Python implementations of all Px and Py scaling cases (1D and related diagnostics).
	Automated figure generation at high resolution suitable for direct use in manuscripts and supplementary materials.
	CSV outputs for quantitative diagnostics (e.g., κ, first-mode fractions, RMS residuals, moments).
	Modular functions for analytic Px, modal coefficients, and collapse analysis that can be reused in new studies.
	Repository Structure
	case1_...px: Scaling with x/L for different L at fixed κ, including distribution collapse and residuals.
	case2_...px: Effect of tether position, with overlays across L and CSV statistics. 
	case3_...px: Confinement-strength (κ) scaling, kappa diagnostics, and regime identification. 
	case4_...px: Persistence length effects and standardized collapse vs Gaussian reference.
	case5_...px: Boundary-layer inner scaling near absorbing walls.
	case6_...px: σ–scaling tests (tether-centered) across L and κ, including negative tests.
	case1_...py: Geometric similarity y/R
	Case 2_py: Varying polymer flexibility at fixed confinement width 
	Case3_...py: Image-Method Distributions Across Confinement Regimes
	Case4_...py: Chain-length variation.
	Case5_...py: Image–method collapse with α and β scalings

	Modal analysis
	Phase diagram
Installation
	Requires Python 3.x with NumPy, Matplotlib, SciPy, and Pandas.
	Install dependencies using your preferred environment manager (e.g., pip install numpy matplotlib scipy pandas).
Usage
	Run any case script directly to reproduce the corresponding figures and CSV diagnostics.
	Adjust physical parameters (N, a, L, κ, tether position) and numerical settings (grids, tolerances, nmax) via the user parameter blocks at the top of each script.
	Generated figures and CSV files are written to the figures/ and data/ subdirectories with descriptive filenames.
Applications
	Benchmarking analytic and numerical models of confined polymers.
	Validating scaling hypotheses and regime boundaries (blob, transition, deflection).
	Producing publication quality figures and supplementary diagnostics from a single, coherent codebase.
Citation
If you use ScalingCases in published work, please cite the associated manuscript and mention this repository as the source of the scaling case implementations and diagnostics.
https://github.com/soumyakabi/Scaling-of-Endpoint-Probability-distribution-function-of-confined-polymer
