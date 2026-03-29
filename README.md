# SurrogateNN_DSGE

This repository is a gradual Python/JAX port of the Julia code in `MacroModelling` as used inside `/Volumes/MacMini/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl`.

The porting rule for this repository is strict:

- work feature by feature
- keep the Julia source read-only
- only expose behavior that has been implemented and tested
- prefer JAX-native kernels that can later feed NumPyro and GPU execution

## Current status

Implemented:

- discrete Lyapunov solver for `A X A^T + C = X`
- JAX-native doubling kernel
- JAX-native dense direct solver fallback
- Julia-compatible Lyapunov algorithm names `bartels_stewart`, `bicgstab`, `gmres`, and `dqgmres`, with tested residual parity and direct fallback when the iterative path is cut short; `dqgmres` is a compatibility alias routed to the SciPy GMRES backend
- discrete Sylvester solver for `A X B + C = X`
- Julia-compatible iterative Sylvester algorithm names `bicgstab`, `gmres`, and `dqgmres`, with tested parity against the dense direct solve and direct fallback when the Krylov path is cut short; `dqgmres` is a compatibility alias routed to the SciPy GMRES backend
- linear Gaussian state-space simulation, Kalman likelihood, filtering, and RTS smoothing
- first-order inversion-filter likelihoods with per-period contributions and warmup support
- parsed-model observable-name resolution and first-order state-space construction for Kalman estimation
- parsed-model Kalman loglikelihood helpers for named observable data in levels, including per-period likelihood contributions, Julia-style sorted-observable handling for array inputs, and verified `Smets_Wouters_2007_HLT` parity against Julia for total likelihood, per-period likelihood, and filtered/smoothed Kalman state extraction
- parsed-model inversion loglikelihood helpers for both first-order and stochastic extended path likelihoods, including SEP inversion diagnostics and Julia-style runtime override keywords
- SEP inversion now carries shifted nonlinear warm starts across observation periods and reports period-level carry usage plus SEP predict-call counts in the diagnostics, which materially improves the sparse-tree FOM path used by the switching-order workflow
- regime-switching likelihood mixing with supplied hard masks or gate probabilities, plus a parsed-model bridge that mixes ROM Kalman and FOM inversion per-period likelihoods
- a parsed-model switching pipeline report helper that runs ROM likelihoods, FOM likelihoods, switching likelihoods, hard-mask decompositions, gate summaries, SEP diagnostics, and optional runtime comparisons in one call, so sparse-tree SEP switching paths can be compared without manual stitching
- the switching comparison layer now also reports gate-decision quality against oracle per-period regime choices and against a budget-matched oracle with the same number of nonlinear periods, plus probability-quality diagnostics such as Brier score, log loss, AUC, and hard-threshold regret; this makes the switching methodology auditable rather than only runnable
- the switching comparison layer now also reports a budget frontier over the gate scores, so the ROM/FOM methodology can be judged on how close its top-k nonlinear choices stay to the same-budget oracle across a range of nonlinear-compute budgets instead of at only one threshold
- the parsed-model layer now also has a parameter-sweep likelihood-surface report, so ROM, FOM, and switching likelihood totals can be compared across parameter draws using ranking-sensitive diagnostics such as Pearson/Spearman correlation, top-draw overlap, best-draw agreement, and switching-vs-ROM error ratios relative to the FOM surface
- gate-stat computation, threshold calibration, probability mapping, padding, and automatic hard-regime assignment utilities for the switching layer
- first-order rollout helpers and parsed-model linear gate-stat computation from supplied shock paths, including named observable/shock sigma inputs
- JAX-native gate-stat kernels and a compiled parsed-model first-order supplied-shock gate-stat helper, so linear gate diagnostics can now be built under `jax.jit` from solved first-order paths as well
- compiled first-order filter helpers for observed-shock reconstruction, filtered/smoothed state extraction, filter-derived gate statistics, and JAX-native gate-probability construction, so the switching-order first-order filter workflow can now stay inside JAX as well
- Julia-style first-order observed-shock / observed-variable estimation helpers, linear filter state extractors, and filter-based linear gate-stat workflows for both Kalman and inversion filters
- parser/source compatibility smoke across upstream MacroModelling model files now passes on `42/43` checked `.jl` model sources; the remaining `testqipf.jl` failure is explicitly flagged as an upstream `1GAMM` typo rather than silently accepted
- raw-source first-order solve smoke now succeeds on `18/42` parse-compatible upstream `.jl` model files without any manual steady-state seed, after replacing the old all-ones default guess with a MacroModelling-style feasibility heuristic and widening the steady-state feasibility search to larger geometric restart probes on both the NumPy and JAX paths; the largest remaining blockers are harder nonlinear steady states, `qipf.jl` helper-function execution, and a few large-model Newton/OBC edge cases
- source-level helper-backed parameter definitions now support the upstream `QMIPF_solve_SS(...)` routine used by `qipf.jl`, so raw `qipf.jl` advances through parameter resolution and reaches the actual steady-state/solution layer instead of failing at a missing-function error
- tail `if` / `elseif` / `else` conditionals inside `@parameters` blocks are now supported through block end, which matches the upstream QIPF-style source pattern used to override definitions like `SS_U` when `sigma == 1`; raw `models/QIPF/testttfmodel_ttf_minimum.jl` now parses successfully
- switching diagnostics and comparison helpers for gated episodes, loglikelihood decompositions, and runtime summaries
- callback-based switching likelihood helpers including named-parameter selection/override, conditional and additive-residual loglikelihood utilities, generic inversion-step helpers, linear-reference likelihood comparison helpers, Julia-style shock reconstruction from epsilon means, and chunked-sampling orchestration
- posterior-chain convenience helpers for switching workflows, including `theta_draws`, `epsilon_means_from_chain`, and `chunk_stats`, with support for NumPyro `MCMC` objects plus raw/checkpoint-style sample mappings
- optional NumPyro inference helpers for subset priors, parameter-vector assembly, and concrete log-density evaluation on top of the parsed-model Kalman likelihood
- JAX first-order structural likelihood and NumPyro wrappers that can run compiled kernels like `NUTS` on the parsed-model first-order path with either an explicit steady state or automatic JAX steady-state and calibration-equation solves
- JAX first-order switching likelihood and NumPyro wrappers for fixed gate probabilities or hard masks, so ROM Kalman and FOM inversion can now be mixed on the compiled first-order path as well
- compiled first-order switching likelihoods and NumPyro wrappers with automatic filter-derived gates, so the first-order switching-order estimation path can now run inside JAX/NumPyro without supplying gate probabilities by hand
- quadratic matrix equation doubling solver plus a generalized Schur / ordered-QZ solver for the Julia `:schur` path
- explicit Schur / ordered-QZ determinacy diagnostics for first-order models, including stable-root counts, unique/indeterminate/no-stable classification, and parsed-model wrappers to inspect the Schur branch directly
- parsed-model state-space, likelihood, filtering, gate-stat, and concrete/compiled NumPyro helpers now expose `qme_algorithm` so first-order workflows can explicitly choose between the doubling and Schur/QZ solution branches
- MacroModelling-style `@model` / `@parameters` source parsing for the first-order path
- parsed model and parameter block options for common upstream directives, including `max_obc_horizon`, `simplify`, `verbose`, `silent`, `symbolic`, and `perturbation_order`
- basic `max` / `min` OBC syntax parsing with parsed-model `has_obc` detection, steady-state support, inactive-branch first-order solves, and SEP path solves on simple bound models
- branch-frozen OBC derivative evaluation around active steady-state branches, so binding `max` / `min` constraints no longer linearize with spurious `0.5` derivatives at the kink
- SEP Jacobian selection via `jacobian_method=\"auto\" | \"autodiff\" | \"finite_difference\" | \"subgradient\"`, with parsed OBC SEP solves automatically switching `auto` to a branch-frozen subgradient Jacobian on the Gauss-Hermite path
- parsed-model OBC violation diagnostics, including single-period violation evaluation, full-path violation checks, and first-order rollout diagnostics that flag when the linearized path breaches a `max` / `min` bound
- parsed-model `get_irf` and `simulate_model` runtime helpers for first-order and deterministic SEP paths, including named shock selection, grouped nested variable/shock-name inputs, Julia-style selector tokens such as `:all_excluding_obc` and `:all_excluding_auxiliary_and_obc`, explicit shock histories, Julia-style `shocks="simulate"` random simulations with deterministic seeding, variable selection, a dedicated first-order OBC enforcement path for simple direct or monotone-transformed current-variable `max` / `min` equations such as `log(R[0]) = max(...)`, recovery of implied OBC shock sequences when explicit OBC shock channels are present, receding-horizon OBC shock-sequence optimization across `max_obc_horizon + 1` periods on the dedicated first-order path, Julia-style `ignore_obc` override when OBC-tagged shocks are present, and `max_obc_horizon`-aware SEP horizon extension for unsupported first-order OBC runtime requests
- MacroModelling-style inline time-index `for` loops inside `@model` equations, including additive and `operator = :*` forms
- explicit curly-brace indexed identifiers such as `y{H}[0]` and `rho{H}{F}` across parsed model and parameter blocks
- top-level `for`-block expansion in `@model` for explicit identifier lists like `[H, F]`, named source-level collections like `countries = [:H, :F]`, and integer ranges
- order-independent parameter definitions in `@parameters` blocks
- calibration equations in `@parameters` blocks, including both `par | lhs = rhs` and `lhs = rhs | par` forms
- indexed parameter-family broadcasting in `@parameters`, including `alpha = 0.3` style direct definitions and one-family calibration equations like `y[ss] = target | beta`
- `@parameters ... guess = Dict(...)` default steady-state guesses for variables and calibrated parameters
- inequality bounds in `@parameters` such as `0 < alpha < 1`, `x >= 0`, and `10 >= R`, with bounded Newton projection for steady-state and calibration solves
- MacroModelling special-function aliases including `normpdf`, `dnorm`, `pnorm`, `normlogpdf`, `erfinv`, and `erfcinv`
- lazy symbolic matrix construction and cached `sympy.lambdify` compilation for parsed models, so large MacroModelling-style sources parse without eagerly compiling every derivative object
- symbolic Jacobian, Hessian, and third-order derivative evaluation with Julia-compatible compressed ordering
- damped Newton non-stochastic steady-state solver
- steady-state and calibrated-parameter Newton restart heuristics plus finite-difference Jacobian fallback on both the NumPy and JAX paths, including generic sign-preserving unit-scale feasibility restarts when the model-specific default guess family stays outside the residual domain, so compiled first-order estimation no longer fails immediately on non-finite default guesses or singular autodiff Jacobians
- the NumPy steady-state path now also has a bounded nonlinear least-squares rescue stage followed by a cleanup Newton pass when multi-restart Newton stalls at a finite moderate-residual point
- the steady-state Newton stack now also has a last-resort square-system residual homotopy continuation fallback on both the NumPy and JAX paths, which only activates after the restart and least-squares layers fail and only replaces the incumbent result when it reaches the true target system at the final homotopy level
- conservative symbolic steady-state seeding from uniquely solvable steady-state equations when `@parameters ... symbolic = true`, on both the NumPy and JAX steady-state paths
- a small nearest-parameter steady-state cache on both the NumPy and JAX paths, so repeated solves reuse nearby converged steady states as the default guess instead of restarting from the generic heuristic every time
- calibrated-parameter resolution from either the joint steady-state solve or a supplied steady state
- first-order DSGE perturbation solver with Julia `RBC_CME` fixture coverage
- second-order DSGE perturbation solver with Julia-compatible compression matrices and Sylvester solve
- second-order stochastic steady-state solver for both standard and pruned second-order updates
- parsed-model second-order solve path from MacroModelling-style source through symbolic Hessians to the perturbation solution
- third-order DSGE perturbation solver with Julia-compatible tensor compression/permutation machinery and Sylvester solve
- third-order stochastic steady-state solver for standard third-order solutions plus pruned third-order state updates
- parsed-model third-order solve path from MacroModelling-style source through symbolic third derivatives to the perturbation solution
- generic callback-based stochastic extended path solver with Gauss-Hermite branching
- sparse fishbone stochastic extended path branching for callback-based and parsed-model SEP solves, including real `sep_sparse_tree` runtime support in the inversion bridge
- sparse-tree SEP now precomputes parent/child/shock tree metadata once per solve instead of rebuilding the same fishbone combinatorics inside every Newton residual evaluation
- adaptive Levenberg-Marquardt damping, Julia-style SEP config controls for `linear_solver`, `fallback_solver`, stall detection, and bounded backtracking line search, explicit config validation, and checked warm-start support for more robust nonlinear SEP solves
- Julia-style SEP acceptance tolerances via `SEPConfig(accept_tol=...)`, so parsed-model SEP runtimes can now accept a finite residual band even when strict Newton convergence is missed, matching the updated upstream `sep_accept_tol` behavior instead of requiring the inversion wrapper to handle that case alone
- automatic first-order linear-path warm starts for parsed-model SEP solves and deterministic SEP runtime fallbacks, with explicit `initial_guess` still taking precedence
- parsed-model SEP now also mirrors the updated Julia OBC control flow more closely when explicit OBC shock channels are present: if the initial nonlinear SEP path is rejected or still violates the transformed OBC constraints, the runtime recovers a linear OBC shock sequence on the first-order path, injects those anticipated shocks back into the deterministic SEP horizon, and re-runs SEP iteratively; the SEP-backed runtime helpers now return those adjusted OBC shocks instead of silently reporting only the user-supplied path
- the high-level switching bridge is now regression-tested on a nonlinear sparse-tree SEP FOM path as well, so ROM Kalman plus sparse-tree SEP inversion can be compared end to end against a manual likelihood mixture on the same model
- HMC expectation backend for both SEP callback APIs, including parsed-model SEP solves and optional parallel tempering
- parsed-model `solve_sep_at_noise_level(...)`, `homotopy_sep(...)`, and `homotopy_chained_trajectory(...)` utilities, porting the updated Julia sigma-continuation SEP robustness workflow; `sigma = 0` now forces a deterministic perfect-foresight SEP step, intermediate noise levels scale deterministic shocks directly, adaptive subdivision retries harder nonlinear paths before giving up, and the chained helper turns that continuation logic into a period-by-period nonlinear trajectory generator
- parsed-model stochastic extended path solve path with JAX dynamic residual evaluation and residual-expectation averaging over future branches
- focused tests for residuals, symmetry, fallback behavior, JIT, autodiff, parser parity, inversion filtering, switching likelihoods, gate calibration, and multi-model JAX compile smoke across upstream model files

Not implemented yet:

- perturbation orders above third
- ambiguous multi-family calibration-equation broadcasting remains guarded rather than inferred
- the remaining non-equation `@parameters` directives from the Julia macro layer beyond `guess`, bounds, `silent`, `symbolic`, `perturbation_order`, `precompile`, and the common upstream `simplify` / `verbose` options
- full Julia-style occasionally binding constraint enforcement around kinks, including the broader shock-sequence/runtime surface beyond the new dedicated first-order receding-horizon OBC shock optimization, full kink-aware runtime switching, and the remaining OBC enforcement modes outside the newly supported direct / monotone-transformed / simple complementarity subset
- the new first-order OBC runtime path now covers direct and simple monotone-transformed current-variable `max` / `min` equations, but more general parsed OBC models still fall back to deterministic SEP when `ignore_obc = false`
- the hard remaining raw-source bottleneck is now clearly the largest nonlinear steady states rather than OBC syntax support alone; on a manual upstream spot-check, the new hybrid rescue reduced the raw `Guerrieri_Iacoviello_2017.jl` steady-state residual from about `4.3e-1` to about `1.2e-2` at the default budget, but did not yet fully converge it automatically
- the broader Julia runtime selection surface is still narrower here than upstream, even though grouped nested name inputs and the main selector tokens like `:all_excluding_obc` and `:all_excluding_auxiliary_and_obc` are now supported
- the remaining sparse-tree-specific Jacobian/runtime optimizations and the broader OBC-specific SEP machinery beyond the new subgradient / finite-difference Jacobian safeguards; the new parsed-model SEP shock-reinjection loop currently reuses the linear OBC shock recovery path rather than a full nonlinear subdifferential Newton / nonlinear shock optimizer
- fully GPU-native generalized QZ / ordered-QZ primitives; the current JAX-facing `schur` QME path uses a SciPy host callback in the primal solve because JAX does not yet expose generalized `qz` / `ordqz`
- the public first-order default now matches Julia and uses `schur`; request `qme_algorithm=\"doubling\"` explicitly if you want to stay on the fully JAX-native doubling path
- the new determinacy diagnostics are currently tied to the Schur/QZ path; the doubling path still solves the QME but does not provide a comparable stable-root decomposition report
- fully JAX-traceable parsed-model structural likelihoods beyond the first-order path, including higher-order estimation edges for compiled NumPyro kernels like `NUTS` and `HMC`
- the older high-level parsed-model switching/filter bridge in `model.py` still exists for concrete NumPy-oriented use, but the first-order switching-order estimation path now has compiled JAX counterparts end to end; the remaining gaps are mainly higher-order and nonlinear switching surfaces rather than the first-order filter gate path
- the broader regime-switching estimation harness beyond the currently ported likelihood mixer and gate/regime utilities

Progress is tracked in [docs/porting_progress.md](docs/porting_progress.md).
