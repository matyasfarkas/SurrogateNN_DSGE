## Methodology Review

Date: March 23, 2026

Scope reviewed:

- [switching.py](/Volumes/MacMini/matyasfarkas/Documents/GitHub/SurrogateNN_DSGE/src/surrogatenn_dsge/switching.py)
- [inversion.py](/Volumes/MacMini/matyasfarkas/Documents/GitHub/SurrogateNN_DSGE/src/surrogatenn_dsge/inversion.py)
- [sep.py](/Volumes/MacMini/matyasfarkas/Documents/GitHub/SurrogateNN_DSGE/src/surrogatenn_dsge/sep.py)
- [inference.py](/Volumes/MacMini/matyasfarkas/Documents/GitHub/SurrogateNN_DSGE/src/surrogatenn_dsge/inference.py)
- [model.py](/Volumes/MacMini/matyasfarkas/Documents/GitHub/SurrogateNN_DSGE/src/surrogatenn_dsge/model.py)

### Executive assessment

The project is strongest as a systems contribution, not as an algorithmic one.

Taken separately, the building blocks are not novel:

- first-order perturbation
- stochastic extended path
- sparse-tree branching
- regime switching via a gate
- HMC-based expectation approximations
- neural surrogates

The possible contribution is the integrated pipeline:

- ROM1 for cheap broad likelihood coverage
- FOM via sparse-tree SEP when linearization is unreliable
- a switching layer that chooses when nonlinear work is worth paying for
- eventual learned surrogates to replace or approximate the FOM leg

That combination is potentially useful and publishable. It is not automatically convincing. The burden of proof is empirical and methodological rather than conceptual.

### What is already strong

1. The codebase now exposes a real end-to-end path rather than disconnected pieces.

The current stack supports:

- parsed MacroModelling-style model input
- first-order Schur and doubling solutions
- Kalman and inversion likelihoods
- sparse-tree SEP nonlinear likelihoods
- switching mixtures between ROM and FOM
- JAX/NumPyro first-order estimation paths

This matters because many methodological proposals in this area remain at the sketch level.

2. The nonlinear leg is not purely decorative.

Sparse-tree SEP, carry warm starts, homotopy continuation, and OBC-specific handling show that the nonlinear path is being treated as the serious bottleneck it actually is.

3. The methodology is increasingly auditable.

Before this pass, the switching stack could produce a mixed likelihood, but it could not answer the most important methodological question:

- did the gate choose the right nonlinear periods?

That gap is now narrower because the code reports:

- oracle regret
- budget-matched oracle regret
- captured positive nonlinear gain share
- wasted nonlinear cost
- probability-quality metrics such as Brier score and AUC

This is the right direction. Approximation methods need decision diagnostics, not just aggregate totals.

### What is not yet convincing

1. The novelty claim is still fragile.

The methodology is only interesting if the switching logic produces a materially better speed-accuracy tradeoff than the obvious baselines:

- pure ROM
- pure FOM
- naive fixed nonlinear windows
- top-k ex post nonlinear periods

Without that comparison, the pipeline is a sophisticated engineering stack, not yet a demonstrated methodological advance.

2. The hardest models still fail at the steady-state layer.

This is not a cosmetic issue. It directly weakens the contribution because the models that most need the nonlinear machinery are also the ones least likely to run robustly end to end.

On the updated upstream tree, the main remaining failures are still concentrated in:

- `Smets_Wouters_2007_HLT_obc.jl`
- `Smets_Wouters_2007_HLT_obc_smooth.jl`
- `QMIPF_final.jl`
- `qipf.jl`

That means the methodology is still strongest on toy, moderate, or already-cooperative models. A serious paper needs the difficult models to be first-class citizens.

3. The switching gate still risks looking post hoc.

The gate is built from linear shock and observation diagnostics. That is sensible, but it is not yet clear that these statistics are the right sufficient signals for nonlinear relevance.

The failure mode is obvious:

- the gate can be well calibrated against its own proxies while still missing the true high-value nonlinear periods in likelihood space

The new oracle-regret diagnostics make this falsifiable, but the burden now is to show small regret on hard models.

4. The surrogate story is still prospective.

Right now the codebase contains many of the ingredients needed for surrogate estimation, but not yet the proof that a surrogate will preserve inference quality. That is the highest-risk piece of the eventual contribution.

The most serious surrogate-specific risks are:

- approximation error concentrated near OBC kinks
- posterior bias from small local likelihood errors
- gate-surrogate interaction, where the surrogate looks good conditional on one gate but bad under another
- training on trajectories that do not sufficiently cover the inference-relevant state space

### Harsh originality verdict

Originality by component: low.

Originality by integration: moderate.

Potential scientific value: high only if the empirical regret/runtime tradeoff is demonstrated on hard models.

Potential practical value: high, because a robust ROM1/FOM switching stack is useful even if not theoretically groundbreaking.

Current maturity as a methodology: promising, but not yet convincing.

### What was added in this review pass

The most important new addition is methodological instrumentation rather than another solver:

- `oracle_nonlinear_mask(...)`
- `optimal_nonlinear_mask_for_budget(...)`
- `evaluate_gate_decisions(...)`
- `evaluate_gate_probabilities(...)`

These now flow through the high-level `switching_pipeline_report(...)`, so every ROM/FOM switching comparison can report:

- how far the chosen gate is from the unconstrained oracle
- how far it is from the best same-budget nonlinear schedule
- how much nonlinear benefit it captures
- how much nonlinear work it wastes
- how well the gate probabilities rank truly useful nonlinear periods

This does not by itself prove the methodology. It does something more important first: it makes it possible to disprove weak versions of the methodology quickly and honestly.

### What would make the contribution convincing

1. Show low switching regret on difficult models.

The key metric is not just `switching_total - fom_total`. It is:

- regret vs unconstrained oracle
- regret vs same-budget oracle

If the gate is good, the same-budget regret should be small.

2. Show real runtime gains at that same regret level.

The methodology only matters if it dominates a pure-FOM baseline and is materially better than a naive same-budget nonlinear schedule.

3. Stress-test OBC episodes.

The paper-quality evidence needs targeted cases where:

- OBCs bind
- linearization is locally misleading
- the switching layer still identifies the right windows

4. Make surrogate claims only after inference diagnostics exist.

A surrogate should not be judged by path RMSE alone. It needs:

- likelihood regret
- filtered-state regret
- posterior shift diagnostics

5. Solve the hardest steady-state bottlenecks.

If the large OBC/QMIPF-style models remain fragile, the methodology will read as selectively demonstrated.

### Bottom line

This is now a credible research platform.

It is not yet a fully convincing methodology.

The strongest current claim is:

- a serious, tested ROM1/FOM switching infrastructure exists, with sparse-tree SEP and growing methodological diagnostics

The strongest claim that is not yet justified is:

- this infrastructure already demonstrates a superior approximation methodology for nonlinear DSGE estimation on hard models

That stronger claim will only be credible once the regret diagnostics stay favorable on the hard models that still strain the steady-state and OBC layers.
