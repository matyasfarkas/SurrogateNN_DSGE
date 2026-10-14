# Referee Review and Revision Record

Paper: `Real Determinacy Screens for DSGE Estimation`

Author: Matyas Farkas

## Referee Report

### Recommendation

Accept as a short methods note after minor author metadata updates. The paper makes a narrow but useful econometric point: a numerical complex-QZ extraction failure is not a valid posterior-support restriction in a real DSGE model. The result is algebraically correct under the stated assumptions and economically relevant because the old rule excludes a large share of otherwise valid HLT prior draws.

### Summary

The paper studies determinacy screening in Bayesian DSGE estimation. The posterior support is defined by a stable-root count, a graph condition, and a residual check. The paper proves that complex stable generalized eigenvalues of a real pencil do not imply complex economic dynamics. They occur in conjugate pairs and span real invariant planes. Therefore, if the stable subspace has the right dimension and satisfies the graph condition, the implied policy rule is real. The paper then reports a one-million-draw HLT audit in which a complex-QZ rule rejects 464,685 draws that real QZ accepts.

### Major Comments

1. The original version did not frame the issue as an econometric support problem strongly enough. This matters because a wrong determinacy classifier changes the posterior, not just a numerical diagnostic.

2. The proof needed to separate three objects: complex roots, complex bases, and complex-valued recovered policy matrices. Complex roots can be valid; a complex basis is harmless; a complex recovered policy matrix is a numerical warning unless the root count or graph condition fails.

3. The generalized eigenvalue statement needed to acknowledge infinite roots. DSGE pencils can have infinite generalized eigenvalues when the denominator of the generalized eigenvalue ratio is zero. Those roots are outside the finite stable set and should not be counted as stable finite roots.

4. The Cayley result was at risk of being overclaimed. The audit supports equivalence on the tested HLT loose-prior region; it does not prove global equivalence for all DSGE pencils.

5. The doubling result needed sharper interpretation. Doubling is useful as a speed device, candidate solver, or warm start, but it is not a final determinacy classifier because it accepts non-unique cases in the audit.

6. The paper needed an explicit AI-use disclosure. If the paper goes into a dissertation, journal submission, or replication archive, the disclosure should state that AI assisted drafting/editing and that the author remains responsible for all claims.

### Minor Comments

1. Add the author's institutional affiliation and email before circulation.

2. If submitted as a standalone note, add a small reproducibility paragraph pointing to the exact benchmark script and commit hash.

3. If submitted as an appendix to the main paper, keep the current three-page format and cite it from the numerical-methods section.

4. The title is clear, but a slightly more specific journal-style title could be `Determinacy Screens and Posterior Support in DSGE Estimation`.

## Revision Response

### Response to Major Comment 1

The revised paper now opens with the Bayesian posterior and the determinacy indicator. It states that an incorrect numerical definition of the support changes the estimand.

### Response to Major Comment 2

The proposition now distinguishes stable finite complex roots, complex-QZ implementation output, and the real policy matrix implied by the stable deflating subspace.

### Response to Major Comment 3

The econometric object section now states that infinite generalized roots may arise and that the stable-root count is over finite generalized eigenvalues with modulus below one.

### Response to Major Comment 4

The Cayley language was weakened. The paper now says the real-part Cayley screen matches real QZ in the HLT audit, not that it is globally equivalent.

### Response to Major Comment 5

The paper now explicitly classifies doubling as a speed device that requires verification, because it accepts 84,606 real-QZ non-unique cases in the audit.

### Response to Major Comment 6

The paper now includes an `Acknowledgments and AI Use Disclosure` section crediting OpenAI language models for drafting/editing assistance while assigning responsibility to the author.

## Publication Readiness Checklist

- The central claim appears in the abstract and introduction.
- The econometric object is defined as a posterior-support restriction.
- The theorem states assumptions explicitly.
- The proof avoids overclaiming and covers conjugate pairs, realification, graph representation, and damped real cycles.
- The HLT audit table reports counts and shares.
- The Cayley statement is audit-specific rather than universal.
- The doubling statement is conservative.
- The compiled PDF is three pages.
- Remaining author-only item: add affiliation/email if needed for submission.
