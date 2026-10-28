# Conference submission checklist

## Completed

- [x] Central claim is a validation architecture, not an unproved speedup.
- [x] Abstract is 142 words and states the main numerical findings and claim boundary.
- [x] Introduction states one central contribution and reports the main results before the literature discussion.
- [x] Complete 23-equation model scope is stated consistently.
- [x] Positive and negative numerical findings are both reported.
- [x] Production-horizon likelihood, posterior, and speed claims are explicitly excluded.
- [x] Archived GPU execution timings are separated from utilization and speedup claims.
- [x] The GPU launcher records continuous utilization, memory, power, temperature, clock, and estimated energy for future evidence runs.
- [x] Every numerical evidence table is generated from machine-readable artifacts.
- [x] Bibliographic metadata and title capitalization have been checked.
- [x] LaTeX build has no warnings, undefined references, or overfull/underfull boxes.
- [x] All 14 rendered pages have been visually checked.
- [x] Focused evidence and profiling tests pass: 130 passed.
- [x] Data/code availability and computational-assistance statements are included.

## Author checks before upload

- [ ] Confirm whether the conference requires anonymous review; replace `Anonymous` only if permitted.
- [ ] Apply the conference template and verify its page, font, abstract, and bibliography limits.
- [ ] Confirm the conference's computational-assistance disclosure wording and submission location.
- [ ] Add the public replication URL only if doing so does not compromise anonymous review.
- [ ] Read the final PDF once without editing, checking author names, dates, and every headline claim.

## Claim boundary

The current draft is suitable as a conference methods paper about auditable
validation and reduced-horizon algorithmic parity. It is not evidence of a
production structural posterior, posterior recovery, or an end-to-end GPU speed
advantage. Those claims remain subject to the hard gates in Table 9.
The archived GPU suite has run-start snapshots only; its zero-utilization values
are not workload measurements. Continuous telemetry is enabled for the next
dedicated-device run but does not replace the same-work timing gate.
