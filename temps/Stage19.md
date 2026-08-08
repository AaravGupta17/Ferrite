# Stage 19 — Documentation

## Bottom Line Up Front

Documentation is a feature, not an afterthought — and it is already this project's stated value. The rule from `temps/documentation.md`: BLUF everything, docs ship with code, honesty over polish. **Done when:** a new contributor reads the README, builds, and runs the demo in under a minute, then reads `explain.md` and understands every subsystem.

Ferrite already follows these standards with `README.md`, `temps/explain.md`, `temps/roadmap.md`, and `temps/documentation.md`. This stage is about completeness and polish, not a rewrite.

## Deliverables

- Professional README
- Architecture guide
- API docs
- Tutorials
- Examples
- FAQ
- Contributing guide
- Coding guidelines

## How to Proceed

1. **README is the front door.** It must answer, in under a minute: what is this, how do I build it, how do I run it, what does it achieve. The current README is a two-line stub — the roadmap P5 items (quickstart, architecture image, benchmark table, demo link, honest limitations) are the concrete plan.
2. **Architecture guide = `explain.md`, kept true.** It already covers every subsystem in dependency order. When code changes, the explainer changes in the same commit — that is the documented rule and the one that must not slip.
3. **API docs come from the headers, not in addition to them.** The header comments are the API docs: one paragraph per public function stating behavior, return, and constraints. Generate a rendered page from headers (doxygen-style) only when a consumer asks; keep the source of truth in the `.h` files.
4. **Tutorials and examples ship as runnable code.** A `demo/` directory (roadmap P2) with a README is the tutorial: trained → exported → run → accuracy and latency. Examples are real, small programs in `tests/` or `examples/`, not pseudo-code.
5. **FAQ grows from real questions.** Collect actual questions from issues and conversations. Two-dozen real entries beat a hundred invented ones. Keep answers short and link to `explain.md` for depth.
6. **Contributing guide is short and concrete.** It should reference this file and `AGENTS.md`: how to build, how to test (one `test_*` binary), the commit style, the one-subsystem-per-commit rule, and the docs-ship-with-code rule.
7. **Coding guidelines are the ones in AGENTS.md.** Naming, contracts, invariants. They are only real if review enforces them — the contributing guide says so explicitly.
8. **Audit against the standards.** Before finishing this stage, run the review checklist in `temps/documentation.md` Section 5 on every doc: every claim true, every number real, every limitation stated.

**Verify.** The cold-read test: someone with no context builds and runs the demo in under a minute using only the README. Then they read `explain.md` top to bottom and can trace one inference through every layer.

**Do not** pad. `documentation.md` is explicit: short is a feature. A doc that repeats the code is noise; a doc that explains *why* earns its place.
