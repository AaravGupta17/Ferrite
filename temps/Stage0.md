# Stage 0 — Foundation (Current)

## Bottom Line Up Front

Foundation is the base everything else builds on: a decision about what Ferrite is, a folder structure, a working build, and shared infrastructure (logging, errors, config). **Done when:** a clean clone builds with one documented command and all tests are green.

This is the stage Ferrite is in today. Most of the foundation already exists — see `temps/roadmap.md` Phase 0 for the remaining hygiene and build work.

## Deliverables

- Decide project vision and architecture
- Design folder structure
- CMake build system
- Logging system
- Error handling framework
- Configuration system
- Coding standards
- Documentation skeleton

## How to Proceed

1. **Vision first.** Write the project's goal in one paragraph and lock the acceptance criteria in `temps/roadmap.md` Section 2 before writing code. Every later decision tests against this paragraph.
2. **Architecture.** Adopt the layered dependency rule already in place: `importer/` → `graph/` → `planner/` → `runtime/` → `ops/` + `simd/` + `core/`. Layers depend downward only. Never let an upper layer be imported by a lower one.
3. **Folder structure.** Keep one directory per subsystem with `module.c` + `module.h`. Do not invent new top-level folders without a reason. `temps/` holds working docs.
4. **Build system.** Close the CMake gap (roadmap Phase 0). Ship a `CMakeLists.txt` that produces the same binaries as the Makefile, so CLion on Windows and `make` in WSL agree. One documented command must build everything.
5. **Logging and errors.** Start small. A `fe_log` helper and the existing `FeStatus` codes are enough. Add levels and context only when debugging demands it.
6. **Configuration.** Do not build config machinery before there is something to configure. Constants in headers are fine at this stage.
7. **Coding standards.** Write them into `AGENTS.md`: naming (`fe_` prefix), header guards, `FeStatus` returns, kernel contract, commit style. Standards only count if enforced by review.
8. **Documentation skeleton.** Stand up `README.md`, `temps/explain.md`, `temps/roadmap.md`, `temps/documentation.md`. Content can be thin; the structure must be stable.

**Do not** start Stage 1 until a clean clone → one command → all tests green. A shaky build poisons every later stage.
