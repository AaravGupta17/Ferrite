# Stage 20 — Infrastructure

## Bottom Line Up Front

Infrastructure makes the project safe to change: CI that builds and tests every commit, formatting and static analysis that keep the code consistent, and versioned releases. **Done when:** a pull request runs the full test suite automatically, and a tagged release is a one-command, reproducible step.

Ferrite currently has a Makefile, a Windows/CLion dev setup, and no CI. The roadmap's Phase 0 CMake decision gates most of this stage.

## Deliverables

- GitHub Actions
- Unit testing CI
- Formatting
- clang-tidy
- cppcheck
- Code coverage
- Releases
- Versioning

## How to Proceed

1. **CI runs the existing tests first.** A GitHub Actions workflow that does `make all && ./test_*` on Ubuntu with `gcc` is the foundation. It is valuable the day it lands; expand it after. The runner is Linux — the Makefile's home — so CI and local Unix builds agree.
2. **Test every sanitizer, not just default.** CI matrix: build with `-fsanitize=address,undefined`, build the AVX2 targets with `-O3 -mavx2 -mfma`, and run the whole suite under each. A failure in any configuration fails the PR.
3. **Formatting is a contract, not a preference.** Pick `clang-format` with one `.clang-format` and run it in CI (`--dry-run --Werror`). The style is already consistent in the codebase — lock it in before it drifts. No format wars: the config decides.
4. **clang-tidy for correctness rules.** Enable the checks that catch real bugs (null deref, bounds, uninitialized reads). Keep the check list small and enforced; a check that fires everywhere gets disabled and ignored.
5. **cppcheck as a second pass.** It finds leaks and lifetime issues that tidy misses. Run it in CI on the same code. Report count is zero on the core; warnings are bugs or noise — triage each.
6. **Coverage gates the suite, not the repo.** `-fprofile-arcs -ftest-coverage`, upload a report per PR, and fail on *decreasing* coverage on the diff. Stage 7 decides what is worth covering.
7. **Versioning is semver from the first release.** `MAJOR.MINOR.PATCH` with a `FE_FERRITE_VERSION` macro (Stage 18). The git tag is the source of truth; the header mirrors it. `0.x` until the C API stabilizes.
8. **Releases are reproducible.** A release = a tagged commit + the CI artifacts. Notes are generated from the commit log (one subsystem per commit makes this clean). No manual build-and-upload rituals.

**Verify.** The Actions badge is green on the default branch. A PR with a failing test, a format violation, or a coverage drop is blocked. A `v0.1.0` tag produces artifacts from a clean runner.

**Do not** add a CI gate you cannot fix in minutes. Every CI failure trains people to ignore CI — keep the check set honest.
