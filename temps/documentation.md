# Ferrite — Documentation Maintenance

## Bottom Line Up Front

Documentation is a feature, not an afterthought. It is how this project reads to outsiders, and it is how the author remembers why decisions were made. Keep it short, current, and honest about limitations. When code changes, the docs change in the same commit.

**Three rules.**

1. **BLUF everything.** Bottom line first, then supporting detail.
2. **Docs ship with code.** No code change without updating its doc — and vice versa.
3. **Honesty over polish.** Document what does not work yet. An honest limitation beats a silent gap.

---

## 1. Documentation Inventory

What exists and what belongs where.

| Document | Location | Purpose | Owner cadence |
|---|---|---|---|
| `README.md` | repo root | Public face. What the project is, quickstart, benchmarks, demo link. | Update at each milestone |
| `temps/explain.md` | `temps/` | Deep technical guide. How every subsystem works and connects. | Update with each subsystem change |
| `temps/roadmap.md` | `temps/` | Where the project is going. Goal, priorities, timeline. | Update when priorities or scope change |
| `documentation.md` | `temps/` | This file. How documentation is maintained. | Revise when standards change |
| Header comments | `*.h` | One-paragraph contract per public function or type. | Same commit as the code |
| Commit messages | git | Why a change exists, scoped to one subsystem. | Every commit |

**Naming.** Markdown files use `snake_case.md` at repo root; scratch/working docs live in `temps/`. Do not create new root docs without a reason — prefer extending the four above.

---

## 2. Writing Standards

Write to the military standard: clear, direct, and short. This is the voice used across `explain.md` and `roadmap.md`.

**Structure.**

- Open every document and every section with a **Bottom Line Up Front**. One to five lines. The reader gets the answer before the explanation.
- Lead with the most important information. Support it with detail, in order of importance.
- Use headings that answer a question or state a fact, not vague labels.

**Language.**

- **Active voice.** "The engine dispatches nodes" — not "nodes are dispatched by the engine."
- **Short sentences and short paragraphs.** One idea each. Break long runs into bullets.
- **Plain words.** No jargon, no filler, no "please note," no "in order to" where "to" works.
- **Action verbs.** "Wire," "dispatch," "ship," "measure."
- **Concrete numbers over claims.** "~13.4× speedup," "512 nodes / 1024 tensors," "L_out = (L − K + 2·pad)/stride + 1."

**Formatting.**

- Use tables for comparisons, lists, and status (like the inventory and state tables above).
- Use **bold** for the bottom-line sentence of each section.
- Use fenced code blocks for code, structs, and terminal output.
- Keep line width reasonable; do not paste auto-wrapped walls of text.

---

## 3. What to Document, Where

**README.md (public face).**
- What Ferrite is, in two lines.
- Quickstart: clone → build → run one command → see output.
- Architecture diagram image.
- Benchmark table and speedup story.
- Link to the demo and the explainer.
- Honest limitations section.

**Explain.md (deep guide).**
- One section per subsystem, in dependency order: `core/`, `graph/`, `ops/`, `simd/`, `planner/`, `runtime/`, `importer/`, `quantization/`, `tools/`, `tests/`.
- Each section: bottom line, key structs and functions, how it connects to the next layer, known gaps.
- File references with line numbers where useful (`runtime/engine.c:17`).

**Roadmap.md (direction).**
- Current state table — update status and gaps as work lands.
- Goal and acceptance criteria — change only when the goal changes.
- Priorities and timeline — update done-when criteria as phases complete.
- Risks and decisions — log decisions as they are made, not after.

**Header comments (`.h` files).**
- One paragraph per public function or type. State what it does, what it returns, and any constraints (alignment, contiguous requirements, ownership).
- Do not comment the obvious inside function bodies. Reserve inline comments for why, not what.

**Commit messages.**
- One subsystem, one commit. Match the existing style: `feat(scope): short summary`.
- Say what changed and why it matters. Do not restate the diff.

---

## 4. Keeping Docs in Sync

**The rule.** A change is not done until its documentation is done.

**Checklist before committing code:**

1. Does the change touch behavior or structure? Update `explain.md` in the same commit.
2. Does it change status, scope, or timeline? Update `roadmap.md`.
3. Does it affect the public face (build, usage, benchmarks)? Update `README.md`.
4. Does it add or change a public function? Update the header comment.
5. Does the commit message state why, not just what?

**Checklist before committing docs:**

1. Is every claim still true against the current code? Verify, do not guess.
2. Are file references and line numbers current?
3. Is every limitation still a limitation? Remove items that are now fixed.
4. Does it still follow BLUF and the writing standards above?

**Anti-patterns to reject:**

- Docs that describe planned behavior that is not implemented.
- Claimed numbers with no benchmark behind them.
- "Unimplemented op" sections that list ops the dispatcher silently drops — the demo must fail loudly instead.
- Doc-only commits that drift from the code (unless the doc is correcting a factual error).

---

## 5. Review Checklist

Use this before considering any doc finished.

- [ ] Opens with a bottom line (document and every major section).
- [ ] Most important information first.
- [ ] Active voice throughout.
- [ ] No jargon, filler, or vague words.
- [ ] Every number is real and reproducible.
- [ ] Every limitation is stated, not hidden.
- [ ] File references and line numbers are current.
- [ ] Matches the voice and format of `explain.md` and `roadmap.md`.

---

## 6. Routine Maintenance

A short, repeatable cycle keeps docs healthy without much effort.

- **Per commit.** Header comments and commit message with the code.
- **Per milestone** (end of each roadmap phase). Update README, explainer, and roadmap in one pass. Mark phases "done when" as complete.
- **Quarterly or on demand.** Read the README cold. If it does not answer "what is this and how do I run it" in under a minute, tighten it.
- **Before sharing.** Run the review checklist in Section 5.

**First-pass work order** (for a doc backlog):

1. Make README truthful and runnable — public face first.
2. Update explainer to match the current code.
3. Move completed roadmap items to "done"; refresh risks.
4. Prune stale limitations and update the state table.

---

## 7. Do / Do Not

**Do.**

- Write the bottom line first.
- Update docs in the same commit as the code.
- State real numbers, real gaps, and real decisions.
- Keep files in their intended locations.

**Do not.**

- Do not add new root-level Markdown files without a reason.
- Do not let docs drift from code — fix within the same commit.
- Do not pad length. Short is a feature.
- Do not hide unsupported features behind silent behavior.
