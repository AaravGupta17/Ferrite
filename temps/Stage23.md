# Stage 23 — Community

## Bottom Line Up Front

Community is how Ferrite gets contributors and feedback: issue triage, discussions, contribution templates, and release notes. It is process, not code. **Done when:** a new contributor can find a task, open a PR, and have it reviewed under the project's rules.

Community building presupposes the infrastructure (Stage 20) and the contributing guide (Stage 19). Do not open the doors before the house is safe to walk around in.

## Deliverables

- GitHub Issues
- Discussions
- Discord
- Good First Issues
- Contribution templates
- Release notes

## How to Proceed

1. **Issue templates set the contract.** Two templates: bug (what you did, what happened, expected, environment) and feature (problem, suggested approach, scope). A template that asks for the environment catches the Windows vs. Linux build issues before they eat a day.
2. **Discussions are for ideas, issues are for work.** Enable GitHub Discussions for "should we?" conversations; issues track concrete, actionable work. Enforce the split in the contributing guide, or issues become a graveyard of ideas.
3. **Discord only after there is a community.** A server with two people is maintenance. Spin it up when the Discussions have active traffic and someone volunteers to moderate. It is a channel, not a deliverable.
4. **Good First Issues are curated, not labeled.** Every entry has: a clear scope, the files to touch, the test to run, and an estimate. The label alone does nothing; the description does. Draw them from the Stage roadmap's gaps (e.g., a missing dispatch case, a planner integration).
5. **Contribution templates = the contributing guide, rendered.** PR checklist: tests run (list them), docs updated in the same commit, one subsystem, commit message follows `feat(scope):`. This mirrors AGENTS.md — the humans and the agents follow the same rules.
6. **Release notes come from the commit log.** The one-subsystem-per-commit style makes them write themselves: group by subsystem, note the user-facing change, link the benchmark numbers. Auto-generate from the tag diff (Stage 20), then edit for voice.
7. **Triage discipline.** Issues are triaged on arrival: bug, feature, question, or won't-fix — labeled and responded to within days. A stale issue is closed or re-labeled. An unmaintained backlog trains people not to report.

**Verify.** A newcomer opens a Good First Issue, gets it merged, and the PR followed the template without hand-holding. The release notes for the latest tag are complete and accurate.

**Do not** open the community channels before CI exists. Nothing kills a community faster than an untested repo that rejects PRs for missing sanitizer passes.
