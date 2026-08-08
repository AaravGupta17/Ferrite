# Stage 21 — Website

## Bottom Line Up Front

The website is the public face beyond the README: what Ferrite is, how to use it, and — most importantly — the benchmark dashboard that proves it. **Done when:** a visitor lands, understands the project in one minute, and can build the demo from the docs.

A website is a distribution channel, not a feature of the runtime. Build it only when the project has results worth showing (demo + benchmark table from Stages 8 and 19). It should not gate anything in the codebase.

## Deliverables

- Landing page
- Documentation site
- Benchmark dashboard
- Roadmap
- Blog

## How to Proceed

1. **Landing page = the README, one screen.** Project name, one-line pitch, architecture diagram, three headline numbers (e.g., zero-dependency, 97% MNIST, ~13.4× AVX2), link to docs and demo. Nothing else. Cut anything that does not survive a ten-second glance.
2. **Static site, not a framework.** A static generator (or plain HTML/Markdown) served from a CI release is enough. No build pipeline should be required to ship docs. The site content is generated from the repo's Markdown so it cannot drift from the source.
3. **Benchmark dashboard shows reproducible numbers.** The Stage 8 table, rendered, with the environment header and a link to the benchmark commands. Live-updating is optional and a maintenance burden — a table updated per release is honest and cheap.
4. **Roadmap is `temps/roadmap.md` rendered.** The public roadmap and the internal one are the same document. If they differ, the site is wrong.
5. **Blog only if there is something to say.** A post per release or per milestone (the AVX2 speedup story, the quant accuracy tradeoff) is content. A blog with placeholder posts is a liability. Publish less, publish real.
6. **Hosting is whatever works.** GitHub Pages is the zero-cost default and integrates with CI. Custom domains and analytics are polish — skip until a consumer exists.
7. **The site must not be a second documentation set.** It links to `explain.md`, the API headers, tutorials, and the FAQ (Stage 19). Duplicating content creates drift; referencing the canonical source prevents it.

**Verify.** A fresh visitor follows the site to a working build of the demo without touching the repo. The benchmark numbers on the site match the committed benchmark results.

**Do not** build the site before the demo and benchmarks exist. A website advertising empty claims is the opposite of this project's honesty-over-polish rule.
