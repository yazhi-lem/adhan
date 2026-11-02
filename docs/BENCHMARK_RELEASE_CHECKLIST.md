# Tamil tokenizer benchmark: release checklist

Part 4 of [#74](https://github.com/yazhi-lem/adhan/issues/74). Nothing here is done by code.
Tick each line, with a link as evidence, before the benchmark folder and post are announced.

## Before publishing

- [ ] Parts 1-3 merged (corpus builder, harness, reproduce script)
- [ ] Modern and Tanglish sources chosen; licence, URL and line count recorded in the spec
- [ ] No unlicensed text in the repo (check `git ls-files data eval_manifests`)
- [ ] Spec and manifest committed; `scripts/build_eval_corpus.py --verify` passes
- [ ] Native reviewer has read the Tamil and Tanglish samples and the post's Tamil text
- [ ] Results generated with `scripts/reproduce_benchmark.sh` and committed
- [ ] Clean-checkout test: fresh clone, install, `scripts/reproduce_benchmark.sh --check` passes
- [ ] Every number in the post is copied from `docs/TOKENIZER_BENCH_RESULTS.md`; no `[[FILL` left
- [ ] Gemma and Llama not used anywhere in the results
- [ ] Sovereignty rule respected: no corpus text sent to Claude or any foreign API

## Decisions that are not ours alone

- [ ] Valavan K agrees to make the benchmark public (about a week before the date)
- [ ] Public home: folder in this repo (`benchmarks/tamil-tokenizer`) or a separate repo
- [ ] Where the post is published, and who approves it

## Publishing and handover

- [ ] Remove the draft banner from `benchmarks/tamil-tokenizer/README.md`
- [ ] Publish the post
- [ ] Hand the links to the launch owner (LEM-226) and monitor replies for 48 hours
- [ ] Link the evidence (PRs, tables, post) on #74 and move it to Done after review

## Rules to remember

- Valavan K reviews each PR first.
- Do not change a due date without the lead agreeing.
- If blocked, comment on the issue, tag Valavan K and raise it at the Monday check-in.
