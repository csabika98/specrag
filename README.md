# specrag

Retrieval over the UK Building Regulations Approved Documents, with an
evaluation harness that reports real numbers.

Most portfolio RAG projects are a notebook that answers one question
convincingly. This one measures whether the right paragraph was retrieved at
all, on a question set with known answers, and publishes the score.

[![CI](https://github.com/csabika98/specrag/actions/workflows/ci.yml/badge.svg)](https://github.com/csabika98/specrag/actions/workflows/ci.yml)

---

## Current scores

Approved Document K (*protection from falling, collision and impact*),
111 chunks, `all-MiniLM-L6-v2`, 18 golden questions:

| Metric | Score |
|---|---|
| hit@1 | 0.833 |
| hit@3 | 0.889 |
| hit@5 | 1.000 |
| MRR | 0.880 |
| recall@5 | 1.000 |

Reproduce with `python -m specrag.ingest --parts K && python -m specrag.eval`.

These are retrieval scores, not answer scores. Nothing here generates text yet
— see [Roadmap](#roadmap).

---

## Why measure retrieval separately

Retrieval is the half of a RAG system that fails quietly. A generator handed
the wrong clause still writes a fluent, confident answer, and the failure only
surfaces when someone checks the citation. Scoring retrieval on its own makes
that failure visible, and it has two practical consequences:

- the evaluation needs no API key and no model download, so **CI runs it on
  every pull request**
- a regression points at a chunking or embedding change, not at a model
  release

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt

python -m specrag.ingest --parts K     # download, parse, chunk, embed, index
python -m specrag.eval                 # score retrieval against the golden set
```

Add `--stub` to either command to use a deterministic hashing embedder instead
of the real model: no download, meaningless retrieval quality, useful for
checking the pipeline runs.

Runs on CPU. Developed on a 2014 dual-core i5 with 1.2GB of free RAM: the
model loads in ~23s and embedding 111 chunks takes ~5s.

---

## How it works

```
gov.uk Content API  ->  PDF  ->  cleaned text  ->  clauses  ->  chunks  ->  FAISS
                                                                              |
                                              golden questions  ->  metrics <-+
```

### The corpus

The Approved Documents are the statutory guidance to the Building Regulations
for England. They are fetched through the gov.uk Content API rather than
scraped, because the API gives attachment URLs directly and does not move when
the page template changes.

### Cleaning (`ingest/parse.py`)

Three kinds of noise wreck retrieval if left in:

- **Watermark.** `ONLINE VERSION` is stamped roughly four times per page —
  261 occurrences in Approved Document K alone.
- **Running headers.** These cannot be matched on the phrase
  `Approved Document K, 2013 edition`, because the PDF content stream splits
  the header mid-word; extraction yields fragments like `t K, 2013 edition`,
  sometimes doubled on one line. A line is treated as a header when removing
  the `<part>, <year> edition` fragments and their surrounding boilerplate
  leaves under four characters. That keeps the cover text
  (`2013 edition - for use in England`) and the contents heading
  (`Main changes in the 2013 edition`), which are body text. Measured on Part
  K: 62 header lines removed, the 2 genuine mentions kept.
- **Ligatures.** The typesetter emits `ﬁ`, `ﬂ`, `ﬃ`. Without NFKC folding, a
  query for "profiles" never matches the text's "proﬁles".

### Chunking (`ingest/chunk.py`)

Approved Documents are written as numbered clauses, so chunks are built from
clauses rather than a fixed character window. That is what lets an answer cite
`Approved Document K, paragraph 1.34` instead of "somewhere on page 31".

Two details that turned out to matter:

- **A clause number must be followed by prose.** Dimension tables contain
  lines like `2.5 m clear width`, which a naive `^\d+\.\d+` reads as clause
  2.5.
- **A line is a section heading only when a clause follows it.** PDF
  extraction pulls loose labels out of diagrams — `Going`, `Open riser`,
  `Nosing` — and those are followed by more labels, never by a clause.

The heading is prepended to the text that gets embedded. Clause 1.2 reads
*"Measure the rise and going as shown in Diagram 1.1"* and never mentions
stairs; without its heading it cannot be retrieved for a query about stairs.

### Index (`index/store.py`)

A flat FAISS inner-product index over unit-norm vectors, so the score is
cosine similarity. A few thousand chunks does not justify an approximate
index, which would add tuning and failure modes for no measurable gain.

The store depends on an `Embedder` protocol rather than on
sentence-transformers directly. That is what lets the whole retrieval path run
in tests against a deterministic hashing stub. The stub uses `crc32`, not the
builtin `hash()`, which is salted per process and would make an index built in
one run unusable by a query in the next.

---

## Development

```bash
pytest                      # 107 tests, offline, ~1s
pytest --cov=specrag        # 76% overall; 99-100% on parsing, chunking, metrics
ruff check src tests
ruff format --check src tests
```

CI runs lint, format check and the suite on Python 3.11, 3.12 and 3.13.

Coverage is 76% rather than higher because the two CLI entry points and the
network paths in `ingest/fetch.py` are not unit tested. The pure logic they
call is: the API payload parsers, the text cleaner, the chunker, the store and
every metric.

---

## Limitations

Known and not yet addressed:

- **Diagram text is noise.** Label text extracted from diagrams ends up in the
  body of whichever clause precedes it. It is not promoted to a heading, but
  it still dilutes the embedding.
- **Tables are flattened.** The rise-and-going table becomes a run of numbers
  with no column structure, so numeric lookups work less well than prose ones.
- **One part indexed.** The scores above are Part K only. Cross-part retrieval
  is untested, and the Approved Documents cross-reference each other heavily.
- **18 golden questions is small.** Enough to catch a regression, not enough
  for a confident absolute number. A single question moving changes hit@1 by
  0.056.
- **Questions were written from the clauses.** They are therefore biased
  towards text that exists, and say nothing about how the system handles a
  question the corpus cannot answer.

---

## Roadmap

- [x] Ingestion, cleaning, clause-aware chunking
- [x] FAISS index with a model-free test path
- [x] Retrieval metrics and a golden question set
- [ ] Generation with citations, behind a pluggable provider
- [ ] LoRA fine-tune of the embedding model on domain pairs, scored
      before/after against this same harness
- [ ] All parts indexed, with cross-reference handling

---

## Licence and attribution

Code: MIT.

The Approved Documents are © Crown copyright, published by the Ministry of
Housing, Communities and Local Government under the
[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
No corpus text is committed to this repository; `python -m specrag.ingest`
downloads it.

This project is not affiliated with or endorsed by any UK government body, and
nothing it produces is a substitute for the published regulations.
