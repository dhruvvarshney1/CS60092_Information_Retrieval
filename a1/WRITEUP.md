# Programming Assignment I — Methodology Writeup

**Course:** Information Retrieval
**Task:** Preprocessing, indexing and Boolean search on the Cranfield collection
**Group:** group1
**Language:** Python 3 (standard library only — no IR library used)

---

## 1. Overview

The system has three stages, each a separate program:

```
cran.all ──▶ preprocess.py ──▶ group1_processed.all ──▶ build_index.py ──▶ group1_cran.index
                                                                                  │
                                          "boundary AND layer" ──▶ boolean_search.py ──▶ docid list
```

The query text goes through exactly the same preprocessing functions as the
documents, which is what makes the query vocabulary and the index vocabulary
comparable — if `layers` in a document becomes `layer`, the query word `layers`
must become `layer` too.

---

## 2. Parsing the collection

`cran.all` is a flat file with four tags per record: `.I` (docid), `.T` (title),
`.A` (author), `.B` (affiliation / bibliographic reference), `.W` (abstract).
`read_collection()` is a small state machine over the lines: a line matching
`^\.[IABTW]` switches the current field, any other line is appended to the
buffer only when the current field is `.T` or `.W`. The `.A` and `.B` fields are
discarded as the assignment requires. The parser is a generator, so one document
is in memory at a time rather than the whole 1.6 MB file.

One property of Cranfield worth noting: the title is repeated as the first
sentence of the abstract. We deliberately do not de-duplicate it — the repetition
gives title words a natural weight of 2 in the token stream, which is useful for
the term-frequency work of later assignments and cannot affect a Boolean index,
whose postings are duplicate-free anyway.

---

## 3. Preprocessing

Four independent functions, as required, integrated by `preprocess_document()`:

| Step | Function | Description |
|---|---|---|
| i | `tokenize(text)` | text → list of raw tokens |
| ii | `stem_tokens(tokens)` | tokens → Porter stems |
| iii | `remove_stopwords(tokens, sw)` | drops stop words |
| iv | `normalize(tokens)` | case folding, punctuation stripping, compound splitting |

### 3.1 Order of the steps

The pipeline runs

```
tokenize → normalize → stop-word removal → stemming
```

The reasoning:

* **Normalization before stop-word removal.** The stop list contains lower-case
  surface forms (`the`, `and`, `about`). A raw token `The` or `and,` would not
  match; after case folding and punctuation stripping it does.
* **Stop-word removal before stemming.** The stop list is a list of *unstemmed*
  words. Stemming first would turn `having` into `have` and `only` into `onli`,
  and the latter would no longer match the list. Removing first also means the
  stemmer is called ~100,000 fewer times.
* An optional second pass (`--stem-stopwords`) filters again after stemming,
  against the *stemmed* stop list. This catches inflected stop words such as
  `using → use`. It is off by default because the supplied list contains content
  words such as `system`, `computer`, `interest` and `fill`, and a stemmed match
  on those would silently remove legitimate terms (`filling`, `detailed`).

### 3.2 Tokenization

A token is a maximal run of letters and digits, optionally joined by a single
internal hyphen, apostrophe, dot or slash:

```
[A-Za-z0-9]+(?:[-'./][A-Za-z0-9]+)*
```

Everything else — spaces, commas, parentheses, the slashes Cranfield uses for
emphasis (`/destalling/`), mathematical symbols — acts as a separator. Keeping
the internal joiners at this stage rather than splitting immediately means the
normalizer, not the tokenizer, decides what to do with `boundary-layer` and
`0.5`, so each decision lives in exactly one place.

### 3.3 Normalization

Applied in this order to each token:

1. **Case folding** — `Wing`, `WING`, `wing` → `wing`.
2. **Apostrophe deletion** — `reynolds's` → `reynoldss` → (rule 4) `reynolds`.
3. **Compound splitting** — a hyphen or slash splits the token into its parts:
   `boundary-layer-control` → `boundary`, `layer`, `control`;
   `and/or` → `and`, `or`. This is the right choice for a Boolean system: a user
   who searches for `layer` should retrieve a document that only says
   `boundary-layer`. The unsplit compound is not indexed separately, so the
   vocabulary does not grow.
4. **Residual punctuation stripping** — any non-alphanumeric character left
   (e.g. the dot in `fig.`) is deleted.
5. **Numeric tokens dropped** — in this collection pure numbers are years, page
   numbers, figure numbers and measurement values (`1958`, `324`, `0.5`). They
   add ~2,000 useless index terms and are never sensible Boolean query words.
   `--keep-numbers` restores them. Alphanumeric mixtures (`b747`, `naca0012`)
   are kept because they are genuine identifiers.
6. **Single characters dropped** — the leftovers of formulae (`x`, `y`, `n`).

This function is deliberately the one place where the "same treatment for
documents and queries" guarantee is enforced; `boolean_search.py` imports it
rather than reimplementing anything.

### 3.4 Stop-word removal

The 358-word list is read into a Python `set` (hashed, so O(1) per lookup) and
normalized with the same rules as the documents. Removal deletes 100,424 tokens —
43% of the corpus — while costing only 269 vocabulary types, which is exactly the
expected profile: stop words are very few types occurring very often.

### 3.5 Stemming

`porter_stemmer.py` is a from-scratch implementation of Porter's five-step
suffix-stripping algorithm. The core is the measure *m* of a stem (the number of
vowel–consonant sequences) together with the tests `_cons`, `_vowelinstem`,
`_doublec`, `_cvc`, on which the conditions of every rule are built. The steps
run in order: 1a/1b/1c (plurals, `-ed`/`-ing`, terminal `y`), 2 and 3 (double
suffixes → single, `-ical`, `-ful`, `-ness`), 4 (removal of `-ant`, `-ence`, …
when *m* > 1), 5a/5b (final `-e`, double `-ll`).

Verification: the implementation reproduces Porter's published output for his
sample vocabulary (`python3 porter_stemmer.py` prints the table:
`caresses → caress`, `ponies → poni`, `agreed → agre`, `hopping → hop`,
`formalize → formal`, `controll → control`, and so on).

Results are memoised in a dictionary. The corpus has 130,830 tokens but only
6,988 distinct ones at that point, so the actual stemmer runs about 19 times
fewer than the naive call count.

### 3.6 Attrition through the pipeline

| Stage | Tokens | Distinct types |
|---|---:|---:|
| Raw tokens | 236,516 | 9,283 |
| After normalization | 231,254 | 7,257 |
| After stop-word removal | 130,830 | 6,988 |
| After stemming | 130,830 | **4,334** |

Stemming conflates 6,988 word forms into 4,334 stems — a 38% reduction in
vocabulary at no cost in tokens, which is the whole point of it.

---

## 4. Indexing

`build_index.py` reads `group1_processed.all` and builds a hash table
`term → list of docids`.

Because documents are read in ascending docid order, a posting list is already
sorted by construction; no sorting of postings is needed. Duplicates within a
document are suppressed in O(1) with a `last[term]` dictionary that records the
last docid appended for each term — cheaper than membership testing on the list
and cheaper than building a set per term.

Writing costs one sort of the 4,334 terms (`O(V log V)`), which produces the
required lexicographical order. Total construction is therefore
**O(N + V log V)** in the number of tokens N, and the whole index is built in
under a second.

Index statistics:

| Quantity | Value |
|---|---:|
| Vocabulary size | 4,334 |
| Maximum docid | 1400 |
| Total postings | 77,001 |
| Mean postings per term | 17.8 |
| Largest posting list | `flow` (730 docs) |
| Index file size | 348 KB |

The header line of the file is `4334, 1400`, exactly the "vocabulary size,
maximum docid" pair specified in the assignment.

---

## 5. Boolean search

### 5.1 Query handling

`parse_query()` accepts `<word> AND <word>`, `<word> OR <word>`, a single word,
and — as an extension beyond the requirement — `<word> AND NOT <word>`.
Operators are recognised case-insensitively. Each query word is pushed through
`tokenize → normalize → (stop-word check) → stem`, so `Laws` becomes `law` and
`aeroelastic` becomes `aeroelast`. A query word that is a stop word, or that
normalizes away entirely, produces an explicit error rather than a silent empty
result.

### 5.2 Reaching the postings — search over the index file

The assignment asks for the search to be performed over the index file, so the
default mode does exactly that, without loading the file. Since the terms are
written in lexicographical order, the file supports **binary search on byte
offsets**: seek to the midpoint, discard the partial line, read the next
complete line, compare its term with the target, and halve the window. Narrowing
stops at a 4 KB window, below which one sequential read is cheaper than more
seeks. The invariant maintained is that the low boundary is always the start of a
line whose term is smaller than the target, so the short final scan cannot
overshoot.

Cost: **O(log F)** seeks — about 7 for this file — and roughly 4 KB read, with
no start-up cost at all. The alternative `--mode memory` loads the whole index
into a hash table (O(1) lookup, ~90 ms start-up) and is the better choice when
many queries are answered in one run. Both were checked to return identical
posting lists for all 4,334 terms.

### 5.3 Merging the posting lists

Both posting lists are sorted, so both operators are answered by a single
simultaneous walk of the two lists — never by a nested loop and never by
converting to a Python `set` (which would lose the sorted order and force a
re-sort of the output).

* **OR** — `union()`: advance whichever cursor points at the smaller docid and
  emit it; on equality emit once and advance both. **O(m + n)**, output sorted.
* **AND** — `intersect()`: the same walk, emitting only on equality.
  **O(m + n)**.
* **AND NOT** — `difference()`: emit from the left list only when the right list
  has moved past it. **O(m + n)**.

### 5.4 Bonus: skip pointers

The plain merge wastes work when the two lists are very unequal in length —
`flow` (730 postings) against `slipstream` (15) forces the long list to be
stepped through one posting at a time. `intersect_with_skips()` adds implicit
skip pointers of length √n: when the cursor on one list is behind the other
list's current docid, it jumps √n postings at a time while the value at the
landing point is still smaller than the target, then falls back to single steps.
√n is the classic optimum — it balances the number of skips against the amount
of list left unskippable.

Measured over 20,000 random term pairs from the actual index:

| Merge | Comparisons (all pairs) | Comparisons (pairs with ≥20× length ratio) |
|---|---:|---:|
| Linear | 479,144 | 223,708 |
| Skip pointers | 237,479 (**−50%**) | 52,046 (**−77%**) |

The saving is exactly where theory predicts: the more skewed the two lists, the
more the skips pay. On wall-clock time over this small collection the two are a
wash (≈2 µs per query either way) because the average posting list is only 18
entries long and Python's interpreter overhead per operation dominates the
comparison count; the algorithmic saving would translate into real time on a
web-scale collection or in a compiled language. `--no-skip` selects the plain
merge for comparison. Both were verified to produce identical results to a
set-based reference intersection on 5,000 random term pairs.

### 5.5 Complexity summary

| Operation | Cost |
|---|---|
| Preprocessing the collection | O(N) tokens, one pass, streaming |
| Index construction | O(N + V log V) |
| Term lookup, disk mode | O(log F) seeks |
| Term lookup, memory mode | O(1) after O(V) load |
| AND / OR / AND NOT merge | O(m + n) |
| AND with skips | O(m + n) worst case, ≈O(m + √n · m) when n ≫ m |

---

## 6. Test queries and results

`test_queries.txt` holds 16 queries; two of them (`similarity AND laws`,
`structural AND aeroelastic`) are taken from queries 001 and 002 of `cran.qry`.
The full docid lists are in `group1_boolean_results.txt`. Summary:

| Query | Stemmed | df₁ | df₂ | Matches |
|---|---|---:|---:|---:|
| similarity AND laws | similar AND law | 149 | 53 | 17 |
| aeroelastic AND models | aeroelast AND model | 18 | 177 | 9 |
| heated AND aircraft | heat AND aircraft | 306 | 71 | 8 |
| structural AND aeroelastic | structur AND aeroelast | 107 | 18 | 9 |
| high OR speed | high OR speed | 236 | 292 | 407 |
| boundary AND layer | boundari AND layer | 469 | 414 | 370 |
| slipstream AND wing | slipstream AND wing | 15 | 226 | 11 |
| shock AND waves | shock AND wave | 240 | 210 | 149 |
| supersonic OR hypersonic | superson OR hyperson | 270 | 170 | 412 |
| turbulent AND flows | turbul AND flow | 144 | 730 | 99 |
| buckling AND shells | buckl AND shell | 127 | 106 | 57 |
| pressure AND distributions | pressur AND distribut | 552 | 361 | 217 |
| cylinders OR spheres | cylind OR sphere | 178 | 39 | 206 |
| aerodynamics AND experimental | aerodynam AND experiment | 179 | 339 | 49 |
| xyzzy AND flow | xyzzi AND flow | 0 | 730 | 0 |
| flow AND NOT turbulent | flow AND NOT turbul | 730 | 144 | 631 |

Observations:

* Stemming is doing its job — `laws`, `models`, `waves`, `flows`, `shells`,
  `distributions`, `cylinders` all match their singular forms in the documents,
  and `aeroelastic → aeroelast` matches `aeroelasticity` as well.
* Query 1 of `cran.qry` asks about similarity laws for aeroelastic models of
  heated high-speed aircraft. `similarity AND laws` returns 17 documents, among
  them docids 13, 56, 486 — and 13, 56 and 486 are all listed as relevant to
  query 1 in `cran_rel.txt`. The Boolean model finds them, but with no ranking
  it gives no way to tell them from the other 14, which is precisely the
  limitation that motivates the ranked models of the next assignment.
* `xyzzy AND flow` shows the missing-term case: an absent term yields an empty
  posting list and an empty result, with no error.

---

## 7. Limitations and possible extensions

* Only two-word queries with a single operator are supported, as specified.
  General Boolean expressions would need a proper parser (shunting-yard into a
  query tree) and an evaluator that orders the merges by increasing document
  frequency, which is the standard optimisation for multi-way ANDs.
* Skip pointers are computed on the fly from the list length; a production index
  would store them explicitly alongside compressed postings (variable-byte or
  gamma coding of docid gaps would cut this index well below its 348 KB).
* No phrase queries — that needs positional postings, which the current
  `.S` token stream already contains enough information to build.
* The Boolean model returns an unranked set. Term weighting (tf-idf) and the
  vector space model are the natural next step, and `cran_rel.txt` then allows
  precision/recall evaluation.

---

## 8. Program listing

The submitted programs, in the order they run:

1. `preprocess.py` — collection parser, `tokenize`, `normalize`,
   `remove_stopwords`, `stem_tokens`, pipeline, CLI.
2. `porter_stemmer.py` — `PorterStemmer` class (steps 1a–5b), memoised `stem()`,
   self-test against Porter's sample vocabulary.
3. `build_index.py` — `build_index`, `write_index`, CLI.
4. `boolean_search.py` — `process_query_term`, `parse_query`, `Index`
   (binary-search and in-memory lookup), `intersect`, `intersect_with_skips`,
   `union`, `difference`, `run_query`, batch / single / interactive CLI.
5. `run_all.sh` — end-to-end driver.

All source files are commented at function level; see `README.md` for exact
commands and flags.
