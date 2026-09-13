# Information Retrieval — Programming Assignment I
## Preprocessing, Indexing and Boolean Search on the Cranfield collection

**Group name:** `group1` (change it with the `--group` flag on every program;
all output files are prefixed with it)

Everything here is written from scratch in Python 3 (standard library only).
No information-retrieval library is used. The Porter stemmer is also our own
implementation of the 1980 algorithm and is verified against Porter's published
sample vocabulary.

---

## 1. Files

### Programs
| File | Purpose |
|---|---|
| `porter_stemmer.py` | Porter stemming algorithm, implemented from scratch. Run it directly to see it stem Porter's sample word list. |
| `preprocess.py` | The four preprocessing functions (tokenization, stemming, stop-word removal, normalization) integrated into one program. Produces `group1_processed.all`. |
| `build_index.py` | Builds the inverted index `group1_cran.index`. |
| `boolean_search.py` | Boolean retrieval (`AND`, `OR`, plus `AND NOT` as an extension) over the index file. |
| `run_all.sh` | Runs the whole pipeline end to end. |

### Inputs
| File | Purpose |
|---|---|
| `cran.all` | The Cranfield collection, 1400 documents (supplied as `cran_all.1400`). |
| `stopwords.txt` | English stop-word list, 358 words. |
| `test_queries.txt` | The test Boolean queries used for the submitted result file. |

### Outputs (submitted)
| File | Contents |
|---|---|
| `group1_processed.all` | Preprocessed collection, `.I` docid / `.S` token stream. |
| `group1_cran.index` | Inverted index, one term per line, lexicographically sorted. |
| `group1_boolean_results.txt` | Docid lists returned for the test queries. |
| `WRITEUP.md` | Methodology, design decisions, statistics and complexity analysis. |

---

## 2. How to run

Requirements: Python 3.6 or later. Nothing to install.

```sh
# everything at once
sh run_all.sh group1

# or step by step
python3 preprocess.py    --input cran.all --stopwords stopwords.txt --group group1
python3 build_index.py   --group group1
python3 boolean_search.py --group group1 --queryfile test_queries.txt
```

Single query, or interactive:

```sh
python3 boolean_search.py --group group1 --query "boundary AND layer"
python3 boolean_search.py --group group1          # interactive prompt
```

Useful flags:

| Flag | Effect |
|---|---|
| `--group NAME` | Prefix for all output file names. |
| `--keep-numbers` | Keep purely numeric tokens during normalization (dropped by default). |
| `--stem-stopwords` | Apply a second stop-word filter after stemming. |
| `--mode memory` \| `binary` | Index lookup by hash table in RAM, or by binary search directly on the index file (default). |
| `--no-skip` | Use the plain linear intersection instead of the skip-pointer one. |

---

## 3. Output formats

`group1_processed.all` — one block per document:

```
.I 1
.S
experiment investig aerodynam wing slipstream experiment investig aerodynam
wing slipstream experiment studi wing propel slipstream order determin
...
```

Tokens are kept in document order and repetitions are preserved, so the file
can also be reused for term-frequency work in later assignments.

`group1_cran.index` — header line, then one term per line:

```
4334, 1400
aerodynam 1,5,11,13,14,29,32,33,36,44,...
slipstream 1,409,453,484,1064,...
```

The two header integers are the vocabulary size and the maximum docid indexed.

`group1_boolean_results.txt` — one block per query:

```
Query      : boundary AND layer
Stemmed    : boundari AND layer
df         : boundari = 469, layer = 414
Matches    : 370
Time       : 0.272 ms
Docids     : 1,2,3,4,7,8,9,12,16,...
```

---

## 4. Results at a glance

| Quantity | Value |
|---|---|
| Documents processed | 1400 |
| Raw tokens (title + abstract) | 236,516 |
| Tokens after normalization | 231,254 |
| Tokens after stop-word removal | 130,830 |
| Vocabulary (distinct stems) | **4,334** |
| Maximum docid | **1400** |
| Total postings | 77,001 (17.8 per term average) |
| Index file size | 348 KB |

Full discussion in `WRITEUP.md`.

---

## 5. Notes

* Only the `.T` (title) and `.W` (abstract) fields are processed; `.A` and `.B`
  are ignored as the assignment requires.
* Cranfield repeats the title at the start of the abstract, so title terms
  naturally appear twice in the token stream. Neither the index nor the Boolean
  results are affected, since postings are duplicate-free.
* `cran_rel.txt` (relevance judgements) and `cran.qry` are not used by this
  assignment; they are for ranked retrieval and evaluation in later assignments.
  Two of our test queries were nevertheless derived from queries 001 and 002 of
  `cran.qry`.
* Correctness checks that were run: the stemmer reproduces Porter's published
  output for his sample vocabulary; the on-disk binary search returns identical
  posting lists to the in-memory hash table for all 4,334 terms; the linear,
  skip-pointer, and set-based intersections agree on 5,000 random term pairs.
