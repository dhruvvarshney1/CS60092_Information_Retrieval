#!/usr/bin/env python3
"""
preprocess.py  --  Programming Assignment I (Information Retrieval)
====================================================================

Preprocesses the Cranfield collection (cran.all, 1400 documents) and writes a
single annotated token file  <group>_processed.all.

Four independent functions implement the four required steps
(they can be imported and used separately):

    tokenize(text)                 -> list of raw token strings
    normalize(tokens)              -> list of normalized token strings
    remove_stopwords(tokens, sw)   -> list with stop words removed
    stem_tokens(tokens)            -> list of Porter stems

They are integrated by  preprocess_document()  /  main()  in the pipeline

    raw text --> tokenize --> normalize --> stop-word removal --> stemming

Only the .T (title) and .W (abstract) fields are processed;
.A (author) and .B (bibliography/affiliation) are ignored, as required.

Output format
-------------
    .I 1
    .S
    experiment investig aerodynam wing slipstream
    ...

Usage
-----
    python3 preprocess.py --input cran.all --stopwords stopwords.txt \
                          --group group1 --output group1_processed.all
"""

import argparse
import os
import re
import sys

from porter_stemmer import stem as porter_stem

# --------------------------------------------------------------------------- #
# 0.  Reading the collection                                                    #
# --------------------------------------------------------------------------- #

# Tags used by the Cranfield collection
_TAG_RE = re.compile(r'^\.([IABTW])\s*(.*)$')

# Fields we index: title and abstract only
_INDEXED_FIELDS = ('T', 'W')


def read_collection(path):
    """Parse the Cranfield file and yield (docid, text) pairs.

    `text` is the concatenation of the .T (title) and .W (abstract) fields.
    The .A and .B fields are skipped.  The parser is streaming, so the whole
    1.6 MB file is never held in memory more than one document at a time.
    """
    docid = None
    field = None
    buf = []

    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            line = line.rstrip('\r\n')
            m = _TAG_RE.match(line)
            if m:
                tag, rest = m.group(1), m.group(2).strip()
                if tag == 'I':
                    # flush the previous document
                    if docid is not None:
                        yield docid, ' '.join(buf)
                    docid = int(rest)
                    buf = []
                    field = None
                else:
                    field = tag
                    if tag in _INDEXED_FIELDS and rest:
                        buf.append(rest)
                continue
            if field in _INDEXED_FIELDS and line.strip():
                buf.append(line.strip())

    if docid is not None:
        yield docid, ' '.join(buf)


# --------------------------------------------------------------------------- #
# i.  TOKENIZATION                                                             #
# --------------------------------------------------------------------------- #

# A token is a maximal run of letters/digits, optionally glued by a single
# internal hyphen, apostrophe, dot or slash (e.g. "boundary-layer",
# "reynolds's", "0.5", "and/or").  Everything else is a separator.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'./][A-Za-z0-9]+)*")


def tokenize(text):
    """Step (i): convert a piece of text into a list of raw token strings.

    Punctuation, brackets, the slashes Cranfield uses for emphasis
    (/destalling/), formula symbols and white space act as delimiters.
    The relative order of the tokens is preserved.
    """
    return _TOKEN_RE.findall(text)


# --------------------------------------------------------------------------- #
# ii. NORMALIZATION                                                            #
# --------------------------------------------------------------------------- #

_NON_ALNUM = re.compile(r"[^a-z0-9]")
_ALL_DIGITS = re.compile(r"^[0-9]+$")


def normalize(tokens, keep_numbers=False, min_length=2):
    """Step (iv): normalize raw tokens so that surface variants collapse.

    Rules applied, in order:
      1. case folding                        Wing / WING  -> wing
      2. apostrophes are deleted             reynolds's   -> reynoldss -> reynolds
      3. hyphen / slash compounds are split  boundary-layer -> boundary, layer
         (the parts are kept individually so that a query for either part
          matches; the compound itself is not indexed)
      4. any remaining non-alphanumeric character is stripped
      5. purely numeric tokens are discarded (they are page/year/figure
         numbers in this collection); use keep_numbers=True to retain them
      6. tokens shorter than `min_length` characters are discarded

    Returns a new list (may be longer than the input because of rule 3).
    """
    out = []
    for tok in tokens:
        tok = tok.lower().replace("'", "")
        # split compounds on hyphen / slash; a lone dot inside a number
        # (0.5) is handled by the numeric rule below
        parts = re.split(r"[-/]", tok) if ('-' in tok or '/' in tok) else [tok]
        for p in parts:
            p = _NON_ALNUM.sub('', p)
            if not p:
                continue
            if not keep_numbers and _ALL_DIGITS.match(p):
                continue
            if len(p) < min_length:
                continue
            out.append(p)
    return out


# --------------------------------------------------------------------------- #
# iii. STOP-WORD REMOVAL                                                       #
# --------------------------------------------------------------------------- #

def load_stopwords(path):
    """Read the stop-word list (one word per line) into a set.

    The words are normalized with the same rules as the documents so that the
    two vocabularies are comparable.
    """
    words = set()
    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            w = line.strip().lower()
            if not w:
                continue
            w = _NON_ALNUM.sub('', w)
            if w:
                words.add(w)
    return words


def remove_stopwords(tokens, stopwords):
    """Step (iii): drop every token that appears in the stop-word list."""
    return [t for t in tokens if t not in stopwords]


# --------------------------------------------------------------------------- #
# ii. STEMMING                                                                 #
# --------------------------------------------------------------------------- #

def stem_tokens(tokens):
    """Step (ii): reduce every token to its Porter stem."""
    return [porter_stem(t) for t in tokens]


# --------------------------------------------------------------------------- #
# Integration                                                                  #
# --------------------------------------------------------------------------- #

def preprocess_document(text, stopwords, keep_numbers=False,
                        stem_stopwords=False):
    """Run the full pipeline on one document / query string.

    tokenize -> normalize -> stop-word removal -> stemming

    Stop words are removed *before* stemming because the stop list is a list
    of surface forms.  With stem_stopwords=True a second filter is applied
    after stemming (against the stemmed stop list), which also removes
    inflected stop words such as "using" -> "use".
    """
    tokens = tokenize(text)
    tokens = normalize(tokens, keep_numbers=keep_numbers)
    tokens = remove_stopwords(tokens, stopwords)
    tokens = stem_tokens(tokens)
    if stem_stopwords:
        stemmed_sw = {porter_stem(w) for w in stopwords}
        tokens = remove_stopwords(tokens, stemmed_sw)
    return tokens


def _wrap(tokens, width=76):
    """Pretty-print the token stream in lines of at most `width` characters."""
    lines, cur = [], ''
    for t in tokens:
        if cur and len(cur) + 1 + len(t) > width:
            lines.append(cur)
            cur = t
        else:
            cur = t if not cur else cur + ' ' + t
    if cur:
        lines.append(cur)
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cranfield preprocessing")
    ap.add_argument('--input', default='cran.all',
                    help='Cranfield collection file (default: cran.all)')
    ap.add_argument('--stopwords', default='stopwords.txt',
                    help='stop-word list, one word per line')
    ap.add_argument('--group', default='group1', help='group name prefix')
    ap.add_argument('--output', default=None,
                    help='output file (default: <group>_processed.all)')
    ap.add_argument('--keep-numbers', action='store_true',
                    help='keep purely numeric tokens')
    ap.add_argument('--stem-stopwords', action='store_true',
                    help='also filter stop words after stemming')
    args = ap.parse_args(argv)

    out_path = args.output or '%s_processed.all' % args.group

    if not os.path.exists(args.input):
        sys.exit("error: collection file '%s' not found" % args.input)
    if not os.path.exists(args.stopwords):
        sys.exit("error: stop-word file '%s' not found" % args.stopwords)

    stopwords = load_stopwords(args.stopwords)

    n_docs = 0
    n_tokens = 0
    vocab = set()
    with open(out_path, 'w', encoding='utf-8') as out:
        for docid, text in read_collection(args.input):
            tokens = preprocess_document(text, stopwords,
                                         keep_numbers=args.keep_numbers,
                                         stem_stopwords=args.stem_stopwords)
            n_docs += 1
            n_tokens += len(tokens)
            vocab.update(tokens)
            out.write('.I %d\n.S\n' % docid)
            for line in _wrap(tokens):
                out.write(line + '\n')

    print("documents processed : %d" % n_docs)
    print("tokens written      : %d" % n_tokens)
    print("distinct stems      : %d" % len(vocab))
    print("stop words loaded   : %d" % len(stopwords))
    print("output              : %s" % out_path)


if __name__ == '__main__':
    main()
