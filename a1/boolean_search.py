#!/usr/bin/env python3
"""
boolean_search.py  --  Programming Assignment I (Information Retrieval)
=======================================================================

Answers two-word Boolean queries  ("<w1> AND <w2>",  "<w1> OR <w2>")  against
the index file  <group>_cran.index  produced by build_index.py.

The query words go through exactly the same pipeline as the documents
(tokenize -> normalize -> stop-word removal -> Porter stemming) so that the
query vocabulary and the index vocabulary agree.

Search is performed *over the index file*, in one of two modes:

  --mode binary   (default)  the index file is left on disk.  Because the
                             terms are stored in lexicographical order, a
                             posting list is found with a binary search over
                             the byte offsets of the file:
                             O(log F) seeks, ~1 KB read, no start-up cost.
  --mode memory              the whole index is loaded into a hash table:
                             O(1) lookup, useful when many queries are run
                             in one batch.

Merging of the two posting lists (bonus part -- efficient algorithms):

  AND :  linear merge, O(m+n), with optional skip pointers every
         sqrt(len) postings, which lets the merge jump over long runs of
         non-matching docids  (--skip, on by default).
  OR  :  linear merge of two sorted lists, O(m+n), no duplicates.

Both are strictly better than the naive O(m*n) "for each x in A: if x in B"
and they never materialise a Python set, so the output stays sorted.

Usage
-----
  single query
      python3 boolean_search.py --group group1 --query "aeroelastic AND aircraft"
  batch of queries from a file (one query per line)
      python3 boolean_search.py --group group1 --queryfile test_queries.txt
  interactive
      python3 boolean_search.py --group group1
"""

import argparse
import math
import os
import sys
import time

from preprocess import tokenize, normalize, load_stopwords
from porter_stemmer import stem as porter_stem


# --------------------------------------------------------------------------- #
# Query processing                                                             #
# --------------------------------------------------------------------------- #

OPERATORS = ('AND', 'OR', 'NOT')


def process_query_term(word, stopwords=None):
    """Normalize + stem a single query word exactly like a document token."""
    toks = normalize(tokenize(word))
    if stopwords:
        toks = [t for t in toks if t not in stopwords]
    if not toks:
        return None
    return porter_stem(toks[0])


def parse_query(query, stopwords=None):
    """Split '<w1> <OP> <w2>' into (term1, operator, term2).

    A single-word query is returned as (term1, None, None).
    'A AND NOT B' is supported as an extension.
    Raises ValueError on a malformed query.
    """
    parts = query.strip().split()
    if not parts:
        raise ValueError("empty query")

    # locate the (first) operator
    op_pos = None
    for i, p in enumerate(parts):
        if p.upper() in ('AND', 'OR'):
            op_pos = i
            break

    if op_pos is None:
        if len(parts) != 1:
            raise ValueError("expected '<word> AND|OR <word>', got: %s" % query)
        t = process_query_term(parts[0], stopwords)
        if t is None:
            raise ValueError("query word '%s' is a stop word / empty" % parts[0])
        return t, None, None

    op = parts[op_pos].upper()
    left = parts[:op_pos]
    right = parts[op_pos + 1:]
    negate = False
    if right and right[0].upper() == 'NOT':
        negate = True
        right = right[1:]
    if len(left) != 1 or len(right) != 1:
        raise ValueError("only two-word queries are supported: %s" % query)

    t1 = process_query_term(left[0], stopwords)
    t2 = process_query_term(right[0], stopwords)
    if t1 is None or t2 is None:
        raise ValueError("a query word is a stop word or contains no letters")
    if negate:
        op = op + ' NOT'
    return t1, op, t2


# --------------------------------------------------------------------------- #
# Index access                                                                 #
# --------------------------------------------------------------------------- #

class Index(object):
    """Access to <group>_cran.index, either on disk or in memory."""

    def __init__(self, path, mode='binary'):
        self.path = path
        self.mode = mode
        self.table = None
        with open(path, 'r', encoding='utf-8') as fh:
            header = fh.readline().split(',')
            self.vocab_size = int(header[0])
            self.max_docid = int(header[1])
            self.data_start = fh.tell()
        if mode == 'memory':
            self._load()

    # ---- in-memory ------------------------------------------------------- #
    def _load(self):
        self.table = {}
        with open(self.path, 'r', encoding='utf-8') as fh:
            fh.readline()                      # skip header
            for line in fh:
                sp = line.find(' ')
                if sp < 0:
                    continue
                term = line[:sp]
                self.table[term] = [int(x) for x in line[sp + 1:].split(',') if x.strip()]

    # ---- binary search on the sorted index file -------------------------- #
    def _postings_from_disk(self, term):
        """Binary search over the byte offsets of the sorted index file.

        Invariant: `lo` is always the offset of the beginning of a line whose
        term is < `term` (or the first data line), so the final linear scan
        starting at `lo` is guaranteed to reach the wanted line first.
        The search stops narrowing at a 4 KB window -- below that a single
        sequential read is cheaper than more seeks.
        """
        target = term.encode('utf-8')
        with open(self.path, 'rb') as fh:
            fh.seek(0, os.SEEK_END)
            hi = fh.tell()
            lo = self.data_start

            while hi - lo > 4096:
                mid = (lo + hi) // 2
                fh.seek(mid)
                fh.readline()                  # align: discard partial line
                pos = fh.tell()                # start of a complete line
                if pos >= hi:
                    hi = mid
                    continue
                line = fh.readline()
                if not line:
                    hi = mid
                    continue
                key = line.split(b' ', 1)[0]
                if key < target:
                    lo = fh.tell()             # still a line start
                else:
                    hi = pos

            fh.seek(lo)
            while True:
                line = fh.readline()
                if not line:
                    return []
                key, _, rest = line.partition(b' ')
                if key == target:
                    return [int(x) for x in rest.split(b',') if x.strip()]
                if key > target:
                    return []

    def postings(self, term):
        """Return the (ascending) posting list of `term`, or []."""
        if self.table is not None:
            return self.table.get(term, [])
        return self._postings_from_disk(term)


# --------------------------------------------------------------------------- #
# Merge algorithms                                                             #
# --------------------------------------------------------------------------- #

def intersect(p1, p2):
    """AND: standard linear merge of two sorted posting lists.  O(m+n)."""
    out = []
    i = j = 0
    n1, n2 = len(p1), len(p2)
    while i < n1 and j < n2:
        a, b = p1[i], p2[j]
        if a == b:
            out.append(a)
            i += 1
            j += 1
        elif a < b:
            i += 1
        else:
            j += 1
    return out


def intersect_with_skips(p1, p2):
    """AND with skip pointers of length sqrt(n).

    When the two lists are of very different lengths the skips let the long
    list jump ahead sqrt(n) postings at a time instead of stepping one by
    one, which is where most of the comparisons are saved.
    """
    n1, n2 = len(p1), len(p2)
    if n1 == 0 or n2 == 0:
        return []
    s1 = max(1, int(math.sqrt(n1)))
    s2 = max(1, int(math.sqrt(n2)))
    out = []
    i = j = 0
    while i < n1 and j < n2:
        a, b = p1[i], p2[j]
        if a == b:
            out.append(a)
            i += 1
            j += 1
        elif a < b:
            # try to skip in p1
            while i + s1 < n1 and p1[i + s1] < b:
                i += s1
            i += 1
        else:
            while j + s2 < n2 and p2[j + s2] < a:
                j += s2
            j += 1
    return out


def union(p1, p2):
    """OR: linear merge of two sorted posting lists, duplicates removed."""
    out = []
    i = j = 0
    n1, n2 = len(p1), len(p2)
    while i < n1 and j < n2:
        a, b = p1[i], p2[j]
        if a == b:
            out.append(a)
            i += 1
            j += 1
        elif a < b:
            out.append(a)
            i += 1
        else:
            out.append(b)
            j += 1
    out.extend(p1[i:])
    out.extend(p2[j:])
    return out


def difference(p1, p2):
    """AND NOT: postings in p1 that are not in p2.  O(m+n)."""
    out = []
    i = j = 0
    n1, n2 = len(p1), len(p2)
    while i < n1:
        if j >= n2 or p1[i] < p2[j]:
            out.append(p1[i])
            i += 1
        elif p1[i] == p2[j]:
            i += 1
            j += 1
        else:
            j += 1
    return out


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #

def run_query(index, query, stopwords=None, use_skips=True):
    """Execute one Boolean query.  Returns (docids, info-dict)."""
    t1, op, t2 = parse_query(query, stopwords)
    start = time.time()
    p1 = index.postings(t1)
    if op is None:
        result = list(p1)
        p2 = []
    else:
        p2 = index.postings(t2)
        if op == 'AND':
            result = intersect_with_skips(p1, p2) if use_skips else intersect(p1, p2)
        elif op == 'OR':
            result = union(p1, p2)
        elif op == 'AND NOT':
            result = difference(p1, p2)
        else:
            raise ValueError("unsupported operator: %s" % op)
    elapsed = (time.time() - start) * 1000.0
    info = {'terms': (t1, t2), 'op': op, 'df1': len(p1), 'df2': len(p2),
            'hits': len(result), 'ms': elapsed}
    return result, info


def format_result(query, result, info):
    t1, t2 = info['terms']
    lines = ['Query      : %s' % query]
    if info['op'] is None:
        lines.append('Stemmed    : %s' % t1)
        lines.append('df         : %s = %d' % (t1, info['df1']))
    else:
        lines.append('Stemmed    : %s %s %s' % (t1, info['op'], t2))
        lines.append('df         : %s = %d, %s = %d' % (t1, info['df1'], t2, info['df2']))
    lines.append('Matches    : %d' % info['hits'])
    lines.append('Time       : %.3f ms' % info['ms'])
    lines.append('Docids     : %s' % (','.join(str(d) for d in result) if result else '(none)'))
    return '\n'.join(lines) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description="Boolean retrieval over the Cranfield index")
    ap.add_argument('--group', default='group1', help='group name prefix')
    ap.add_argument('--index', default=None,
                    help='index file (default: <group>_cran.index)')
    ap.add_argument('--stopwords', default='stopwords.txt')
    ap.add_argument('--query', default=None, help='a single Boolean query')
    ap.add_argument('--queryfile', default=None,
                    help='file with one Boolean query per line')
    ap.add_argument('--output', default=None,
                    help='result file (default: <group>_boolean_results.txt)')
    ap.add_argument('--mode', choices=('binary', 'memory'), default='binary',
                    help='index access method (default: binary search on disk)')
    ap.add_argument('--no-skip', action='store_true',
                    help='use the plain linear merge instead of skip pointers')
    args = ap.parse_args(argv)

    index_path = args.index or '%s_cran.index' % args.group
    out_path = args.output or '%s_boolean_results.txt' % args.group
    if not os.path.exists(index_path):
        sys.exit("error: index '%s' not found -- run build_index.py first" % index_path)

    stopwords = load_stopwords(args.stopwords) if os.path.exists(args.stopwords) else set()
    index = Index(index_path, mode=args.mode)
    use_skips = not args.no_skip

    queries = []
    if args.query:
        queries.append(args.query)
    if args.queryfile:
        with open(args.queryfile, 'r', encoding='utf-8') as fh:
            queries.extend([l.strip() for l in fh
                            if l.strip() and not l.startswith('#')])

    # ---- interactive mode ------------------------------------------------ #
    if not queries:
        print("Boolean search over %s  (%d terms, max docid %d)"
              % (index_path, index.vocab_size, index.max_docid))
        print("Enter queries like:  boundary AND layer      (blank line to quit)")
        while True:
            try:
                q = input('query> ').strip()
            except EOFError:
                break
            if not q:
                break
            try:
                res, info = run_query(index, q, stopwords, use_skips)
                print(format_result(q, res, info))
            except ValueError as e:
                print("error: %s" % e)
        return

    # ---- batch mode ------------------------------------------------------ #
    with open(out_path, 'w', encoding='utf-8') as out:
        out.write('# Boolean search results -- index: %s (%d terms, max docid %d)\n'
                  % (index_path, index.vocab_size, index.max_docid))
        out.write('# merge: %s\n\n'
                  % ('skip-pointer intersection' if use_skips else 'linear intersection'))
        for q in queries:
            try:
                res, info = run_query(index, q, stopwords, use_skips)
                block = format_result(q, res, info)
            except ValueError as e:
                block = 'Query      : %s\nError      : %s\n' % (q, e)
            out.write(block + '\n')
            print(block)
    print("results written to %s" % out_path)


if __name__ == '__main__':
    main()
