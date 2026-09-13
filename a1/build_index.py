#!/usr/bin/env python3
"""
build_index.py  --  Programming Assignment I (Information Retrieval)
====================================================================

Builds the inverted index  <group>_cran.index  from the preprocessed file
<group>_processed.all produced by preprocess.py.

File format (exactly as specified in the assignment)
----------------------------------------------------
    <vocabulary size>, <maximum docid indexed>
    <term> <docid>,<docid>,<docid>
    ...

  * one term per line, terms sorted in lexicographical order
  * postings list = ascending, comma separated, duplicate-free docids

Example
-------
    4334, 1400
    aerodynam 1,10,11,...
    slipstream 1,532,...

Usage
-----
    python3 build_index.py --input group1_processed.all --group group1
"""

import argparse
import os
import sys


def build_index(processed_path):
    """Read the preprocessed file and return (index, max_docid).

    `index` maps term -> list of docids in ascending order.
    A per-term "last docid seen" check keeps the postings duplicate-free in
    O(1) per token, so the whole index is built in a single linear pass over
    the token stream (no sorting of postings is needed afterwards, because
    the documents are read in ascending docid order).
    """
    index = {}
    last = {}          # term -> last docid appended, for duplicate suppression
    docid = None
    max_docid = 0
    in_tokens = False

    with open(processed_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith('.I'):
                docid = int(line[2:].strip())
                max_docid = max(max_docid, docid)
                in_tokens = False
                continue
            if line == '.S':
                in_tokens = True
                continue
            if not in_tokens or docid is None:
                continue
            for term in line.split():
                if last.get(term) == docid:
                    continue          # already recorded for this document
                index.setdefault(term, []).append(docid)
                last[term] = docid

    return index, max_docid


def write_index(index, max_docid, out_path):
    """Write the index in the required format, terms lexicographically sorted."""
    with open(out_path, 'w', encoding='utf-8') as out:
        out.write('%d, %d\n' % (len(index), max_docid))
        for term in sorted(index):
            out.write('%s %s\n' % (term, ','.join(str(d) for d in index[term])))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the Cranfield inverted index")
    ap.add_argument('--input', default=None,
                    help='preprocessed file (default: <group>_processed.all)')
    ap.add_argument('--group', default='group1', help='group name prefix')
    ap.add_argument('--output', default=None,
                    help='index file (default: <group>_cran.index)')
    args = ap.parse_args(argv)

    in_path = args.input or '%s_processed.all' % args.group
    out_path = args.output or '%s_cran.index' % args.group

    if not os.path.exists(in_path):
        sys.exit("error: '%s' not found -- run preprocess.py first" % in_path)

    index, max_docid = build_index(in_path)
    write_index(index, max_docid, out_path)

    postings = sum(len(v) for v in index.values())
    print("vocabulary size     : %d" % len(index))
    print("maximum docid       : %d" % max_docid)
    print("total postings      : %d" % postings)
    print("avg postings / term : %.2f" % (postings / float(len(index))))
    print("output              : %s" % out_path)


if __name__ == '__main__':
    main()
