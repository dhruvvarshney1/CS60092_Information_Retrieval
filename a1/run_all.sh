#!/bin/sh
# run_all.sh -- reproduce every submitted output file from scratch.
#
#   sh run_all.sh [groupname]
#
# Expects cran.all, stopwords.txt and test_queries.txt in the current directory.

set -e
GROUP=${1:-group1}

echo "== 1. preprocessing =========================================="
python3 preprocess.py --input cran.all --stopwords stopwords.txt --group "$GROUP"

echo
echo "== 2. indexing =============================================="
python3 build_index.py --group "$GROUP"

echo
echo "== 3. boolean search (test queries) ========================="
python3 boolean_search.py --group "$GROUP" --queryfile test_queries.txt \
        --mode binary > /dev/null
echo "results -> ${GROUP}_boolean_results.txt"

echo
echo "Done.  Deliverables:"
ls -l "${GROUP}_processed.all" "${GROUP}_cran.index" "${GROUP}_boolean_results.txt"
