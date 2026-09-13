#!/usr/bin/env python3
"""PyTerrier/Terrier sparse retrieval pipeline for PA2."""
import argparse
import os
import shutil
import sys
import time
from collections import Counter

import pandas as pd
import pyterrier as pt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'a1'))
from preprocess import load_stopwords, preprocess_document, read_collection


def topics(path, stopwords):
    rows, current, text = [], None, []
    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith('.I'):
                if current is not None:
                    rows.append((str(len(rows) + 1), ' '.join(text)))
                current, text = line[2:].strip(), []
            elif not line.startswith('.W') and current is not None:
                text.append(line.strip())
    if current is not None:
        rows.append((str(len(rows) + 1), ' '.join(text)))
    return pd.DataFrame(
        [(qid, ' '.join(preprocess_document(query, stopwords))) for qid, query in rows],
        columns=['qid', 'query'])


def qrels(path):
    rows = []
    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            qid, docno, grade = line.split()[:3]
            rows.append((qid, docno, int(grade)))
    return pd.DataFrame(rows, columns=['qid', 'docno', 'label'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--collection', default='cran.all.1400')
    ap.add_argument('--queries', default='cran.qry')
    ap.add_argument('--qrels', default='cranqrel')
    ap.add_argument('--stopwords', default='../a1/stopwords.txt')
    ap.add_argument('--index', default='dhruvvarshney_terrier_index')
    ap.add_argument('--output', default='dhruvvarshney_pyterrier_results.txt')
    ap.add_argument('--metrics', default='dhruvvarshney_pyterrier_metrics.csv')
    args = ap.parse_args()

    java_candidates = [os.environ.get('JAVA_HOME', ''),
                       os.path.join(sys.prefix, 'Library', 'lib', 'jvm'),
                       sys.prefix]
    for java_home in java_candidates:
        if os.path.exists(os.path.join(java_home, 'bin', 'server', 'jvm.dll')):
            os.environ['JAVA_HOME'] = java_home
            break
    pt.java.init()
    stopwords = load_stopwords(args.stopwords)
    index_path = os.path.abspath(args.index)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    docs = ({'docno': str(docid), 'toks': Counter(preprocess_document(text, stopwords))}
            for docid, text in read_collection(args.collection))
    indexer = pt.index.IterDictIndexer(index_path, pretokenised=True, meta={'docno': 20})
    started = time.perf_counter()
    if not os.path.exists(os.path.join(index_path, 'data.properties')):
        indexref = indexer.index(docs)
    else:
        indexref = pt.IndexRef.of(index_path)
    indexing_ms = (time.perf_counter() - started) * 1000

    test_topics, test_qrels = topics(args.queries, stopwords), qrels(args.qrels)
    configs = [
        ('TF_IDF', 'TF_IDF', {}),
        ('BM25_k08_b05', 'BM25', {'bm25.k_1': '0.8', 'bm25.b': '0.5'}),
        ('BM25_k12_b075', 'BM25', {'bm25.k_1': '1.2', 'bm25.b': '0.75'}),
        ('BM25_k16_b075', 'BM25', {'bm25.k_1': '1.6', 'bm25.b': '0.75'}),
        ('BM25_k12_b1', 'BM25', {'bm25.k_1': '1.2', 'bm25.b': '1.0'}),
    ]
    retrievers = [pt.BatchRetrieve(indexref, wmodel=model, controls=controls,
                                   num_results=1000)
                  for _, model, controls in configs]
    metrics = ['map', 'recip_rank', 'P.10', 'recall', 'ndcg_cut.10']
    experiment = pt.Experiment(retrievers, test_topics, test_qrels,
                               eval_metrics=metrics,
                               names=[name for name, _, _ in configs],
                               filter_by_qrels=False)
    experiment.insert(0, 'indexing_ms', indexing_ms)
    experiment.to_csv(args.metrics, index=False)

    best_name = experiment.loc[experiment['map'].idxmax(), 'name']
    best = retrievers[[name for name, _, _ in configs].index(best_name)]
    run = best.transform(test_topics)
    with open(args.output, 'w', encoding='utf-8') as fh:
        for row in run.itertuples(index=False):
            fh.write(f'{row.qid} Q0 {row.docno} {int(row.rank) + 1} {row.score:.8f} {best_name}\n')
    print(experiment.to_string(index=False))
    print(f'best={best_name}; indexing_ms={indexing_ms:.1f}')


if __name__ == '__main__':
    main()
