#!/usr/bin/env python3
"""Sparse ranked retrieval and evaluation for Cranfield PA2.

Only the Python standard library and the preprocessing code from PA1 are used.
"""
import argparse
import math
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'a1'))
from preprocess import load_stopwords, preprocess_document, read_collection


def read_queries(path, stopwords):
    queries = {}
    current = None
    ordinal = 0
    text = []
    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith('.I'):
                if current is not None:
                    queries[ordinal] = preprocess_document(' '.join(text), stopwords)
                ordinal += 1
                current, text = int(line[2:].strip()), []
            elif line.startswith('.W'):
                continue
            elif current is not None:
                text.append(line.strip())
    if current is not None:
        queries[ordinal] = preprocess_document(' '.join(text), stopwords)
    return queries


def read_qrels(path):
    qrels = defaultdict(dict)
    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 3:
                q, doc, grade = map(int, parts[:3])
                qrels[q][doc] = grade
    return dict(qrels)


class SparseIndex:
    def __init__(self, collection, stopwords):
        self.postings = defaultdict(dict)
        self.lengths = {}
        self.doc_terms = {}
        for docid, text in read_collection(collection):
            terms = preprocess_document(text, stopwords)
            counts = Counter(terms)
            self.doc_terms[docid] = counts
            self.lengths[docid] = len(terms)
            for term, tf in counts.items():
                self.postings[term][docid] = tf
        self.docs = sorted(self.doc_terms)
        self.avgdl = sum(self.lengths.values()) / len(self.docs)
        self.n = len(self.docs)
        self.idf = {t: math.log((self.n + 1) / (len(p) + 0.5))
                    for t, p in self.postings.items()}

    def rank(self, terms, model='bm25', k1=1.2, b=0.75, limit=1000):
        scores = defaultdict(float)
        qtf = Counter(terms)
        if model == 'tfidf':
            qweights = {t: (1 + math.log(tf)) * self.idf.get(t, 0)
                        for t, tf in qtf.items() if t in self.postings}
            qnorm = math.sqrt(sum(x * x for x in qweights.values())) or 1.0
            for term, qw in qweights.items():
                for docid, tf in self.postings[term].items():
                    dw = (1 + math.log(tf)) * self.idf[term]
                    scores[docid] += (dw / self._doc_norm(docid)) * (qw / qnorm)
        else:
            for term in qtf:
                if term not in self.postings:
                    continue
                idf = self.idf[term]
                for docid, tf in self.postings[term].items():
                    norm = 1 - b + b * self.lengths[docid] / self.avgdl
                    scores[docid] += idf * tf * (k1 + 1) / (tf + k1 * norm)
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return [(docid, score) for docid, score in ranked[:limit]]

    def _doc_norm(self, docid):
        return math.sqrt(sum(((1 + math.log(tf)) * self.idf[t]) ** 2
                             for t, tf in self.doc_terms[docid].items())) or 1.0


def dcg(grades):
    return sum((2 ** grade - 1) / math.log2(i + 2)
               for i, grade in enumerate(grades))


def evaluate(run, qrels, k=10):
    aps = []
    rr = []
    p10 = []
    recalls = []
    ndcgs = []
    for qid, judgments in qrels.items():
        if qid not in run:
            continue
        ranked = [doc for doc, _ in run.get(qid, [])]
        relevant = {d for d, g in judgments.items() if g > 0}
        hits = 0
        precisions = []
        for i, doc in enumerate(ranked, 1):
            if doc in relevant:
                hits += 1
                precisions.append(hits / i)
        aps.append(sum(precisions) / len(relevant) if relevant else 0.0)
        rr.append(1 / next((i for i, d in enumerate(ranked, 1) if d in relevant), float('inf')))
        top = ranked[:k]
        p10.append(sum(d in relevant for d in top) / k)
        recalls.append(sum(d in relevant for d in ranked) / len(relevant) if relevant else 0.0)
        actual = [judgments.get(d, 0) for d in top]
        ideal = sorted(judgments.values(), reverse=True)[:k]
        ndcgs.append(dcg(actual) / dcg(ideal) if dcg(ideal) else 0.0)
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return {'MAP': mean(aps), 'MRR': mean(rr), 'P@10': mean(p10),
            'Recall': mean(recalls), 'nDCG@10': mean(ndcgs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--collection', default='cran.all.1400')
    ap.add_argument('--queries', default='cran.qry')
    ap.add_argument('--qrels', default='cranqrel')
    ap.add_argument('--stopwords', default='../a1/stopwords.txt')
    ap.add_argument('--output', default='dhruvvarshney_ranked_results.txt')
    ap.add_argument('--report', default='dhruvvarshney_experiment_results.txt')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        assert dcg([3, 2, 1]) > dcg([1, 2, 3])
        assert evaluate({1: [(2, 1.0), (1, 0.5)]}, {1: {1: 1}})['MRR'] == 0.5
        print('self-test passed')
        return
    stopwords = load_stopwords(args.stopwords)
    started = time.perf_counter()
    index = SparseIndex(args.collection, stopwords)
    indexing_ms = (time.perf_counter() - started) * 1000
    queries = read_queries(args.queries, stopwords)
    qrels = read_qrels(args.qrels)
    configs = [('tfidf', None, None), ('bm25', 0.8, 0.5),
               ('bm25', 1.2, 0.75), ('bm25', 1.6, 0.75),
               ('bm25', 1.2, 1.0)]
    results = {}
    for model, k1, b in configs:
        run = {qid: index.rank(terms, model, k1 or 1.2, b or 0.75)
               for qid, terms in queries.items()}
        results[(model, k1, b)] = (run, evaluate(run, qrels))
    best = max(results, key=lambda config: results[config][1]['MAP'])
    best_run, best_metrics = results[best]
    with open(args.output, 'w', encoding='utf-8') as out:
        for qid in sorted(best_run):
            for rank, (docid, score) in enumerate(best_run[qid], 1):
                out.write(f'{qid} Q0 {docid} {rank} {score:.8f} dhruvvarshney\n')
    with open(args.report, 'w', encoding='utf-8') as out:
        out.write(f'collection_docs={index.n}\nterms={len(index.postings)}\n')
        out.write(f'indexing_ms={indexing_ms:.3f}\nqueries={len(queries)}\n')
        out.write('model,k1,b,MAP,MRR,P@10,Recall,nDCG@10\n')
        for config, (_, metrics) in results.items():
            model, k1, b = config
            out.write('%s,%s,%s,%s\n' % (model, k1 or '', b or '',
                      ','.join(f'{metrics[k]:.6f}' for k in metrics)))
        out.write(f'best={best}\n')
        out.write('best_metrics=' + ','.join(f'{k}={v:.6f}' for k, v in best_metrics.items()) + '\n')
    print(f'indexed {index.n} documents, {len(index.postings)} terms in {indexing_ms:.1f} ms')
    print('best:', best, best_metrics)
    print('wrote:', args.output, args.report)


if __name__ == '__main__':
    main()
