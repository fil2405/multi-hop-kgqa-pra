import sys
import random 
import gc
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import conf as f
import time
import tracemalloc
from collections import defaultdict
from functools import lru_cache

SRC_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SRC_DIR / "data"

for p in [str(SRC_DIR), str(DATA_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import data.load_graph as lg
import data.parse_qa as pq_parser
from data.entity_linker import EntityLinker
import pra_pipeline.path_traversal as pt


def extract_rel(g, hop=1):
    relations = set()
    valid_transitions = {}
    #dead-end nodes are excluded from multi-hop paths
    for head, preds in g.items():
        for pred1, tails in preds.items():
            relations.add(pred1)
            
            if pred1 not in valid_transitions:
                valid_transitions[pred1] = set()
        
            for tail in tails:
                if tail in g:
                    for pred2 in g[tail].keys():
                        valid_transitions[pred1].add(pred2)

    valid_paths = [[r] for r in relations]
    
    for _ in range(hop - 1): #if hop=1 for doesn't start
        next_paths = []
        for path in valid_paths:
            last_rel = path[-1]
            if last_rel in valid_transitions:
                for next_rel in valid_transitions[last_rel]:
                    next_paths.append(path + [next_rel])
        valid_paths = next_paths

    return valid_paths


def build_features(hop=1, split="train", out_path=None):
    random.seed(42)
    
    g = lg.load_graph()
    qa_data = pq_parser.parser(hop=hop, split=split)
    linker = EntityLinker(g)
    
    candidate_paths = extract_rel(g, hop=hop)
    print(f"Found {len(candidate_paths)} valid {hop}-hop paths")

    expected_cols = ['question_id', 'answer', 'label']
    path_to_col = {}
    for i, path in enumerate(candidate_paths):
        col_name = "_THEN_".join(path)
        path_to_col[i] = col_name
        expected_cols.extend([col_name, f'{col_name}_X_q'])
        
    float_cols = [c for c in expected_cols if c not in ['question_id', 'answer', 'label']]

    #out-of-core
    chunk_size = 50000 #800 MB per cicle
    data = []
    parquet_writer = None
    saved_chunks = 0

    def flush_chunk(batch):
        nonlocal parquet_writer, saved_chunks
        if not batch:
            return
        df_chunk = pd.DataFrame(batch, columns=expected_cols)
        df_chunk.fillna(0.0, inplace=True)
        df_chunk[float_cols] = df_chunk[float_cols].astype('float32')
        df_chunk['question_id'] = df_chunk['question_id'].astype('int32')
        df_chunk['label'] = df_chunk['label'].astype('int8')

        #write the chunk on parquet file
        table = pa.Table.from_pandas(df_chunk)
        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(out_path, table.schema)
        
        parquet_writer.write_table(table)
        saved_chunks += 1
        print(f"Saved chunk {saved_chunks}")
        
        del df_chunk
        del table
        gc.collect()

    unique_relations = set(r for path in candidate_paths for r in path)
    candidate_paths_sets = [set(p) for p in candidate_paths]

    rel_keywords = {}
    for rel in unique_relations:
        base_rel = rel.replace('inv_', '')
        rel_keywords[rel] = f.REL_KEYWORDS.get(base_rel, [base_rel])

    @lru_cache(maxsize=100000)
    def cached_prob(h, p_tuple):
        return pt.prob(g, h, list(p_tuple))

    for q_idx, qa in enumerate(qa_data):
        raw_head = qa['topic_entity']
        head = linker.link(raw_head) 
        
        if head is None:
            continue

        q_text = qa['question'].lower()
        
        path_results = {}
        candidate_scores = defaultdict(float)
        
        valid_rels = {rel for rel, keywords in rel_keywords.items() if any(kw in q_text for kw in keywords)}
        #if the word is not in the question

        valid_paths_idx = [i for i, p_set in enumerate(candidate_paths_sets) if p_set.issubset(valid_rels)]
        
        for i in valid_paths_idx:
            path = candidate_paths[i]
            reached_nodes = cached_prob(head, tuple(path))
            path_results[i] = reached_nodes
            
            for node, p_val in reached_nodes.items():
                candidate_scores[node] += p_val
                    
        true_answers = set(qa['answers'])
        reached_set = set(candidate_scores.keys())
        
        if split == "train":
            positives = reached_set & true_answers
            negatives = list(reached_set - true_answers)
            if len(negatives) > 5: #negative sampling for train set
                negatives = random.sample(negatives, 5)
            final_candidates = list(positives) + negatives
        else:
            positives = reached_set & true_answers
            sorted_negatives = sorted(list(reached_set - true_answers), key=lambda x: candidate_scores[x], reverse=True)
            final_candidates = list(positives) + sorted_negatives[:15]

        #create the dataset
        for candidate in final_candidates:
            row = {
                'question_id': q_idx,
                'answer': candidate, 
            }

            for i in valid_paths_idx:
                prob = path_results[i].get(candidate, 0.0)
                if prob > 0:
                    col_name = path_to_col[i]
                    row[col_name] = prob
                    row[f'{col_name}_X_q'] = prob

            row['label'] = 1 if candidate in true_answers else 0                
            data.append(row)

            if len(data) >= chunk_size:
                flush_chunk(data)
                data.clear()

    if data:
        flush_chunk(data)
        data.clear()

    if parquet_writer:
        parquet_writer.close()
        
    print("File writing completed")


if __name__ == '__main__':
    out_dir = SRC_DIR / "pra_pipeline" / "dataset_processed" / f"{f.HOP}-hop"
    out_dir.mkdir(parents=True, exist_ok=True)

    tracemalloc.start()
    start_time = time.perf_counter()

    for split in ["train", "dev", "test"]:
        out_parq = out_dir / f"{split}_set.parquet"
        print(f"\n--- Processing {split.upper()} set ---")
        build_features(hop=f.HOP, split=split, out_path=out_parq)
        print(f"Dataset saved in {out_parq}")

    elapsed_time = time.perf_counter() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\n=== Benchmark Summary ({f.HOP}-hop) ===")
    print(f"Total Execution Time : {elapsed_time:.2f} s ({elapsed_time/60:.2f} min)")
    print(f"Peak RAM Usage       : {peak_memory / (1024 * 1024):.2f} MB")