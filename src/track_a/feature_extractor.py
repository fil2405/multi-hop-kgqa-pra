#type: ignore
import sys
import random 
import gc
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import conf as f

SRC_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SRC_DIR / "data"

for p in [str(SRC_DIR), str(DATA_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import data.load_graph as lg
import data.parse_qa as pq_parser
from data.entity_linker import EntityLinker
import track_a.path_traversal as pt

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
    
    g = lg.load_graph()
    qa_data = pq_parser.parser(hop=hop, split=split)
    linker = EntityLinker(g)
    
    candidate_paths = extract_rel(g, hop=hop)
    print(f"Found {len(candidate_paths)} valid {hop}-hop paths")

    data = []

    for q_idx, qa in enumerate(qa_data):
        raw_head = qa['topic_entity']
        head = linker.link(raw_head) 
        
        if head is None:
            continue

        q_text = qa['question'].lower()
        
        path_results = {}
        candidate_scores = {}
        
        for i, path in enumerate(candidate_paths):
            flag = 1
            for p in path:
                base_rel = p.replace('inv_', '')
                keywords = f.REL_KEYWORDS.get(base_rel, [base_rel])
                if not any(kw in q_text for kw in keywords): #if the word is not in the question
                    flag = 0
                    break
                    
            if flag == 1:
                reached_nodes = pt.prob(g, head, path)
                path_results[i] = reached_nodes
                
                for node, p_val in reached_nodes.items():
                    candidate_scores[node] = candidate_scores.get(node, 0.0) + p_val
                    
        true_answers = set(qa['answers'])
        reached_set = set(candidate_scores.keys())
        
        if split == "train":
            positives = reached_set & true_answers
            negatives = list(reached_set - true_answers)
            if len(negatives) > 5:  #negative sampling for train set
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

            for i, path in enumerate(candidate_paths):
                if i in path_results:
                    prob = path_results[i].get(candidate, 0.0)
                    if prob > 0:
                        col_name = "_THEN_".join(path) 
                        row[col_name] = prob
                        row[f'{col_name}_X_q'] = prob

            row['label'] = 1 if candidate in qa['answers'] else 0                
            data.append(row)

    #out-of-core
    print(f"\nWriting {len(data)} rows directly to disk...")
    
    expected_cols = ['question_id', 'answer', 'label']
    for path in candidate_paths:
        col_name = "_THEN_".join(path)
        expected_cols.extend([col_name, f'{col_name}_X_q'])
        
    float_cols = [c for c in expected_cols if c not in ['question_id', 'answer', 'label']]

    chunk_size = 50000 #800 MB per cicle
    parquet_writer = None
    
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        df_chunk = pd.DataFrame(chunk)
        
        df_chunk = df_chunk.reindex(columns=expected_cols, fill_value=0.0)
        
        df_chunk[float_cols] = df_chunk[float_cols].astype('float32')
        
        #write the chunk on parquet file
        table = pa.Table.from_pandas(df_chunk)
        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(out_path, table.schema)
        
        parquet_writer.write_table(table)
        print(f"Saved chunk {i // chunk_size + 1} of {(len(data) // chunk_size) + 1}")
        
        del df_chunk
        del table
        gc.collect()

    if parquet_writer:
        parquet_writer.close()
        
    print("File writing completed")


if __name__ == '__main__':

    out_dir = SRC_DIR / "track_a" / "dataset_processed" / f"{f.HOP}-hop"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "dev", "test"]:
        out_parq = out_dir / f"{split}_set.parquet"
        print(f"\n--- Processing {split.upper()} set ---")
        build_features(hop=f.HOP, split=split, out_path=out_parq)
        print(f"Dataset saved in {out_parq}")