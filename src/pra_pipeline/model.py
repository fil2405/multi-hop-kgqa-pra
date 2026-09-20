#type: ignore
import gc
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import SGDClassifier, LogisticRegression
import conf as f

SRC_DIR = Path(__file__).resolve().parent.parent

def evaluate_dataset(model, parquet_path, feature_cols):
    parquet_file = pq.ParquetFile(parquet_path)
    results = []
    
    for batch in parquet_file.iter_batches(batch_size=50000):
        df_chunk = batch.to_pandas().fillna(0.0)
        X_chunk = df_chunk[feature_cols].astype('float32').values
        
        out_df = df_chunk[['question_id', 'label']].copy()
        out_df['pred_score'] = model.predict_proba(X_chunk)[:, 1]
        results.append(out_df)
            
    eval_df = pd.concat(results, ignore_index=True)
    
    #sort candidates per question by predicted score
    eval_df = eval_df.sort_values(['question_id', 'pred_score'], ascending=[True, False])
    
    #assign a rank to each candidate
    eval_df['rank'] = eval_df.groupby('question_id').cumcount() + 1
    
    #Hits@1
    top1 = eval_df.groupby('question_id').first()
    hits_at_1 = (top1['label'] == 1).mean()
    
    #MRR 
    correct_hits = eval_df[eval_df['label'] == 1]
    best_ranks = correct_hits.groupby('question_id')['rank'].min()
    total_questions = eval_df['question_id'].nunique()
    mrr = (1.0 / best_ranks).sum() / total_questions
    
    return hits_at_1, mrr

def train():
    train_path = SRC_DIR / "track_a" / f.TRAIN_PATH
    
    parquet_file = pq.ParquetFile(train_path)
    feature_cols = [c for c in parquet_file.schema.names if c not in ['question_id', 'answer', 'label']]
    
    #training based on hop count
    if f.HOP == 1:
        print("Training Logistic Regression for 1-hop...")
        train_df = pd.read_parquet(train_path).fillna(0.0)
        X_train = train_df[feature_cols].astype('float32').values
        y_train = train_df['label'].values
        
        model = LogisticRegression(max_iter=500, solver='lbfgs', random_state=42)
        model.fit(X_train, y_train)
        
        del train_df, X_train, y_train
        gc.collect()
        
    else:
        print(f"Training Logistic Regression for {f.HOP}-hop...")
        pen = 'l1' if f.HOP == 3 else 'l2'
        a = 0.001 if f.HOP == 3 else 0.0001
        
        model = SGDClassifier(loss='log_loss', penalty=pen, alpha=a, random_state=42)
        
        for i, batch in enumerate(parquet_file.iter_batches(batch_size=50000)):
            df_chunk = batch.to_pandas().fillna(0.0)
            X_chunk = df_chunk[feature_cols].astype('float32').values
            y_chunk = df_chunk['label'].values
            
            model.partial_fit(X_chunk, y_chunk, classes=[0, 1])
            print(f"Processed train batch {i+1}...")

    #evaluation
    print("\nEvaluating Train set...")
    train_hits, train_mrr = evaluate_dataset(model, train_path, feature_cols)
    print(f"Train Hits@1: {train_hits:.4f} | MRR: {train_mrr:.4f}")

    print("\nEvaluating Dev set...")
    dev_path = SRC_DIR / "track_a" / f.DEV_PATH
    dev_hits, dev_mrr = evaluate_dataset(model, dev_path, feature_cols)
    print(f"Dev Hits@1:   {dev_hits:.4f} | MRR: {dev_mrr:.4f}")

    print("\nEvaluating Test set...")
    test_path = SRC_DIR / "track_a" / f.TEST_PATH
    test_hits, test_mrr = evaluate_dataset(model, test_path, feature_cols)
    print(f"Test Hits@1:  {test_hits:.4f} | MRR: {test_mrr:.4f}\n")

if __name__ == '__main__':
    train()