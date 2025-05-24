import pandas as pd
import json
import os

RAW_PATH = '../data/raw/KCC_raw_data.csv'
PROCESSED_PATH = '../data/processed/KCC_processed_data.json'

def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, strip extra whitespace."""
    return ' '.join(text.lower().strip().split())

def preprocess():

    df = pd.read_csv(RAW_PATH)
    
    # Clean questions and answers
    df['questions'] = df['questions'].astype(str).apply(clean_text)
    df['answers'] = df['answers'].astype(str).apply(clean_text)
    df = df.dropna(subset=['questions', 'answers']).reset_index(drop=True)
    
    # Create logical Q&A chunks
    processed_data = []
    for idx, row in df.iterrows():
        chunk = {
            'id': idx,
            'question': row['questions'],
            'answer': row['answers']
        }
        processed_data.append(chunk)
    
    # Save as JSON
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    with open(PROCESSED_PATH, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"Preprocessing complete. Processed data saved to {PROCESSED_PATH}.")

if __name__ == '__main__':
    preprocess()
