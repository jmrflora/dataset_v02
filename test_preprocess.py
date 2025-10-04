import pandas as pd

df = pd.read_csv('analysis_results_processed.csv')

# Check a few posts
for i in range(5):
    print(f"\n{'='*60}")
    print(f"POST {i}")
    print(f"{'='*60}")
    print(f"Original title: {df.iloc[i]['title'][:100]}")
    print(f"Cleaned length: {len(df.iloc[i]['cleaned_text'])}")
    print(f"Contains 'maintain': {'maintain' in df.iloc[i]['cleaned_text'].lower()}")
    print(f"Contains 'complex': {'complex' in df.iloc[i]['cleaned_text'].lower()}")
    print(f"Contains 'nested': {'nested' in df.iloc[i]['cleaned_text'].lower()}")
    print(f"Maintainability score: {df.iloc[i]['maintainability']}")