from pathlib import Path
import pandas as pd

eval_df = pd.read_csv(Path('data/eval_dataset.csv'), index_col=0)

# Define baseline as always selecting the first result of the search engine
baseline_data = []
for row in eval_df.fillna('').to_dict('records'):
    first_code_result = eval(row['search_engine_results_top_1'])[0]['code']
    relevant_codes = row['relevant_results'].split('|')
    is_tp = first_code_result in relevant_codes
    is_fp = first_code_result not in relevant_codes
    baseline_data.append({
        'true_positive': is_tp,
        'false_positive': is_fp,
        'true_negative': False,
        'false_negative': False,
    })

metrics = pd.DataFrame.from_records(baseline_data).sum()

f1_score = metrics['true_positive']/(metrics['true_positive'] + (metrics['false_positive']+metrics['false_negative'])/2)

print(f"Baseline F1 score - first search result selection: {f1_score}")