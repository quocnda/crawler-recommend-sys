import pandas as pd
from preprocessing_data import full_pipeline_preprocess_data
from solution.content_base_for_item import ContentBaseBasicApproach
from benchmark_data import BenchmarkOutput
from tqdm import tqdm
def get_recommendations_output(df_test: pd.DataFrame, approach: ContentBaseBasicApproach,  top_k: int) -> pd.DataFrame:
    results = pd.DataFrame()
    set_url_outsource = set()
    for idx, row in tqdm(df_test.iterrows(), total=df_test.shape[0]):
        try:
            outsource_url_company = row['linkedin_company_outsource']
            if outsource_url_company in set_url_outsource:
                continue
            print('Processing outsource URL:', outsource_url_company)
            set_url_outsource.add(outsource_url_company)
            recommended_items = approach.recommend_items(outsource_url_company, top_k)
            recommended_items['linkedin_company_outsource'] = outsource_url_company
            results = pd.concat([results, recommended_items], ignore_index=True)
        except Exception as e:
            print(f"Error processing row {idx} with URL {row['linkedin_company_outsource']}: {e}")
            continue
    readable_results = results[['linkedin_company_outsource', 'industry', 'score']]
    
    return readable_results

def main():
    data_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample.csv"
    data_test_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample_test.csv"
    
    data_raw = full_pipeline_preprocess_data(data_path)
    data_test = full_pipeline_preprocess_data(data_test_path)
    data_test['project_description'] = data_test['background']
    print('---------- Check points handle data base ----------')
    print(data_raw.columns)
    approach_content_base = ContentBaseBasicApproach(data_raw,data_test)
    print('---------- Check points build feature ----------')
    readable_results = get_recommendations_output(data_test, approach_content_base, top_k=10)
    
    benchmark = BenchmarkOutput(readable_results, data_test)
    print('---------- Evaluation Results ----------')
    summary, per_user = benchmark.evaluate_topk(k=10)
    print(summary)
    summary.to_csv('/home/ubuntu/crawl/crawler-recommend-sys/data/benchmark/summary_with_rerank_1.csv', index=False)
    print('---------- Per User Results ----------')
    print(per_user)
    per_user.to_csv('/home/ubuntu/crawl/crawler-recommend-sys/data/benchmark/per_user_with_rerank_1.csv', index=False)
main()