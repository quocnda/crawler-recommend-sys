import pandas as pd
from preprocessing_data import full_pipeline_preprocess_data
from solution.content_base_for_item import ContentBaseBasicApproach
# def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
#     df = full_pipeline_preprocess_data(file_path)
#     print('Data loaded and preprocessed.', df.head())
#     print('------------->')
#     print(df.iloc[1])
#     vector_feature = build_features_transform_for_item(df)
#     print('Vector features built.')
#     print(vector_feature)
#     df = transform_for_item(df, vector_feature)
#     print('-------------')
#     return df
# # print(load_and_preprocess_data("/home/quoc/crawl-company/out_2.csv"))

# data = pd.read_csv("/home/quoc/crawl-company/out_2.csv")
# outsource_company = "https://www.linkedin.com/company/instinctoolscompany/"
# data = full_pipeline_preprocess_data("/home/quoc/crawl-company/out_2.csv")
# vector_feature = build_features_transform_for_item(data)
# profile, hist = build_outsource_profile(data, vector_feature, outsource_company)

# print('PROFILE :', profile)
# print('HIST :', hist)

def main():
    data_path = "/home/quoc/crawl-company/out_2.csv"
    
    
    # data_raw = full_pipeline_preprocess_data(data_path)
    # approach_content_base = ContentBaseBasicApproach(data_raw,)