import re
import numpy as np
import pandas as pd

class DataHandler():
    def __init__(self, link_file: str):
        self.data = pd.read_csv(link_file)
        self.columns_extract = ['reviewer_company', 'Industry', 'Location', 'Client size','background'
                   , 'Services', 'Project description', 'website_url','services_company_outsource','description_company_outsource']

    def extract_columns(self) -> pd.DataFrame:
        """Trích xuất các cột cụ thể từ DataFrame."""
        return self.data[self.columns_extract]
    
    def format_website_url(self,url: str) -> str:
        return url.strip().lower().replace("/", "")
    def handle_client_size(self, df: pd.DataFrame) -> pd.DataFrame:
        def parse_size(val):
            if pd.isna(val):
                return (np.nan, np.nan)
            s = str(val).strip()
            if s.lower() in {'unknown', 'n/a', ''}:
                return (np.nan, np.nan)
            nums = re.findall(r"\d+", s.replace(',', ''))
            if len(nums) >= 2:
                try:
                    a = int(nums[0])
                    b = int(nums[1])
                    return (a, b)
                except ValueError:
                    return (np.nan, np.nan)
            if len(nums) == 1:
                try:
                    a = int(nums[0])
                    return (a, a)
                except ValueError:
                    return (np.nan, np.nan)
            return (np.nan, np.nan)

        parsed = df['Client size'].apply(parse_size)
        df = df.copy()
        df['client_min'] = parsed.apply(lambda x: float(x[0]) if not pd.isna(x[0]) else np.nan)
        df['client_max'] = parsed.apply(lambda x: float(x[1]) if not pd.isna(x[1]) else np.nan)
        df.rename(columns={'Client size': 'client_size'}, inplace=True)
        return df
    
    def reformat_name_columns(self,df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.rename(columns={
            'reviewer_company': 'company_lead_name',
            'Industry': 'industry',
            'Location': 'location',
            'Services': 'services',
            'Project description': 'project_description',
            'website_url': 'website_outsource_url',
        }, inplace=True)
        df['website_outsource_url'] = df['website_outsource_url'].apply(self.format_website_url)
        df['linkedin_company_outsource'] = df['website_outsource_url']
        return df
    
    def process_data(self) -> pd.DataFrame:
        df = self.extract_columns()
        df = self.handle_client_size(df)
        df = self.reformat_name_columns(df)
        return df
def full_pipeline_preprocess_data(file_path: str  = "/home/quoc/crawl-company/out_2.csv") -> pd.DataFrame:
    data_obj = DataHandler(file_path)
    df_processed = data_obj.process_data()
    return df_processed
