import re
import numpy as np
import pandas as pd


COLUMNS_EXTRACT = ['reviewer_company', 'Industry', 'Location', 'Client size','background'
                   , 'Services', 'Project size', 'Project description', 'linkedin Company Outsource']


def read_data(file_path: str) -> pd.DataFrame:
    """Đọc dữ liệu từ file CSV và trả về DataFrame."""
    return pd.read_csv(file_path)


def extract_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Trích xuất các cột cụ thể từ DataFrame."""
    return df[columns]


def handle_client_size(df: pd.DataFrame) -> pd.DataFrame:
    """Tách cột 'Client size' thành hai cột số: client_min và client_max.

    Quy tắc xử lý (giả định hợp lý):
    - Nếu giá trị chứa hai số (ví dụ '51-200 Employees'), client_min=51, client_max=200
    - Nếu chứa một số và có dấu '+', ví dụ '500+', gán client_min=500, client_max=500
    - Nếu chỉ một số (ví dụ '100'), gán cả client_min và client_max bằng số đó
    - Nếu thiếu/không rõ (NaN, 'Unknown', hoặc không tìm thấy số) -> NaN

    Trả về df với hai cột mới (kiểu float, NaN khi không có giá trị) và xóa cột gốc 'Client size'.
    """
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
                # if the string contains a plus sign, keep min==max==a
                return (a, a)
            except ValueError:
                return (np.nan, np.nan)
        return (np.nan, np.nan)

    parsed = df['Client size'].apply(parse_size)
    df = df.copy()
    df['client_min'] = parsed.apply(lambda x: float(x[0]) if not pd.isna(x[0]) else np.nan)
    df['client_max'] = parsed.apply(lambda x: float(x[1]) if not pd.isna(x[1]) else np.nan)
    # Xóa cột gốc
    df.drop(columns=['Client size'], inplace=True)
    return df


def handle_project_size(df: pd.DataFrame) -> pd.DataFrame:
    """Tách cột 'Project size' thành project_min và project_max (số, không có ký hiệu $).

    Xử lý hợp lý:
    - Nếu có hai số -> (min, max)
    - Nếu có một số và chứa từ chỉ 'under' hoặc 'less' -> (0, that)
    - Nếu có một số và chứa '+' hoặc 'more' -> (that, that)
    - Nếu có một số không rõ ngữ cảnh -> (that, that)
    - Nếu không có số hoặc 'unknown' -> (NaN, NaN)
    """
    def parse_project(val):
        if pd.isna(val):
            return (np.nan, np.nan)
        s = str(val).strip()
        if s.lower() in {'unknown', 'n/a', ''}:
            return (np.nan, np.nan)
        # Loại bỏ ký hiệu tiền tệ và chữ thừa
        s_clean = s.replace('$', '').replace('usd', '').lower()
        s_clean = s_clean.replace(',', '')
        nums = re.findall(r"\d+", s_clean)
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
            except ValueError:
                return (np.nan, np.nan)
            low_words = ['under', 'less', '<']
            high_words = ['more', 'plus', 'over', '+']
            if any(w in s_clean for w in low_words):
                return (0, a)
            if any(w in s_clean for w in high_words) or '+' in s_clean:
                return (a, a)
            # mặc định: đặt cả min và max bằng giá trị duy nhất
            return (a, a)
        return (np.nan, np.nan)

    parsed = df['Project size'].apply(parse_project)
    df = df.copy()
    df['project_min'] = parsed.apply(lambda x: float(x[0]) if not pd.isna(x[0]) else np.nan)
    df['project_max'] = parsed.apply(lambda x: float(x[1]) if not pd.isna(x[1]) else np.nan)
    df.drop(columns=['Project size'], inplace=True)
    return df

def format_linkedin_url(url: str) -> str:
    return url.strip().lower().rstrip('/')
def reformat_name_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Định dạng lại tên các cột trong DataFrame theo chuẩn nhất định."""
    df = df.copy()
    df.rename(columns={
        'Industry': 'industry',
        'Location': 'location',
        'Services': 'services',
        'Project description': 'project_description',
        'linkedin Company Outsource': 'linkedin_company_outsource'
    }, inplace=True)
    df = df[~df['linkedin_company_outsource'].isna()]
    df['linkedin_company_outsource'] = df['linkedin_company_outsource'].apply(format_linkedin_url)
    return df

def handle_description_project(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Project description'] = ''
    return df
def full_pipeline_preprocess_data(file_path: str  = "/home/quoc/crawl-company/out_2.csv") -> pd.DataFrame:
    data = read_data(file_path)
    extracted_column_data = extract_columns(data, COLUMNS_EXTRACT)
    extracted_column_data = handle_client_size(extracted_column_data)
    extracted_column_data = handle_project_size(extracted_column_data)
    # extracted_column_data = handle_description_project(extracted_column_data)
    reformatted_data = reformat_name_columns(extracted_column_data)
    return reformatted_data
