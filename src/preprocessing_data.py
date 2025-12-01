"""
Data Preprocessing Module
=========================

Module for loading and preprocessing recommendation system data.

Input:
    - CSV file with columns: reviewer_company, Industry, Location, Client size,
      background, Services, Project description, website_url,
      services_company_outsource, description_company_outsource

Output:
    - Cleaned DataFrame with standardized column names and parsed fields

Usage:
    >>> from preprocessing_data import DataPreprocessor, preprocess_data
    >>> df = preprocess_data("path/to/data.csv")
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# =============================================================================
# Constants
# =============================================================================

# Column name mappings from raw to cleaned
COLUMN_MAPPING = {
    'reviewer_company': 'company_lead_name',
    'Industry': 'industry',
    'Location': 'location',
    'Services': 'services',
    'Project description': 'project_description',
    'website_url': 'website_outsource_url',
    'Client size': 'client_size',
}

# Required columns in raw data
REQUIRED_COLUMNS = [
    'reviewer_company', 
    'Industry', 
    'Location', 
    'Client size',
    'background', 
    'Services', 
    'Project description', 
    'website_url',
    'services_company_outsource',
    'description_company_outsource'
]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ClientSize:
    """
    Parsed client size information.
    
    Attributes:
        min_size: Minimum employee count
        max_size: Maximum employee count
        raw_value: Original string value
    """
    min_size: Optional[float] = None
    max_size: Optional[float] = None
    raw_value: Optional[str] = None
    
    @property
    def midpoint(self) -> Optional[float]:
        """Calculate midpoint of size range."""
        if self.min_size is not None and self.max_size is not None:
            return (self.min_size + self.max_size) / 2
        return None
    
    @property
    def is_valid(self) -> bool:
        """Check if size data is valid."""
        return self.min_size is not None and self.max_size is not None


@dataclass
class PreprocessedRow:
    """
    Preprocessed data row.
    
    Contains all cleaned and standardized fields for a single record.
    """
    company_lead_name: str
    industry: str
    location: str
    services: str
    project_description: str
    background: str
    website_outsource_url: str
    linkedin_company_outsource: str  # User ID
    services_company_outsource: str
    description_company_outsource: str
    client_size: ClientSize = field(default_factory=ClientSize)


# =============================================================================
# Preprocessing Functions
# =============================================================================

def parse_client_size(value: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse client size string to min/max employee counts.
    
    Args:
        value: Client size string (e.g., "51-200", "100+", "10,000-50,000")
    
    Returns:
        Tuple of (min_size, max_size) or (None, None) if unparseable
    
    Examples:
        >>> parse_client_size("51-200")
        (51.0, 200.0)
        >>> parse_client_size("100+")
        (100.0, 100.0)
        >>> parse_client_size("10,000-50,000")
        (10000.0, 50000.0)
        >>> parse_client_size(None)
        (None, None)
    """
    if pd.isna(value):
        return (None, None)
    
    s = str(value).strip()
    
    # Handle invalid values
    if s.lower() in {'unknown', 'n/a', '', 'nan'}:
        return (None, None)
    
    # Extract numbers (handle commas in large numbers)
    nums = re.findall(r"\d+", s.replace(',', ''))
    
    if len(nums) >= 2:
        try:
            min_val = float(nums[0])
            max_val = float(nums[1])
            return (min_val, max_val)
        except ValueError:
            return (None, None)
    
    if len(nums) == 1:
        try:
            val = float(nums[0])
            return (val, val)
        except ValueError:
            return (None, None)
    
    return (None, None)


def normalize_url(url: Optional[str]) -> str:
    """
    Normalize website URL for use as identifier.
    
    Args:
        url: Raw website URL
    
    Returns:
        Cleaned URL string
    
    Examples:
        >>> normalize_url("https://example.com/")
        "httpsexample.com"
        >>> normalize_url("Example.COM")
        "example.com"
    """
    if pd.isna(url):
        return ""
    return str(url).strip().lower().replace("/", "")


def clean_text_field(value: Optional[str]) -> str:
    """
    Clean text field by handling NaN and stripping whitespace.
    
    Args:
        value: Raw text value
    
    Returns:
        Cleaned text string
    """
    if pd.isna(value):
        return ""
    return str(value).strip()


# =============================================================================
# Main Preprocessor Class
# =============================================================================

class DataPreprocessor:
    """
    Data preprocessor for recommendation system data.
    
    Handles loading, cleaning, and transforming raw CSV data into
    a standardized format for the recommendation pipeline.
    
    Attributes:
        file_path: Path to input CSV file
        columns_extract: List of columns to extract from raw data
        data: Raw DataFrame (after loading)
        processed_data: Cleaned DataFrame (after processing)
    
    Example:
        >>> preprocessor = DataPreprocessor("data/raw_data.csv")
        >>> df = preprocessor.process()
        >>> print(df.columns.tolist())
        ['company_lead_name', 'industry', 'location', 'services', ...]
    """
    
    def __init__(
        self, 
        file_path: Union[str, Path],
        columns_extract: Optional[List[str]] = None
    ):
        """
        Initialize preprocessor.
        
        Args:
            file_path: Path to CSV file
            columns_extract: Columns to extract (default: REQUIRED_COLUMNS)
        """
        self.file_path = Path(file_path)
        self.columns_extract = columns_extract or REQUIRED_COLUMNS
        self.data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        
        # Load data on initialization
        self._load_data()
    
    def _load_data(self) -> None:
        """Load data from CSV file."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        
        self.data = pd.read_csv(self.file_path)
        
        # Validate required columns
        missing_cols = set(self.columns_extract) - set(self.data.columns)
        if missing_cols:
            # Try to continue with available columns
            available = [c for c in self.columns_extract if c in self.data.columns]
            self.columns_extract = available
    
    def _extract_columns(self) -> pd.DataFrame:
        """Extract required columns from raw data."""
        return self.data[self.columns_extract].copy()
    
    def _handle_client_size(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse client size column into min/max employee counts.
        
        Args:
            df: Input DataFrame with 'Client size' column
        
        Returns:
            DataFrame with added 'client_min' and 'client_max' columns
        """
        df = df.copy()
        
        if 'Client size' not in df.columns:
            df['client_min'] = np.nan
            df['client_max'] = np.nan
            df['client_size'] = 'unknown'
            return df
        
        # Parse each row
        parsed = df['Client size'].apply(parse_client_size)
        
        df['client_min'] = parsed.apply(lambda x: x[0])
        df['client_max'] = parsed.apply(lambda x: x[1])
        df.rename(columns={'Client size': 'client_size'}, inplace=True)
        
        return df
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename and normalize columns.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with standardized column names
        """
        df = df.copy()
        
        # Rename columns
        rename_map = {k: v for k, v in COLUMN_MAPPING.items() if k in df.columns}
        df.rename(columns=rename_map, inplace=True)
        
        # Normalize website URL
        if 'website_outsource_url' in df.columns:
            df['website_outsource_url'] = df['website_outsource_url'].apply(normalize_url)
            df['linkedin_company_outsource'] = df['website_outsource_url']
        else:
            df['linkedin_company_outsource'] = ""
        
        return df
    
    def process(self) -> pd.DataFrame:
        """
        Run full preprocessing pipeline.
        
        Returns:
            Cleaned and standardized DataFrame
        
        Raises:
            ValueError: If data has not been loaded
        """
        if self.data is None:
            raise ValueError("No data loaded. Call _load_data() first.")
        
        # Extract columns
        df = self._extract_columns()
        
        # Parse client size
        df = self._handle_client_size(df)
        
        # Normalize columns
        df = self._normalize_columns(df)
        
        self.processed_data = df
        return df
    
    def get_stats(self) -> dict:
        """
        Get preprocessing statistics.
        
        Returns:
            Dictionary with data statistics
        """
        if self.processed_data is None:
            return {}
        
        df = self.processed_data
        
        return {
            'total_rows': len(df),
            'unique_users': df['linkedin_company_outsource'].nunique(),
            'unique_industries': df['industry'].nunique() if 'industry' in df.columns else 0,
            'valid_client_size': df['client_min'].notna().sum(),
            'columns': df.columns.tolist()
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def preprocess_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Preprocess data from CSV file.
    
    This is the main entry point for data preprocessing.
    
    Args:
        file_path: Path to CSV file
    
    Returns:
        Preprocessed DataFrame
    
    Example:
        >>> df = preprocess_data("data/sample.csv")
        >>> print(df.shape)
        (1000, 15)
    """
    preprocessor = DataPreprocessor(file_path)
    return preprocessor.process()


# Legacy function name for backward compatibility
def full_pipeline_preprocess_data(file_path: str = "/home/quoc/crawl-company/out_2.csv") -> pd.DataFrame:
    """
    Legacy function for backward compatibility.
    
    Use preprocess_data() for new code.
    
    Args:
        file_path: Path to CSV file
    
    Returns:
        Preprocessed DataFrame
    """
    return preprocess_data(file_path)


# Alias for backward compatibility
DataHandler = DataPreprocessor
