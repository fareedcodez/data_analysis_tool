"""
Data Preprocessor Module

This module handles data preprocessing tasks such as cleaning, 
handling missing values, and feature transformations.
"""

import numpy as np
import pandas as pd
import re
from datetime import datetime


class DataPreprocessor:
    """
    Class for preprocessing data
    """
    def __init__(self, dataframes):
        """
        Initialize with dataframes dictionary
        
        Args:
            dataframes (dict): Dictionary of DataFrames, keyed by sheet name
        """
        self.dataframes = dataframes
        self.processed_data = {}
        
    def preprocess_all_sheets(self):
        """
        Apply preprocessing to all sheets in the dataframes dictionary
        
        Returns:
            dict: Dictionary of preprocessed DataFrames
        """
        for sheet_name, df in self.dataframes.items():
            print(f"\nPreprocessing sheet: {sheet_name}")
            self.processed_data[sheet_name] = self.preprocess_dataframe(df)
        
        return self.processed_data
    
    def preprocess_dataframe(self, df):
        """
        Apply preprocessing steps to a single dataframe
        
        Args:
            df (DataFrame): Pandas DataFrame to preprocess
            
        Returns:
            DataFrame: Preprocessed DataFrame
        """
        # Create a copy to avoid modifying the original
        processed_df = df.copy()
        
        # 1. Remove duplicate rows
        initial_rows = processed_df.shape[0]
        processed_df = processed_df.drop_duplicates()
        dropped_rows = initial_rows - processed_df.shape[0]
        print(f"Removed {dropped_rows} duplicate rows")
        
        # 2. Handle column names - strip whitespace and special characters
        processed_df.columns = [self._clean_column_name(col) for col in processed_df.columns]
        
        # 3. Handle missing values
        processed_df = self._handle_missing_values(processed_df)
        
        # 4. Convert datatypes where appropriate
        processed_df = self._convert_datatypes(processed_df)
        
        # 5. Identify and handle outliers in numeric columns
        processed_df = self._handle_outliers(processed_df)
        
        # 6. Create feature transformations
        processed_df = self._create_transformations(processed_df)
        
        return processed_df
    
    def _clean_column_name(self, column_name):
        """
        Clean column names by removing/replacing special characters
        and extra whitespace
        
        Args:
            column_name (str): Original column name
            
        Returns:
            str: Cleaned column name
        """
        # Replace spaces with underscores, remove special characters
        if isinstance(column_name, str):
            clean_name = column_name.strip()
            clean_name = re.sub(r'\s+', '_', clean_name)  # Replace spaces with underscores
            clean_name = re.sub(r'[^\w\s]', '', clean_name)  # Remove special characters
            return clean_name.lower()  # Convert to lowercase
        return column_name
    
    def _handle_missing_values(self, df):
        """
        Handle missing values in the dataframe
        
        Args:
            df (DataFrame): DataFrame with missing values
            
        Returns:
            DataFrame: DataFrame with handled missing values
        """
        missing_count = df.isna().sum().sum()
        print(f"Found {missing_count} missing values")
        
        if missing_count == 0:
            return df
        
        # For numeric columns: fill with median
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                print(f"  - Filled missing values in '{col}' with median: {median_value}")
        
        # For categorical columns: fill with mode
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            if df[col].isna().sum() > 0:
                if df[col].dropna().empty:
                    # If all values are NaN, fill with "Unknown"
                    df[col] = df[col].fillna("Unknown")
                    print(f"  - Filled missing values in '{col}' with 'Unknown' (all values were NaN)")
                else:
                    mode_value = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_value)
                    print(f"  - Filled missing values in '{col}' with mode: {mode_value}")
        
        # For datetime columns: fill with previous value or next value
        date_cols = df.select_dtypes(include=['datetime']).columns
        for col in date_cols:
            if df[col].isna().sum() > 0:
                # Try forward fill first, then backward fill for any remaining NaNs
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                print(f"  - Filled missing values in '{col}' with forward/backward fill")
                
                # If still has NaNs, use the median date
                if df[col].isna().sum() > 0:
                    # Calculate median date for remaining NaNs
                    median_date = df[col].dropna().median()
                    df[col] = df[col].fillna(median_date)
                    print(f"  - Filled remaining missing values in '{col}' with median date: {median_date}")
        
        return df
    
    def _convert_datatypes(self, df):
        """
        Convert datatypes where appropriate
        
        Args:
            df (DataFrame): DataFrame to convert datatypes
            
        Returns:
            DataFrame: DataFrame with converted datatypes
        """
        # Try to convert string columns that look like dates to datetime
        object_cols = df.select_dtypes(include=['object']).columns
        
        for col in object_cols:
            # Skip empty columns
            if df[col].dropna().empty:
                continue
                
            # Get non-null sample values
            sample_values = df[col].dropna().sample(min(5, len(df[col].dropna())))
            
            if all(isinstance(val, str) for val in sample_values):
                # Check if the column contains date-like strings (common formats)
                date_pattern = r'^\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}'
                
                # Check if the first few values match a date pattern
                if all(re.match(date_pattern, str(val)) for val in sample_values):
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        print(f"  - Converted '{col}' to datetime")
                    except:
                        pass
                
                # Try to convert numeric strings to numeric
                elif all(re.match(r'^-?\d+\.?\d*$', str(val)) for val in sample_values):
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        print(f"  - Converted '{col}' to numeric")
                    except:
                        pass
        
        return df
    
    def _handle_outliers(self, df):
        """
        Identify and handle outliers in numeric columns
        
        Args:
            df (DataFrame): DataFrame to handle outliers
            
        Returns:
            DataFrame: DataFrame with handled outliers
        """
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            # Skip columns with too few values
            if len(df[col].dropna()) <= 5:
                continue
                
            # Using IQR method to detect outliers
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            if outliers > 0:
                print(f"  - Found {outliers} outliers in '{col}'")
                # Cap outliers at boundaries instead of removing them
                df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
                df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
                print(f"  - Capped outliers in '{col}' to range: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        return df
    
    def _create_transformations(self, df):
        """
        Create feature transformations
        
        Args:
            df (DataFrame): DataFrame to transform
            
        Returns:
            DataFrame: DataFrame with added transformations
        """
        # Get numeric columns for transformations
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        # Skip if too few numeric columns
        if len(numeric_cols) <= 1:
            return df
            
        # For demo purposes - in real project, this would be data dependent
        for col in numeric_cols:
            # Skip columns with too few values
            if len(df[col].dropna()) <= 5:
                continue
                
            # Add log transformation for highly skewed numeric columns
            skew = df[col].skew()
            if abs(skew) > 1:  # Highly skewed
                # Handle zero and negative values before log transform
                if (df[col] <= 0).any():
                    min_val = df[col].min()
                    if min_val <= 0:
                        shift = abs(min_val) + 1
                        df[f'{col}_log'] = np.log(df[col] + shift)
                else:
                    df[f'{col}_log'] = np.log(df[col])
                print(f"  - Added log transformation for skewed column '{col}' (skew={skew:.2f})")
                
            # Add polynomial features for selected columns
            # Only do this for a limited number of columns to avoid explosion of features
            if len(numeric_cols) <= 5 and abs(skew) <= 3:
                df[f'{col}_squared'] = df[col] ** 2
                print(f"  - Added squared transformation for '{col}'")
        
        # Add date features if datetime columns exist
        date_cols = df.select_dtypes(include=['datetime']).columns
        for col in date_cols:
            # Extract useful components from dates
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            
            # Create quarter feature
            df[f'{col}_quarter'] = df[col].dt.quarter
            
            print(f"  - Added date components for '{col}'")
        
        return df