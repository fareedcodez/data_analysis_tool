"""
Data Loader Module

This module handles loading data from Excel files.
"""

import os
import pandas as pd


class DataLoader:
    """
    Class for loading data from Excel files
    """
    def __init__(self, file_path):
        """
        Initialize with file path
        
        Args:
            file_path (str): Path to the Excel file
        """
        self.file_path = file_path
        
    def load_data(self):
        """
        Load data from Excel file and return a dictionary of dataframes
        for each sheet
        
        Returns:
            dict: Dictionary of DataFrames, keyed by sheet name
        """
        try:
            # Check if file exists
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"File not found: {self.file_path}")
            
            # Get Excel file sheet names
            xl = pd.ExcelFile(self.file_path)
            sheet_names = xl.sheet_names
            
            if not sheet_names:
                raise ValueError(f"No sheets found in {self.file_path}")
            
            # Read each sheet into a separate dataframe
            dataframes = {}
            for sheet in sheet_names:
                try:
                    dataframes[sheet] = pd.read_excel(self.file_path, sheet_name=sheet)
                    print(f"Loaded sheet '{sheet}' with {dataframes[sheet].shape[0]} rows and {dataframes[sheet].shape[1]} columns")
                    
                    # Show column types
                    print(f"  Column types in '{sheet}':")
                    for col, dtype in zip(dataframes[sheet].columns, dataframes[sheet].dtypes):
                        print(f"    {col}: {dtype}")
                        
                except Exception as e:
                    print(f"Error loading sheet '{sheet}': {str(e)}")
            
            if not dataframes:
                raise ValueError("No data could be loaded from any sheet")
                
            return dataframes
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def get_sheet_info(self):
        """
        Get information about sheets without loading full data
        
        Returns:
            list: List of dictionaries with sheet information
        """
        try:
            xl = pd.ExcelFile(self.file_path)
            sheets_info = []
            
            for sheet in xl.sheet_names:
                # Read just the header row to get columns
                df_header = pd.read_excel(self.file_path, sheet_name=sheet, nrows=0)
                
                # Sample a few rows to estimate data types
                df_sample = pd.read_excel(self.file_path, sheet_name=sheet, nrows=5)
                
                sheets_info.append({
                    'name': sheet,
                    'columns': list(df_header.columns),
                    'column_count': len(df_header.columns),
                    'dtypes': {col: str(df_sample[col].dtype) for col in df_sample.columns}
                })
            
            return sheets_info
            
        except Exception as e:
            print(f"Error getting sheet info: {str(e)}")
            return []