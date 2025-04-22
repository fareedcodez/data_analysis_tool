#!/usr/bin/env python3
"""
Meta Data Analysis Project - Main Execution Module
"""

import os
import argparse
import sys
import shutil
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import modules
from src.data_loader import DataLoader
from src.data_preprocessor import DataPreprocessor
from src.data_analyzer import DataAnalyzer
from src.data_visualizer import DataVisualizer
from src.report_generator import generate_report, generate_summary_report


# Create a sample Excel file if not found
def create_sample_file(path="meta.xlsx"):
    """
    Create a sample Excel file if the target file is not found.
    This is just for demonstration purposes.
    
    Args:
        path (str): Path where to create the sample file
    """
    try:
        import pandas as pd
        import numpy as np
        
        # Create a DataFrame with sample data
        np.random.seed(42)  # For reproducibility
        
        # Create sales data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        product_categories = ['Electronics', 'Clothing', 'Home Goods', 'Books', 'Food']
        regions = ['North', 'South', 'East', 'West', 'Central']
        
        data = {
            'Date': np.random.choice(dates, 500),
            'Product': np.random.choice(product_categories, 500),
            'Region': np.random.choice(regions, 500),
            'Sales': np.round(np.random.lognormal(mean=5, sigma=1, size=500), 2),
            'Units': np.random.randint(1, 50, 500),
            'Customer_Age': np.random.normal(35, 12, 500).astype(int),
            'Satisfaction': np.random.randint(1, 6, 500),
            'Return_Customer': np.random.choice([True, False], 500, p=[0.7, 0.3])
        }
        
        sales_df = pd.DataFrame(data)
        
        # Introduce some missing values
        mask = np.random.random(size=sales_df.shape) < 0.05
        sales_df = sales_df.mask(mask)
        
        # Create performance data
        employee_data = {
            'Employee_ID': [f'EMP{i:03d}' for i in range(1, 51)],
            'Department': np.random.choice(['Sales', 'Marketing', 'IT', 'HR', 'Finance'], 50),
            'Years_Experience': np.random.randint(1, 20, 50),
            'Salary': np.random.normal(60000, 15000, 50).astype(int),
            'Performance_Score': np.random.normal(3.5, 0.8, 50).round(2),
            'Training_Hours': np.random.randint(10, 100, 50)
        }
        
        employee_df = pd.DataFrame(employee_data)
        
        # Create a writer to save multiple sheets
        with pd.ExcelWriter(path) as writer:
            sales_df.to_excel(writer, sheet_name='Sales', index=False)
            employee_df.to_excel(writer, sheet_name='Employee', index=False)
        
        print(f"Created sample file '{path}' with 2 sheets: 'Sales' and 'Employee'")
        return True
    except Exception as e:
        print(f"Failed to create sample file: {str(e)}")
        return False


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Meta Data Analysis Project')
    parser.add_argument('--file', type=str, default='meta.xlsx',
                        help='Path to the Excel file (default: meta.xlsx)')
    parser.add_argument('--vis-dir', type=str, default='visualizations',
                        help='Directory for visualizations (default: visualizations)')
    parser.add_argument('--report-dir', type=str, default='report',
                        help='Directory for the report (default: report)')
    parser.add_argument('--no-report', action='store_true',
                        help='Skip report generation')
    parser.add_argument('--no-visualizations', action='store_true',
                        help='Skip visualization generation')
    parser.add_argument('--create-sample', action='store_true',
                        help='Create a sample Excel file for demonstration')
    parser.add_argument('--summary-only', action='store_true',
                        help='Generate only summary report instead of full report')
    return parser.parse_args()


def main():
    """Main function to orchestrate the data analysis workflow"""
    start_time = datetime.now()
    
    print("=" * 80)
    print(f"META DATA ANALYSIS PROJECT - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Parse arguments
    args = parse_arguments()
    
    # Check if sample creation is requested
    if args.create_sample:
        print("\nCreating sample Excel file...")
        if create_sample_file(args.file):
            print(f"Sample file created at '{args.file}'")
        else:
            print("Failed to create sample file. Make sure pandas and numpy are installed.")
            return 1
    
    # 1. Define file path
    file_path = args.file
    print(f"\nTarget file: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        print("\nYou have the following options:")
        print("1. Place your 'meta.xlsx' file in the project directory")
        print("2. Specify a different file with --file parameter")
        print("3. Create a sample file for demonstration with --create-sample")
        print("\nExample commands:")
        print(f"python {sys.argv[0]} --file path/to/your/excel_file.xlsx")
        print(f"python {sys.argv[0]} --create-sample")
        return 1
    
    # 2. Load data
    print("\nLOADING DATA...")
    loader = DataLoader(file_path)
    dataframes = loader.load_data()
    
    if not dataframes:
        print("Failed to load data. Exiting.")
        return 1
    
    # Set name attribute for each dataframe (needed for some visualizations)
    for sheet_name, df in dataframes.items():
        df.name = sheet_name
    
    # 3. Preprocess data
    print("\nPREPROCESSING DATA...")
    preprocessor = DataPreprocessor(dataframes)
    processed_data = preprocessor.preprocess_all_sheets()
    
    # 4. Analyze data
    print("\nANALYZING DATA...")
    analyzer = DataAnalyzer(processed_data)
    analysis_results = analyzer.analyze_all_sheets()
    
    # 5. Create visualizations
    if not args.no_visualizations:
        print("\nCREATING VISUALIZATIONS...")
        visualizer = DataVisualizer(processed_data, analysis_results, output_dir=args.vis_dir)
        visualizer.create_all_visualizations()
    else:
        print("\nSkipping visualization generation...")
    
    # 6. Generate report
    if not args.no_report:
        print("\nGENERATING REPORT...")
        if args.summary_only:
            generate_summary_report(dataframes, analysis_results, output_dir=args.report_dir)
        else:
            generate_report(dataframes, processed_data, analysis_results, output_dir=args.report_dir)
            # Also generate summary report
            generate_summary_report(dataframes, analysis_results, output_dir=args.report_dir)
    else:
        print("\nSkipping report generation...")
    
    end_time = datetime.now()
    execution_time = end_time - start_time
    
    print("\nANALYSIS COMPLETE!")
    print(f"Execution time: {execution_time}")
    
    # Print output locations
    if not args.no_visualizations:
        print(f"Visualizations saved to: {os.path.abspath(args.vis_dir)}")
    if not args.no_report:
        print(f"Reports saved to: {os.path.abspath(args.report_dir)}")
    
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())