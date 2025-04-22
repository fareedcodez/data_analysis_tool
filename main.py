#!/usr/bin/env python3
"""
Meta Data Analysis Project - Main Execution Module
"""

import os
import argparse
import sys
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import modules
from src.data_loader import DataLoader
from src.data_preprocessor import DataPreprocessor
from src.data_analyzer import DataAnalyzer
from src.data_visualizer import DataVisualizer
from src.report_generator import generate_report


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
    return parser.parse_args()


def main():
    """Main function to orchestrate the data analysis workflow"""
    start_time = datetime.now()
    
    print("=" * 80)
    print(f"META DATA ANALYSIS PROJECT - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Parse arguments
    args = parse_arguments()
    
    # 1. Define file path
    file_path = args.file
    print(f"\nTarget file: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return 1
    
    # 2. Load data
    print("\nLOADING DATA...")
    loader = DataLoader(file_path)
    dataframes = loader.load_data()
    
    if not dataframes:
        print("Failed to load data. Exiting.")
        return 1
    
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
        generate_report(dataframes, processed_data, analysis_results, output_dir=args.report_dir)
    else:
        print("\nSkipping report generation...")
    
    end_time = datetime.now()
    execution_time = end_time - start_time
    
    print("\nANALYSIS COMPLETE!")
    print(f"Execution time: {execution_time}")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())