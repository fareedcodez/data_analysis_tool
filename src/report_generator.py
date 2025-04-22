"""
Report Generator Module

This module generates HTML reports summarizing the analysis results.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime


def generate_report(dataframes, processed_data, analysis_results, output_dir='report'):
    """
    Generate a simple HTML report summarizing the analysis
    
    Args:
        dataframes (dict): Dictionary of original DataFrames
        processed_data (dict): Dictionary of preprocessed DataFrames
        analysis_results (dict): Dictionary of analysis results
        output_dir (str): Directory to save the report
    """
    # Create report directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; color: #333; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #3498db; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
            h3 {{ color: #7f8c8d; margin-top: 25px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; color: #333; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            .insights {{ background-color: #e8f4fd; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            .good {{ color: #27ae60; }}
            .warning {{ color: #e67e22; }}
            .bad {{ color: #c0392b; }}
            .viz-section {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
            .viz-item {{ border: 1px solid #ddd; padding: 10px; border-radius: 5px; }}
            .viz-item h4 {{ margin-top: 0; color: #3498db; }}
            .viz-item img {{ max-width: 100%; height: auto; }}
            .correlation-table {{ max-height: 400px; overflow-y: auto; }}
            .highlight {{ background-color: #ffffcc; }}
        </style>
    </head>
    <body>
        <h1>Data Analysis Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    """
    
    # Add summary section
    html_content += """
        <div class="summary">
            <h2>Analysis Summary</h2>
    """
    
    # Add summary for each sheet
    for sheet_name in dataframes.keys():
        original_df = dataframes[sheet_name]
        processed_df = processed_data[sheet_name]
        
        html_content += f"""
            <h3>Sheet: {sheet_name}</h3>
            <ul>
                <li>Original rows: {original_df.shape[0]}, columns: {original_df.shape[1]}</li>
                <li>After preprocessing: rows: {processed_df.shape[0]}, columns: {processed_df.shape[1]}</li>
                <li>Numeric columns: {len(processed_df.select_dtypes(include=['number']).columns)}</li>
                <li>Categorical columns: {len(processed_df.select_dtypes(include=['object', 'category']).columns)}</li>
                <li>Date columns: {len(processed_df.select_dtypes(include=['datetime']).columns)}</li>
            </ul>
        """
    
    html_content += """
        </div>
    """
    
    # Add detailed section for each sheet
    for sheet_name in dataframes.keys():
        processed_df = processed_data[sheet_name]
        
        html_content += f"""
        <h2>Detailed Analysis: {sheet_name}</h2>
        
        <h3>Column Information</h3>
        <table>
            <tr>
                <th>Column Name</th>
                <th>Data Type</th>
                <th>Non-Null Count</th>
                <th>Description</th>
            </tr>
        """
        
        # Add row for each column
        for col in processed_df.columns:
            dtype = processed_df[col].dtype
            non_null = processed_df[col].count()
            total = len(processed_df)
            null_pct = 100 - (non_null/total*100)
            
            # Determine if missing values are a concern
            if null_pct > 0:
                if null_pct > 20:
                    null_class = "bad"
                elif null_pct > 5:
                    null_class = "warning"
                else:
                    null_class = ""
                null_html = f'<span class="{null_class}">{non_null} / {total} ({(non_null/total*100):.1f}%)</span>'
            else:
                null_html = f'{non_null} / {total} (100%)'
            
            # Generate description based on dtype
            if pd.api.types.is_numeric_dtype(processed_df[col]):
                min_val = processed_df[col].min()
                max_val = processed_df[col].max()
                mean_val = processed_df[col].mean()
                median_val = processed_df[col].median()
                
                description = f"""
                <strong>Min:</strong> {min_val:.2f}, <strong>Max:</strong> {max_val:.2f}<br>
                <strong>Mean:</strong> {mean_val:.2f}, <strong>Median:</strong> {median_val:.2f}
                """
                
                # Add skewness information if available
                if sheet_name in analysis_results and "distribution_stats" in analysis_results[sheet_name]:
                    dist_stats = analysis_results[sheet_name]["distribution_stats"]
                    if col in dist_stats:
                        skew = dist_stats[col]['skewness']
                        dist_type = dist_stats[col]['distribution_type']
                        
                        if abs(skew) > 1:
                            skew_class = "warning"
                        else:
                            skew_class = ""
                            
                        description += f"""<br><strong>Distribution:</strong> <span class="{skew_class}">{dist_type}</span>"""
                
            elif pd.api.types.is_datetime64_dtype(processed_df[col]):
                min_date = processed_df[col].min()
                max_date = processed_df[col].max()
                date_range = max_date - min_date
                
                description = f"""
                <strong>Date range:</strong> {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}<br>
                <strong>Span:</strong> {date_range.days} days
                """
            else:
                unique = processed_df[col].nunique()
                unique_pct = (unique / len(processed_df)) * 100
                
                description = f"""
                <strong>Unique values:</strong> {unique} ({unique_pct:.1f}% of total rows)
                """
                
                # Add top categories if not too many
                if unique <= 10:
                    top_vals = processed_df[col].value_counts().head(5)
                    description += "<br><strong>Top categories:</strong><br>"
                    
                    for val, count in top_vals.items():
                        pct = (count / len(processed_df)) * 100
                        description += f"{val}: {count} ({pct:.1f}%)<br>"
            
            html_content += f"""
            <tr>
                <td>{col}</td>
                <td>{dtype}</td>
                <td>{null_html}</td>
                <td>{description}</td>
            </tr>
            """
        
        html_content += """
        </table>
        """
        
        # Add section for key insights
        html_content += """
        <div class="insights">
            <h3>Key Insights</h3>
        """
        
        # Add correlation insights
        if analysis_results[sheet_name]["correlation"] and 'top_pairs' in analysis_results[sheet_name]["correlation"]:
            top_pairs = analysis_results[sheet_name]["correlation"]['top_pairs']
            
            if top_pairs:
                html_content += """
                <h4>Strongest Correlations:</h4>
                <table>
                    <tr>
                        <th>Variable 1</th>
                        <th>Variable 2</th>
                        <th>Correlation</th>
                    </tr>
                """
                
                for pair in top_pairs:
                    # Set color based on correlation strength
                    corr_value = pair['value']
                    if abs(corr_value) > 0.7:
                        corr_class = "good" if corr_value > 0 else "bad"
                    elif abs(corr_value) > 0.3:
                        corr_class = ""
                    else:
                        corr_class = "warning"
                        
                    html_content += f"""
                    <tr>
                        <td>{pair['col1']}</td>
                        <td>{pair['col2']}</td>
                        <td class="{corr_class}">{corr_value:.3f}</td>
                    </tr>
                    """
                
                html_content += """
                </table>
                """
        
        # Add distribution insights
        if "distribution_stats" in analysis_results[sheet_name]:
            dist_stats = analysis_results[sheet_name]["distribution_stats"]
            
            # Count distribution types
            dist_types = [v['distribution_type'] for v in dist_stats.values() 
                          if v['distribution_type'] is not None]
            dist_counts = {t: dist_types.count(t) for t in set(dist_types)}
            
            if dist_counts:
                html_content += """
                <h4>Distribution Types:</h4>
                <ul>
                """
                
                for dist_type, count in dist_counts.items():
                    html_content += f"<li>{dist_type}: {count} column(s)</li>"
                
                html_content += """
                </ul>
                """
                
            # Add skewed columns
            skewed_cols = [col for col, stats in dist_stats.items() 
                          if abs(stats['skewness']) > 1]
            
            if skewed_cols:
                html_content += """
                <h4>Highly Skewed Columns:</h4>
                <ul>
                """
                
                for col in skewed_cols[:5]:  # Show top 5
                    skew_val = dist_stats[col]['skewness']
                    direction = "right" if skew_val > 0 else "left"
                    html_content += f"<li>{col}: {skew_val:.2f} ({direction}-skewed)</li>"
                
                if len(skewed_cols) > 5:
                    html_content += f"<li>... and {len(skewed_cols) - 5} more</li>"
                    
                html_content += """
                </ul>
                """
        
        # Add relationship insights
        if "relationships" in analysis_results[sheet_name]:
            relationships = analysis_results[sheet_name]["relationships"]
            
            # Add categorical-numeric relationships
            if 'categorical_numeric' in relationships and relationships['categorical_numeric']:
                sig_relations = [r for r in relationships['categorical_numeric'] if r['significant']]
                
                if sig_relations:
                    html_content += """
                    <h4>Significant Relationships (Categories affecting Numeric Values):</h4>
                    <ul>
                    """
                    
                    for i in range(min(5, len(sig_relations))):
                        rel = sig_relations[i]
                        html_content += f"<li>{rel['cat_var']} significantly affects {rel['num_var']} (p={rel['p_value']:.4f})</li>"
                    
                    html_content += """
                    </ul>
                    """
            
            # Add categorical-categorical relationships
            if 'categorical_categorical' in relationships and relationships['categorical_categorical']:
                sig_relations = [r for r in relationships['categorical_categorical'] if r['significant']]
                
                if sig_relations:
                    html_content += """
                    <h4>Significant Associations Between Categories:</h4>
                    <ul>
                    """
                    
                    for i in range(min(5, len(sig_relations))):
                        rel = sig_relations[i]
                        html_content += f"<li>{rel['var1']} is associated with {rel['var2']} (p={rel['p_value']:.4f})</li>"
                    
                    html_content += """
                    </ul>
                    """
        
        html_content += """
        </div>
        """
        
        # Add section for visualizations
        vis_dir = os.path.join("visualizations", sheet_name)
        if os.path.exists(vis_dir):
            html_content += f"""
            <h3>Key Visualizations</h3>
            <p>Here are some key visualizations that provide insights into the data:</p>
            """
            
            # Add correlation heatmap if exists
            heatmap_path = os.path.join(vis_dir, 'correlation_heatmap.png')
            if os.path.exists(heatmap_path):
                html_content += f"""
                <div class="viz-section">
                    <div class="viz-item">
                        <h4>Correlation Heatmap</h4>
                        <p>Shows the strength of relationships between numeric variables.</p>
                        <img src="../{heatmap_path}" alt="Correlation Heatmap">
                    </div>
                """
                
                # Check for pair plot
                pair_plot_path = os.path.join(vis_dir, 'pair_plot.png')
                if os.path.exists(pair_plot_path):
                    html_content += f"""
                    <div class="viz-item">
                        <h4>Pair Plot</h4>
                        <p>Shows relationships between pairs of numeric variables.</p>
                        <img src="../{pair_plot_path}" alt="Pair Plot">
                    </div>
                    """
                    
                html_content += """
                </div>
                """
            
            # Add distribution plots section
            html_content += """
            <h4>Distribution Plots</h4>
            <div class="viz-section">
            """
            
            # Find combined distribution plot
            combined_dist_path = os.path.join(vis_dir, 'combined_distributions.png')
            if os.path.exists(combined_dist_path):
                html_content += f"""
                <div class="viz-item">
                    <h4>Combined Distributions</h4>
                    <p>Overview of key numeric variable distributions.</p>
                    <img src="../{combined_dist_path}" alt="Combined Distributions">
                </div>
                """
            
            # Find pie charts
            pie_chart_path = os.path.join(vis_dir, 'combined_pie_charts.png')
            if os.path.exists(pie_chart_path):
                html_content += f"""
                <div class="viz-item">
                    <h4>Categorical Distributions</h4>
                    <p>Distribution of key categorical variables.</p>
                    <img src="../{pie_chart_path}" alt="Categorical Distributions">
                </div>
                """
                
            html_content += """
            </div>
            """
            
            # Add time series section if available
            time_series_files = [f for f in os.listdir(vis_dir) if f.startswith('timeseries_') and not f.startswith('timeseries_ma_')]
            if time_series_files:
                html_content += """
                <h4>Time Series Analysis</h4>
                <div class="viz-section">
                """
                
                # Add first two time series plots
                for i, file in enumerate(time_series_files[:2]):
                    html_content += f"""
                    <div class="viz-item">
                        <h4>Time Series {i+1}</h4>
                        <p>Trend analysis over time.</p>
                        <img src="../{os.path.join(vis_dir, file)}" alt="Time Series Plot">
                    </div>
                    """
                    
                html_content += """
                </div>
                """
            
            # Add relationship plots section
            boxplot_files = [f for f in os.listdir(vis_dir) if f.startswith('boxplot_')]
            if boxplot_files:
                html_content += """
                <h4>Category Relationships</h4>
                <div class="viz-section">
                """
                
                # Add first two box plots
                for i, file in enumerate(boxplot_files[:2]):
                    html_content += f"""
                    <div class="viz-item">
                        <h4>Categorical Relationship {i+1}</h4>
                        <p>How categories affect numeric values.</p>
                        <img src="../{os.path.join(vis_dir, file)}" alt="Box Plot">
                    </div>
                    """
                    
                html_content += """
                </div>
                """
    
    # Close HTML document
    html_content += """
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(os.path.join(output_dir, 'analysis_report.html'), 'w') as f:
        f.write(html_content)
    
    print(f"Generated report at {os.path.join(output_dir, 'analysis_report.html')}")


def generate_summary_report(dataframes, analysis_results, output_dir='report'):
    """
    Generate a simplified summary report with just key findings
    
    Args:
        dataframes (dict): Dictionary of DataFrames
        analysis_results (dict): Dictionary of analysis results
        output_dir (str): Directory to save the report
    """
    # Create report directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Analysis Summary</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; color: #333; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #3498db; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
            h3 {{ color: #7f8c8d; margin-top: 25px; }}
            .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            .insight {{ background-color: #e8f4fd; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            .good {{ color: #27ae60; }}
            .warning {{ color: #e67e22; }}
            .bad {{ color: #c0392b; }}
        </style>
    </head>
    <body>
        <h1>Data Analysis Summary Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
            <h2>Dataset Overview</h2>
    """
    
    # Add overview of sheets
    total_sheets = len(dataframes)
    total_rows = sum(df.shape[0] for df in dataframes.values())
    total_cols = sum(df.shape[1] for df in dataframes.values())
    
    html_content += f"""
            <p>This dataset contains {total_sheets} sheet(s) with a total of {total_rows} rows and {total_cols} columns.</p>
            <h3>Sheets:</h3>
            <ul>
    """
    
    for sheet_name, df in dataframes.items():
        html_content += f"<li><strong>{sheet_name}</strong>: {df.shape[0]} rows, {df.shape[1]} columns</li>"
    
    html_content += """
            </ul>
        </div>
        
        <h2>Key Findings</h2>
    """
    
    # Add key findings for each sheet
    for sheet_name in dataframes.keys():
        html_content += f"""
        <h3>Sheet: {sheet_name}</h3>
        <div class="insight">
        """
        
        # Add correlation insights
        if (analysis_results[sheet_name]["correlation"] and 
            'top_pairs' in analysis_results[sheet_name]["correlation"] and
            analysis_results[sheet_name]["correlation"]['top_pairs']):
            
            top_pairs = analysis_results[sheet_name]["correlation"]['top_pairs']
            
            html_content += "<h4>Key Relationships:</h4><ul>"
            
            for pair in top_pairs[:3]:  # Show top 3
                corr_value = pair['value']
                strength = "strong" if abs(corr_value) > 0.7 else "moderate"
                direction = "positive" if corr_value > 0 else "negative"
                
                html_content += f"""
                <li>There is a <strong>{strength} {direction}</strong> correlation ({corr_value:.2f}) 
                between <strong>{pair['col1']}</strong> and <strong>{pair['col2']}</strong></li>
                """
            
            html_content += "</ul>"
        
        # Add distribution insights
        if "distribution_stats" in analysis_results[sheet_name]:
            dist_stats = analysis_results[sheet_name]["distribution_stats"]
            
            # Find non-normal distributions
            skewed_cols = [col for col, stats in dist_stats.items() 
                          if abs(stats['skewness']) > 1]
            
            if skewed_cols:
                html_content += "<h4>Distribution Insights:</h4><ul>"
                
                for col in skewed_cols[:3]:  # Show top 3
                    skew_val = dist_stats[col]['skewness']
                    direction = "right" if skew_val > 0 else "left"
                    
                    html_content += f"""
                    <li><strong>{col}</strong> has a <strong>{direction}-skewed</strong> distribution, 
                    suggesting outliers or natural asymmetry</li>
                    """
                
                html_content += "</ul>"
        
        # Add relationship insights
        if "relationships" in analysis_results[sheet_name]:
            relationships = analysis_results[sheet_name]["relationships"]
            
            # Add categorical-numeric relationships
            if ('categorical_numeric' in relationships and 
                relationships['categorical_numeric']):
                
                sig_relations = [r for r in relationships['categorical_numeric'] if r['significant']]
                
                if sig_relations:
                    html_content += "<h4>Category Effects:</h4><ul>"
                    
                    for i in range(min(3, len(sig_relations))):
                        rel = sig_relations[i]
                        html_content += f"""
                        <li>The category <strong>{rel['cat_var']}</strong> has a significant effect 
                        on <strong>{rel['num_var']}</strong> values</li>
                        """
                    
                    html_content += "</ul>"
        
        html_content += """
        </div>
        """
    
    # Add recommendations section
    html_content += """
    <h2>Recommendations</h2>
    <div class="insight">
        <ul>
    """
    
    # Generic recommendations
    html_content += """
            <li>Review the full analysis report for detailed insights</li>
            <li>Examine the correlation heatmap to understand relationships between variables</li>
            <li>For skewed variables, consider transformations (log, square root) before modeling</li>
            <li>Check box plots to understand how categories affect numeric variables</li>
        </ul>
    </div>
    
    <p><a href="analysis_report.html">View Full Analysis Report</a></p>
    
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(os.path.join(output_dir, 'summary_report.html'), 'w') as f:
        f.write(html_content)
    
    print(f"Generated summary report at {os.path.join(output_dir, 'summary_report.html')}")