"""
Data Visualizer Module

This module creates visualizations for the analyzed data.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates


class DataVisualizer:
    """
    Class for creating data visualizations
    """
    def __init__(self, processed_data, analysis_results, output_dir='visualizations'):
        """
        Initialize with processed data and analysis results
        
        Args:
            processed_data (dict): Dictionary of preprocessed DataFrames
            analysis_results (dict): Dictionary of analysis results
            output_dir (str): Directory to save visualizations
        """
        self.processed_data = processed_data
        self.analysis_results = analysis_results
        self.output_dir = output_dir
        
        # Set default styles
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("Set2")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def create_all_visualizations(self):
        """
        Create all visualizations for all sheets
        """
        for sheet_name, df in self.processed_data.items():
            print(f"\nCreating visualizations for sheet: {sheet_name}")
            
            # Create folder for this sheet's visualizations
            sheet_dir = os.path.join(self.output_dir, sheet_name)
            os.makedirs(sheet_dir, exist_ok=True)
            
            # 1. Numeric columns distributions
            self.plot_numeric_distributions(df, sheet_dir)
            
            # 2. Categorical columns distributions
            self.plot_categorical_distributions(df, sheet_dir)
            
            # 3. Correlation heatmap (if correlations exist)
            if (self.analysis_results[sheet_name]["correlation"] and 
                'pearson' in self.analysis_results[sheet_name]["correlation"]):
                self.plot_correlation_heatmap(df, sheet_dir)
            
            # 4. Time series plots for date columns
            self.plot_time_series(df, sheet_dir)
            
            # 5. Pair plots for selected numeric columns
            self.create_pair_plots(df, sheet_dir)
            
            # 6. Categorical relationship visualizations
            self.plot_categorical_relationships(df, sheet_dir)
            
            # 7. Box plots for category vs numeric relationships
            self.plot_catnum_relationships(df, sheet_dir)
    
    def plot_numeric_distributions(self, df, output_dir):
        """
        Create histograms and boxplots for numeric columns
        
        Args:
            df (DataFrame): DataFrame to visualize
            output_dir (str): Directory to save visualizations
        """
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) == 0:
            print("No numeric columns to visualize")
            return
        
        # Limit to max 15 columns for visualization
        if len(numeric_cols) > 15:
            print(f"Limiting visualization to first 15 of {len(numeric_cols)} numeric columns")
            numeric_cols = numeric_cols[:15]
        
        # Create histograms
        for col in numeric_cols:
            plt.figure(figsize=(10, 6))
            
            # Histogram with KDE
            sns.histplot(df[col].dropna(), kde=True)
            
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.tight_layout()
            
            # Save figure
            file_path = os.path.join(output_dir, f'hist_{col}.png')
            plt.savefig(file_path)
            plt.close()
            
            # Create boxplot
            plt.figure(figsize=(10, 4))
            sns.boxplot(x=df[col].dropna())
            plt.title(f'Boxplot of {col}')
            plt.tight_layout()
            
            file_path = os.path.join(output_dir, f'box_{col}.png')
            plt.savefig(file_path)
            plt.close()
        
        # Create a combined distribution plot for key numeric variables
        if len(numeric_cols) >= 2:
            # Select up to 4 numeric columns
            selected_cols = numeric_cols[:min(4, len(numeric_cols))]
            
            plt.figure(figsize=(12, 10))
            for i, col in enumerate(selected_cols, 1):
                plt.subplot(2, 2, i)
                sns.histplot(df[col].dropna(), kde=True)
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
            
            plt.tight_layout()
            file_path = os.path.join(output_dir, 'combined_distributions.png')
            plt.savefig(file_path)
            plt.close()
        
        print(f"Created distribution plots for {len(numeric_cols)} numeric columns")
    
    def plot_categorical_distributions(self, df, output_dir):
        """
        Create bar charts for categorical columns
        
        Args:
            df (DataFrame): DataFrame to visualize
            output_dir (str): Directory to save visualizations
        """
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(cat_cols) == 0:
            print("No categorical columns to visualize")
            return
        
        # Limit to max 10 columns
        if len(cat_cols) > 10:
            print(f"Limiting visualization to first 10 of {len(cat_cols)} categorical columns")
            cat_cols = cat_cols[:10]
        
        for col in cat_cols:
            # Get value counts but limit to top 20 categories
            value_counts = df[col].value_counts().head(20)
            
            # Skip if too many or too few unique values
            if len(value_counts) <= 1 or len(value_counts) > 30:
                continue
                
            plt.figure(figsize=(12, 8))
            
            # Create bar plot
            ax = sns.barplot(x=value_counts.index, y=value_counts.values)
            
            # Add value labels on top of bars
            for i, v in enumerate(value_counts.values):
                ax.text(i, v + 0.1, str(int(v)), ha='center')
            
            plt.title(f'Distribution of {col} (Top 20 Categories)')
            plt.xticks(rotation=45, ha='right')
            plt.xlabel(col)
            plt.ylabel('Count')
            
            # Ensure y-axis uses integers
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            
            plt.tight_layout()
            
            # Save figure
            file_path = os.path.join(output_dir, f'bar_{col}.png')
            plt.savefig(file_path)
            plt.close()
        
        # Create a combined pie chart for categorical columns with few categories
        small_cat_cols = [col for col in cat_cols if 2 <= df[col].nunique() <= 5]
        
        if len(small_cat_cols) >= 1:
            # Select up to 4 categorical columns
            selected_cols = small_cat_cols[:min(4, len(small_cat_cols))]
            
            plt.figure(figsize=(12, 10))
            for i, col in enumerate(selected_cols, 1):
                plt.subplot(2, 2, i)
                df[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
                plt.title(f'Distribution of {col}')
                plt.ylabel('')  # Remove 'None' ylabel
            
            plt.tight_layout()
            file_path = os.path.join(output_dir, 'combined_pie_charts.png')
            plt.savefig(file_path)
            plt.close()
        
        print(f"Created bar plots for categorical columns")
    
    def plot_correlation_heatmap(self, df, output_dir):
        """
        Create correlation heatmap for numeric columns
        
        Args:
            df (DataFrame): DataFrame to visualize
            output_dir (str): Directory to save visualizations
        """
        # Get numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.shape[1] < 2:
            return
        
        # Limit to 15 columns max to keep the heatmap readable
        if numeric_df.shape[1] > 15:
            print(f"Limiting correlation heatmap to first 15 of {numeric_df.shape[1]} numeric columns")
            numeric_df = numeric_df.iloc[:, :15]
        
        # Compute correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        heatmap = sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                  mask=mask, vmin=-1, vmax=1, linewidths=0.5)
        
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        
        # Save figure
        file_path = os.path.join(output_dir, 'correlation_heatmap.png')
        plt.savefig(file_path)
        plt.close()
        
        # Create a focused heatmap of the top correlated features
        if 'top_pairs' in self.analysis_results[df.name]["correlation"]:
            top_pairs = self.analysis_results[df.name]["correlation"]['top_pairs']
            if top_pairs:
                # Extract unique columns from top pairs
                top_cols = set()
                for pair in top_pairs:
                    top_cols.add(pair['col1'])
                    top_cols.add(pair['col2'])
                
                # Create focused correlation matrix
                if len(top_cols) >= 2:
                    top_cols = list(top_cols)
                    focused_corr = numeric_df[top_cols].corr()
                    
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(focused_corr, annot=True, fmt='.2f', cmap='coolwarm', 
                              linewidths=0.5, vmin=-1, vmax=1)
                    
                    plt.title('Correlation Heatmap - Top Related Features')
                    plt.tight_layout()
                    
                    file_path = os.path.join(output_dir, 'top_correlation_heatmap.png')
                    plt.savefig(file_path)
                    plt.close()
        
        print("Created correlation heatmap")
    
    def plot_time_series(self, df, output_dir):
        """
        Create time series plots for date columns
        
        Args:
            df (DataFrame): DataFrame to visualize
            output_dir (str): Directory to save visualizations
        """
        # Find date columns
        date_cols = df.select_dtypes(include=['datetime']).columns
        
        if len(date_cols) == 0:
            print("No datetime columns for time series plots")
            return
        
        # Find numeric columns for plotting against dates
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) == 0:
            print("No numeric columns to plot with datetime")
            return
        
        # Limit to first 5 numeric columns for simplicity
        if len(numeric_cols) > 5:
            numeric_cols = numeric_cols[:5]
        
        # For each date column
        for date_col in date_cols:
            # For each numeric column
            for num_col in numeric_cols:
                # Sort by date
                temp_df = df[[date_col, num_col]].dropna().sort_values(by=date_col)
                
                if len(temp_df) < 2:  # Need at least 2 points
                    continue
                
                plt.figure(figsize=(14, 7))
                plt.plot(temp_df[date_col], temp_df[num_col], marker='o', linestyle='-', alpha=0.7)
                
                plt.title(f'{num_col} over Time ({date_col})')
                plt.xlabel(date_col)
                plt.ylabel(num_col)
                plt.grid(True, alpha=0.3)
                
                # Format date axis
                plt.gcf().autofmt_xdate()
                date_range = (temp_df[date_col].max() - temp_df[date_col].min()).days
                if date_range > 365*2:  # More than 2 years
                    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                elif date_range > 60:  # More than 2 months
                    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                else:
                    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
                
                plt.tight_layout()
                
                # Save figure
                file_path = os.path.join(output_dir, f'timeseries_{date_col}_{num_col}.png')
                plt.savefig(file_path)
                plt.close()
                
                # Create moving average if enough data points
                if len(temp_df) > 10:
                    plt.figure(figsize=(14, 7))
                    
                    # Plot raw data
                    plt.plot(temp_df[date_col], temp_df[num_col], 'o-', alpha=0.4, label='Raw data')
                    
                    # Calculate moving average (window size depends on data length)
                    window = max(3, len(temp_df) // 10)
                    temp_df['rolling_avg'] = temp_df[num_col].rolling(window=window, center=True).mean()
                    
                    # Plot moving average
                    plt.plot(temp_df[date_col], temp_df['rolling_avg'], 'r-', linewidth=2, 
                            label=f'Moving average (window={window})')
                    
                    plt.title(f'{num_col} over Time with Moving Average ({date_col})')
                    plt.xlabel(date_col)
                    plt.ylabel(num_col)
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    
                    # Format date axis
                    plt.gcf().autofmt_xdate()
                    if date_range > 365*2:  # More than 2 years
                        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                    elif date_range > 60:  # More than 2 months
                        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                    else:
                        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
                    
                    plt.tight_layout()
                    
                    # Save figure
                    file_path = os.path.join(output_dir, f'timeseries_ma_{date_col}_{num_col}.png')
                    plt.savefig(file_path)
                    plt.close()
        
        print("Created time series plots")
    
    def create_pair_plots(self, df, output_dir):
        """
        Create pair plots for selected numeric columns
        
        Args:
            df (DataFrame): DataFrame to visualize
            output_dir (str): Directory to save visualizations
        """
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) < 2:
            print("Not enough numeric columns for pair plots")
            return
        
        # Select up to 5 columns for pair plot to avoid overcrowding
        if len(numeric_cols) > 5:
            print(f"Selecting first 5 of {len(numeric_cols)} numeric columns for pair plot")
            selected_cols = numeric_cols[:5]
        else:
            selected_cols = numeric_cols
        
        # Create pair plot
        try:
            pair_plot = sns.pairplot(df[selected_cols], height=2.5, corner=True)
            plt.suptitle('Pair Plot of Numeric Variables', y=1.02)
            
            # Save figure
            file_path = os.path.join(output_dir, 'pair_plot.png')
            pair_plot.savefig(file_path)
            plt.close()
            
            print("Created pair plot")
        except Exception as e:
            print(f"Error creating pair plot: {str(e)}")
        
        # If we have categorical columns, create a colored pair plot
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0 and len(selected_cols) > 1:
            # Find a categorical column with few categories
            for cat_col in cat_cols:
                if 2 <= df[cat_col].nunique() <= 5:
                    try:
                        # Create pair plot with coloring
                        col_pair_plot = sns.pairplot(df[selected_cols + [cat_col]], 
                                                 hue=cat_col, height=2.5, corner=True)
                        
                        plt.suptitle(f'Pair Plot by {cat_col}', y=1.02)
                        
                        # Save figure
                        file_path = os.path.join(output_dir, f'pair_plot_by_{cat_col}.png')
                        col_pair_plot.savefig(file_path)
                        plt.close()
                        
                        print(f"Created colored pair plot by {cat_col}")
                        break  # Just create one colored pair plot
                    except Exception as e:
                        print(f"Error creating colored pair plot: {str(e)}")
    
    def plot_categorical_relationships(self, df, output_dir):
        """
        Plot relationships between categorical variables
        
        Args:
            df (DataFrame): DataFrame to visualize
            output_dir (str): Directory to save visualizations
        """
        # Get categorical columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(cat_cols) < 2:
            return
        
        # Find categorical columns with reasonable number of categories
        valid_cat_cols = []
        for col in cat_cols:
            if 2 <= df[col].nunique() <= 10:
                valid_cat_cols.append(col)
        
        if len(valid_cat_cols) < 2:
            return
            
        # Limit to first 5 columns
        if len(valid_cat_cols) > 5:
            valid_cat_cols = valid_cat_cols[:5]
        
        # Create heatmap for categorical correlations
        for i in range(len(valid_cat_cols)):
            for j in range(i+1, len(valid_cat_cols)):
                col1 = valid_cat_cols[i]
                col2 = valid_cat_cols[j]
                
                # Create crosstab
                try:
                    # Create contingency table
                    cross_tab = pd.crosstab(df[col1], df[col2], normalize='index')
                    
                    # Create heatmap
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(cross_tab, annot=True, fmt='.2f', cmap='Blues')
                    
                    plt.title(f'Relationship between {col1} and {col2}')
                    plt.tight_layout()
                    
                    # Save figure
                    file_path = os.path.join(output_dir, f'heatmap_{col1}_vs_{col2}.png')
                    plt.savefig(file_path)
                    plt.close()
                except Exception as e:
                    print(f"Error creating categorical heatmap: {str(e)}")
        
        print("Created categorical relationship plots")
    
    def plot_catnum_relationships(self, df, output_dir):
        """
        Plot relationships between categorical and numeric variables
        
        Args:
            df (DataFrame): DataFrame to visualize
            output_dir (str): Directory to save visualizations
        """
        # Get categorical and numeric columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        num_cols = df.select_dtypes(include=['number']).columns
        
        if len(cat_cols) == 0 or len(num_cols) == 0:
            return
        
        # Find categorical columns with reasonable number of categories
        valid_cat_cols = []
        for col in cat_cols:
            if 2 <= df[col].nunique() <= 8:  # Up to 8 categories for box plots
                valid_cat_cols.append(col)
        
        if len(valid_cat_cols) == 0:
            return
            
        # Limit to first 3 categorical columns and 5 numeric columns
        if len(valid_cat_cols) > 3:
            valid_cat_cols = valid_cat_cols[:3]
            
        if len(num_cols) > 5:
            num_cols = num_cols[:5]
        
        # Create box plots to show distribution by category
        for cat_col in valid_cat_cols:
            for num_col in num_cols:
                try:
                    plt.figure(figsize=(12, 8))
                    
                    # Create box plot
                    ax = sns.boxplot(x=cat_col, y=num_col, data=df)
                    
                    # Add data points for better visualization
                    sns.stripplot(x=cat_col, y=num_col, data=df, 
                                 size=4, color=".3", alpha=0.3)
                    
                    plt.title(f'{num_col} Distribution by {cat_col}')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    # Save figure
                    file_path = os.path.join(output_dir, f'boxplot_{cat_col}_vs_{num_col}.png')
                    plt.savefig(file_path)
                    plt.close()
                    
                    # Create violin plot for an alternative view
                    plt.figure(figsize=(12, 8))
                    sns.violinplot(x=cat_col, y=num_col, data=df, inner="quartile")
                    plt.title(f'{num_col} Violin Plot by {cat_col}')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    # Save figure
                    file_path = os.path.join(output_dir, f'violin_{cat_col}_vs_{num_col}.png')
                    plt.savefig(file_path)
                    plt.close()
                    
                except Exception as e:
                    print(f"Error creating categorical-numeric plot: {str(e)}")
        
        print("Created categorical-numeric relationship plots")