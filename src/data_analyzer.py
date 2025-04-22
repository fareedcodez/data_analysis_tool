"""
Data Analyzer Module

This module handles exploratory data analysis tasks such as generating
summary statistics and correlation analysis.
"""

import pandas as pd
import numpy as np
from scipy import stats


class DataAnalyzer:
    """
    Class for exploratory data analysis
    """
    def __init__(self, processed_data):
        """
        Initialize with processed data
        
        Args:
            processed_data (dict): Dictionary of preprocessed DataFrames
        """
        self.processed_data = processed_data
    
    def analyze_all_sheets(self):
        """
        Run analysis on all sheets
        
        Returns:
            dict: Dictionary of analysis results for each sheet
        """
        analysis_results = {}
        
        for sheet_name, df in self.processed_data.items():
            print(f"\nAnalyzing sheet: {sheet_name}")
            
            # Basic statistics
            basic_stats = self.get_basic_stats(df)
            
            # Correlation analysis for numeric columns
            correlation = self.get_correlation(df)
            
            # Distribution analysis
            distribution_stats = self.analyze_distributions(df)
            
            # Relationship analysis
            relationships = self.analyze_relationships(df)
            
            # Group results
            analysis_results[sheet_name] = {
                "basic_stats": basic_stats,
                "correlation": correlation,
                "distribution_stats": distribution_stats,
                "relationships": relationships
            }
            
        return analysis_results
    
    def get_basic_stats(self, df):
        """
        Get basic statistics for each column
        
        Args:
            df (DataFrame): DataFrame to analyze
            
        Returns:
            dict: Dictionary of statistics for each column
        """
        # Initialize dictionary to store stats
        stats = {}
        
        # Get all numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        # For numeric columns, get comprehensive stats
        if len(numeric_cols) > 0:
            num_stats = df[numeric_cols].describe().transpose()
            # Add additional metrics
            for col in numeric_cols:
                if col in num_stats.index:
                    stats[col] = num_stats.loc[col].to_dict()
                    # Add skewness and kurtosis
                    stats[col]['skewness'] = df[col].skew()
                    stats[col]['kurtosis'] = df[col].kurtosis()
                    # Add % of missing values
                    stats[col]['missing_pct'] = df[col].isna().mean() * 100
            
            print(f"Calculated statistics for {len(numeric_cols)} numeric columns")
        
        # For categorical columns, get frequency counts
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            value_counts = df[col].value_counts().head(10).to_dict()  # Top 10 values
            unique_count = df[col].nunique()
            missing_count = df[col].isna().sum()
            
            stats[col] = {
                "unique_values": unique_count,
                "missing_values": missing_count,
                "missing_pct": (missing_count / len(df)) * 100,
                "top_values": value_counts
            }
        
        print(f"Calculated statistics for {len(cat_cols)} categorical columns")
        
        # For datetime columns
        date_cols = df.select_dtypes(include=['datetime']).columns
        for col in date_cols:
            min_date = df[col].min()
            max_date = df[col].max()
            
            stats[col] = {
                "min_date": min_date,
                "max_date": max_date,
                "range_days": (max_date - min_date).days if not pd.isna(min_date) and not pd.isna(max_date) else None,
                "missing_values": df[col].isna().sum(),
                "missing_pct": (df[col].isna().sum() / len(df)) * 100
            }
        
        print(f"Calculated statistics for {len(date_cols)} datetime columns")
        
        return stats
    
    def get_correlation(self, df):
        """
        Get correlation matrix for numeric columns
        
        Args:
            df (DataFrame): DataFrame to analyze
            
        Returns:
            dict: Dictionary representation of correlation matrix
        """
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.shape[1] > 1:  # Need at least 2 numeric columns
            try:
                # Calculate Pearson correlation
                pearson_corr = numeric_df.corr(method='pearson').round(2)
                
                # Calculate Spearman correlation (rank correlation)
                spearman_corr = numeric_df.corr(method='spearman').round(2)
                
                print(f"Calculated correlation matrix for {numeric_df.shape[1]} numeric columns")
                
                # Find strongest correlations
                corr_pairs = []
                for i in range(len(pearson_corr.columns)):
                    for j in range(i+1, len(pearson_corr.columns)):
                        col1 = pearson_corr.columns[i]
                        col2 = pearson_corr.columns[j]
                        corr_value = pearson_corr.iloc[i, j]
                        corr_pairs.append((col1, col2, abs(corr_value), corr_value))
                
                # Sort by absolute correlation (descending)
                corr_pairs.sort(key=lambda x: x[2], reverse=True)
                
                # Print top 5 correlations
                if corr_pairs:
                    print("Top correlations:")
                    for i in range(min(5, len(corr_pairs))):
                        col1, col2, _, corr_value = corr_pairs[i]
                        print(f"  - {col1} vs {col2}: {corr_value:.2f}")
                
                return {
                    'pearson': pearson_corr.to_dict(),
                    'spearman': spearman_corr.to_dict(),
                    'top_pairs': [{'col1': p[0], 'col2': p[1], 'value': p[3]} for p in corr_pairs[:5]]
                }
            except Exception as e:
                print(f"Error calculating correlation: {str(e)}")
                return None
        else:
            print("Not enough numeric columns for correlation analysis")
            return None
    
    def analyze_distributions(self, df):
        """
        Analyze the distributions of columns
        
        Args:
            df (DataFrame): DataFrame to analyze
            
        Returns:
            dict: Dictionary of distribution analysis results
        """
        distribution_results = {}
        
        # Analyze numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            # Skip columns with too many missing values
            if df[col].isna().mean() > 0.5:  # More than 50% missing
                continue
                
            # Basic statistics
            mean = df[col].mean()
            median = df[col].median()
            std = df[col].std()
            skew = df[col].skew()
            kurt = df[col].kurtosis()
            
            # Test for normality (Shapiro-Wilk test)
            # Limit to 5000 samples for performance
            sample = df[col].dropna().sample(min(5000, len(df[col].dropna())))
            try:
                shapiro_stat, shapiro_p = stats.shapiro(sample)
                is_normal = shapiro_p > 0.05  # p > 0.05 suggests normal distribution
            except:
                shapiro_stat, shapiro_p = None, None
                is_normal = None
            
            distribution_results[col] = {
                'mean': mean,
                'median': median,
                'std': std,
                'skewness': skew,
                'kurtosis': kurt,
                'normality_test': {
                    'test': 'shapiro',
                    'statistic': shapiro_stat,
                    'p_value': shapiro_p,
                    'is_normal': is_normal
                },
                'mean_median_difference': abs(mean - median),
                'distribution_type': self._determine_distribution_type(skew, kurt, is_normal)
            }
        
        # Count distribution types
        dist_types = [v['distribution_type'] for v in distribution_results.values() 
                    if v['distribution_type'] is not None]
        dist_counts = {t: dist_types.count(t) for t in set(dist_types)}
        
        if dist_counts:
            print("Distribution types detected:")
            for dist_type, count in dist_counts.items():
                print(f"  - {dist_type}: {count} column(s)")
        
        return distribution_results
    
    def _determine_distribution_type(self, skewness, kurtosis, is_normal):
        """
        Determine the likely distribution type based on statistics
        
        Args:
            skewness (float): Skewness of the distribution
            kurtosis (float): Kurtosis of the distribution
            is_normal (bool): Whether the distribution passed normality test
            
        Returns:
            str: Likely distribution type
        """
        if is_normal is None:
            return None
            
        if is_normal:
            return "Normal"
        
        if abs(skewness) < 0.5:
            if kurtosis > 3:
                return "Heavy-tailed"
            elif kurtosis < 1.5:
                return "Uniform-like"
            else:
                return "Normal-like"
        elif skewness > 1:
            return "Right-skewed"
        elif skewness < -1:
            return "Left-skewed"
        else:
            return "Slightly skewed"
    
    def analyze_relationships(self, df):
        """
        Analyze relationships between variables
        
        Args:
            df (DataFrame): DataFrame to analyze
            
        Returns:
            dict: Dictionary of relationship analysis results
        """
        relationships = {}
        
        # Get numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Skip if too few columns
        if len(numeric_cols) < 2 and len(cat_cols) < 1:
            return relationships
        
        # 1. Analyze categorical vs. numeric relationships (ANOVA)
        if len(cat_cols) > 0 and len(numeric_cols) > 0:
            cat_num_relations = []
            
            # Limit to 5 categorical and 5 numeric columns for performance
            cat_sample = list(cat_cols)[:5]
            num_sample = list(numeric_cols)[:5]
            
            for cat_col in cat_sample:
                # Skip if too many categories
                if df[cat_col].nunique() > 10:
                    continue
                    
                for num_col in num_sample:
                    # Prepare data for ANOVA
                    groups = []
                    labels = []
                    
                    for category in df[cat_col].dropna().unique():
                        # Get numeric values for this category
                        values = df[df[cat_col] == category][num_col].dropna()
                        if len(values) > 5:  # Need enough samples
                            groups.append(values)
                            labels.append(str(category))
                    
                    # Run ANOVA if we have at least 2 groups
                    if len(groups) >= 2:
                        try:
                            f_stat, p_value = stats.f_oneway(*groups)
                            significant = p_value < 0.05
                            
                            cat_num_relations.append({
                                'cat_var': cat_col,
                                'num_var': num_col,
                                'test': 'ANOVA',
                                'f_statistic': f_stat,
                                'p_value': p_value,
                                'significant': significant
                            })
                        except:
                            pass
            
            # Sort by significance and p-value
            cat_num_relations.sort(key=lambda x: (not x['significant'], x['p_value']))
            
            if cat_num_relations:
                relationships['categorical_numeric'] = cat_num_relations
                
                # Print top findings
                sig_relations = [r for r in cat_num_relations if r['significant']]
                if sig_relations:
                    print(f"Found {len(sig_relations)} significant relationships between categorical and numeric variables")
                    for i in range(min(3, len(sig_relations))):
                        rel = sig_relations[i]
                        print(f"  - {rel['cat_var']} significantly affects {rel['num_var']} (p={rel['p_value']:.4f})")
        
        # 2. Analyze categorical vs. categorical (Chi-square test)
        if len(cat_cols) >= 2:
            cat_cat_relations = []
            
            # Limit to 5 categorical columns for performance
            cat_sample = list(cat_cols)[:5]
            
            for i in range(len(cat_sample)):
                for j in range(i+1, len(cat_sample)):
                    col1 = cat_sample[i]
                    col2 = cat_sample[j]
                    
                    # Skip if too many categories
                    if df[col1].nunique() > 10 or df[col2].nunique() > 10:
                        continue
                    
                    # Create contingency table
                    try:
                        contingency = pd.crosstab(df[col1], df[col2])
                        
                        # Run chi-square test
                        chi2, p, dof, expected = stats.chi2_contingency(contingency)
                        significant = p < 0.05
                        
                        cat_cat_relations.append({
                            'var1': col1,
                            'var2': col2,
                            'test': 'chi2',
                            'statistic': chi2,
                            'p_value': p,
                            'dof': dof,
                            'significant': significant
                        })
                    except:
                        pass
            
            # Sort by significance and p-value
            cat_cat_relations.sort(key=lambda x: (not x['significant'], x['p_value']))
            
            if cat_cat_relations:
                relationships['categorical_categorical'] = cat_cat_relations
                
                # Print top findings
                sig_relations = [r for r in cat_cat_relations if r['significant']]
                if sig_relations:
                    print(f"Found {len(sig_relations)} significant associations between categorical variables")
                    for i in range(min(3, len(sig_relations))):
                        rel = sig_relations[i]
                        print(f"  - {rel['var1']} is associated with {rel['var2']} (p={rel['p_value']:.4f})")
        
        return relationships