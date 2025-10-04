import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime

sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

class PaperVisualizations:
    def __init__(self, processed_df: pd.DataFrame):
        self.df = processed_df
        
    def plot_topic_distribution(self, topics_data: list, paradigm: str, save_path: str = None):
        """
        Args:
            topics_data: List of topic dictionaries from LDA
            paradigm: 'exception_based' or 'value_based'
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        topic_labels = [f"Topic {t['topic_id']}" for t in topics_data]
        top_words = [', '.join(t['words'][:5]) for t in topics_data]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(topic_labels))
        colors = sns.color_palette("viridis", len(topic_labels))
        
        bars = ax.barh(y_pos, [1] * len(topic_labels), color=colors)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{label}\n{words}" for label, words in zip(topic_labels, top_words)],
                          fontsize=9)
        ax.set_xlabel('Representative Terms', fontsize=12, fontweight='bold')
        
        title_map = {
            'exception_based': 'Exception-Based Error Handling Topics',
            'value_based': 'Value-Based Error Handling Topics'
        }
        ax.set_title(title_map.get(paradigm, 'Topics'), fontsize=14, fontweight='bold', pad=20)
        
        ax.set_xlim(0, 1.2)
        ax.set_xticks([])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved topic distribution to {save_path}")
        
        plt.show()
    
    def plot_temporal_sentiment(self, temporal_df: pd.DataFrame, save_path: str = None):
        """
        Args:
            temporal_df: DataFrame with quarterly sentiment data
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Prepare data
        temporal_df['quarter_str'] = temporal_df['quarter'].astype(str)
        
        for ptype in ['exception_based', 'value_based']:
            subset = temporal_df[temporal_df['type'] == ptype].copy()
            subset = subset.sort_values('quarter')
            
            # Convert quarter to datetime for plotting
            subset['quarter_date'] = subset['quarter'].apply(lambda x: x.to_timestamp())
            
            label_map = {
                'exception_based': 'Exception-Based',
                'value_based': 'Value-Based'
            }
            
            ax.plot(subset['quarter_date'], 
                   subset['avg_sentiment'], 
                   marker='o', 
                   linewidth=2.5,
                   markersize=6,
                   label=label_map[ptype])
            
            # Add trend line
            if len(subset) > 3:
                x_numeric = np.arange(len(subset))
                z = np.polyfit(x_numeric, subset['avg_sentiment'], 1)
                p = np.poly1d(z)
                ax.plot(subset['quarter_date'], p(x_numeric), 
                       linestyle='--', alpha=0.5, linewidth=1.5)
        
        ax.set_xlabel('Time Period (Quarterly)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Sentiment Score', fontsize=12, fontweight='bold')
        ax.set_title('Temporal Trends in Error Handling Sentiment', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved temporal sentiment to {save_path}")
        
        plt.show()
    
    def plot_sentiment_comparison(self, save_path: str = None):
        """
        Args:
            save_path: Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Filter to main types
        plot_data = self.df[self.df['refined_type'].isin(['exception_based', 'value_based'])]
        
        # Sentiment scores
        sns.boxplot(data=plot_data, x='refined_type', y='compound', ax=ax1)
        ax1.set_xlabel('Error Handling Paradigm', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Sentiment Score', fontsize=11, fontweight='bold')
        ax1.set_title('Overall Sentiment Distribution', fontsize=12, fontweight='bold')
        ax1.set_xticklabels(['Exception-Based', 'Value-Based'])
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Benefit-cost ratio
        sns.boxplot(data=plot_data, x='refined_type', y='benefit_cost_ratio', ax=ax2)
        ax2.set_xlabel('Error Handling Paradigm', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Benefit-Cost Ratio', fontsize=11, fontweight='bold')
        ax2.set_title('Benefit-Cost Sentiment Ratio', fontsize=12, fontweight='bold')
        ax2.set_xticklabels(['Exception-Based', 'Value-Based'])
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved sentiment comparison to {save_path}")
        
        plt.show()
    
    def plot_complexity_indicators(self, save_path: str = None):
        """
        Args:
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate per-post averages for each type
        metrics = ['nested', 'unpredictable', 'hidden', 'verbosity']
        exception_data = []
        value_data = []
        
        for metric in metrics:
            exc_subset = self.df[self.df['refined_type'] == 'exception_based']
            val_subset = self.df[self.df['refined_type'] == 'value_based']
            
            exception_data.append(exc_subset[metric].sum() / len(exc_subset))
            value_data.append(val_subset[metric].sum() / len(val_subset))
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, exception_data, width, label='Exception-Based', alpha=0.8)
        bars2 = ax.bar(x + width/2, value_data, width, label='Value-Based', alpha=0.8)
        
        ax.set_xlabel('Complexity Indicator', fontsize=12, fontweight='bold')
        ax.set_ylabel('Occurrences per Post', fontsize=12, fontweight='bold')
        ax.set_title('Complexity Indicators by Error Handling Paradigm', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.legend(fontsize=11)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved complexity indicators to {save_path}")
        
        plt.show()
    
    def plot_term_frequencies(self, term_freq_df: pd.DataFrame, paradigm: str, save_path: str = None):
        """
        Args:
            term_freq_df: DataFrame with term frequencies
            paradigm: 'exception_based' or 'value_based'
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Take top 20 terms
        top_terms = term_freq_df.head(20)
        
        # Create horizontal bar chart
        y_pos = np.arange(len(top_terms))
        ax.barh(y_pos, top_terms['per_post'], alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_terms['term'])
        ax.invert_yaxis()
        ax.set_xlabel('Occurrences per Post', fontsize=12, fontweight='bold')
        
        title_map = {
            'exception_based': 'Most Frequent Terms: Exception-Based Discussions',
            'value_based': 'Most Frequent Terms: Value-Based Discussions'
        }
        ax.set_title(title_map.get(paradigm, 'Term Frequencies'), 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved term frequencies to {save_path}")
        
        plt.show()
    
    def plot_statistical_comparison(self, save_path: str = None):
        """
        Args:
            save_path: Path to save figure
        """
        # Calculate statistics
        exc_data = self.df[self.df['refined_type'] == 'exception_based']
        val_data = self.df[self.df['refined_type'] == 'value_based']
        
        # Perform t-tests
        sentiment_ttest = stats.ttest_ind(exc_data['compound'], val_data['compound'])
        benefit_ttest = stats.ttest_ind(exc_data['benefit_cost_ratio'], val_data['benefit_cost_ratio'])
        nested_ttest = stats.ttest_ind(exc_data['nested'], val_data['nested'])
        
        comparison_data = {
            'Metric': [
                'Sample Size',
                'Mean Sentiment',
                'Std Sentiment',
                'Mean Benefit-Cost',
                'Std Benefit-Cost',
                'Mean Nested (per post)',
                'Maintainability %'
            ],
            'Exception-Based': [
                len(exc_data),
                f"{exc_data['compound'].mean():.3f}",
                f"{exc_data['compound'].std():.3f}",
                f"{exc_data['benefit_cost_ratio'].mean():.3f}",
                f"{exc_data['benefit_cost_ratio'].std():.3f}",
                f"{(exc_data['nested'].sum() / len(exc_data)):.3f}",
                f"{((exc_data['maintainability'] > 0).sum() / len(exc_data) * 100):.1f}%"
            ],
            'Value-Based': [
                len(val_data),
                f"{val_data['compound'].mean():.3f}",
                f"{val_data['compound'].std():.3f}",
                f"{val_data['benefit_cost_ratio'].mean():.3f}",
                f"{val_data['benefit_cost_ratio'].std():.3f}",
                f"{(val_data['nested'].sum() / len(val_data)):.3f}",
                f"{((val_data['maintainability'] > 0).sum() / len(val_data) * 100):.1f}%"
            ],
            'p-value': [
                '-',
                f"{sentiment_ttest.pvalue:.4f}",
                '-',
                f"{benefit_ttest.pvalue:.4f}",
                '-',
                f"{nested_ttest.pvalue:.4f}",
                '-'
            ]
        }
        
        df_table = pd.DataFrame(comparison_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df_table.values,
                        colLabels=df_table.columns,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.25, 0.25, 0.25, 0.15])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(df_table.columns)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(df_table) + 1):
            for j in range(len(df_table.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#E7E6E6')
        
        plt.suptitle('Statistical Comparison of Error Handling Paradigms',
                    fontsize=14, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved statistical comparison to {save_path}")
        
        plt.show()
    
    def generate_all_figures(self, output_dir: str = 'figures'):
        """
        Args:
            output_dir: Directory to save figures
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating all paper figures...")
        
        print("\n1. Sentiment comparison...")
        self.plot_sentiment_comparison(f'{output_dir}/sentiment_comparison.png')
        
        print("\n2. Complexity indicators...")
        self.plot_complexity_indicators(f'{output_dir}/complexity_indicators.png')
        
        print("\n3. Statistical comparison...")
        self.plot_statistical_comparison(f'{output_dir}/statistical_comparison.png')
        
        print("\n=== All figures generated ===")
        print(f"Figures saved to '{output_dir}/' directory")


if __name__ == "__main__":
    df = pd.read_csv("analysis_results_processed.csv")
    
    viz = PaperVisualizations(df)
    
    viz.generate_all_figures()
    
    print("\n complete!")