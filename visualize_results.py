"""
Generate visualizations for model comparison results.

Creates:
1. Metric comparison bar chart - compare all metrics across models
2. Score distribution histograms - show distribution of composite scores
3. Improvement heatmap - show percentage improvements
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class ResultVisualizer:
    """Generate visualizations for model comparison results"""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)

        # Load results
        print(f"Loading results from: {results_dir}")
        self.base_df = pd.read_json(self.results_dir / 'base_model_results.json')
        self.sft_df = pd.read_json(self.results_dir / 'sft_model_results.json')
        self.grpo_df = pd.read_json(self.results_dir / 'grpo_model_results.json')

        print(f"✓ Base Model: {len(self.base_df)} samples")
        print(f"✓ SFT Model: {len(self.sft_df)} samples")
        print(f"✓ GRPO Model: {len(self.grpo_df)} samples")

    def plot_metric_comparison(self):
        """Plot bar chart comparing all metrics across models"""
        print("\nGenerating metric comparison plot...")

        metrics = ['format_exact', 'format_approx', 'answer_exact', 'answer_numerical', 'composite']

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        models = [
            ('Base Model', self.base_df, '#3498db'),
            ('SFT Model', self.sft_df, '#2ecc71'),
            ('GRPO Model', self.grpo_df, '#e74c3c')
        ]

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            means = [df[metric].mean() for _, df, _ in models]
            stds = [df[metric].std() for _, df, _ in models]
            names = [name for name, _, _ in models]
            colors = [color for _, _, color in models]

            bars = ax.bar(names, means, yerr=stds, capsize=5, alpha=0.8, color=colors)
            ax.set_ylabel('Score', fontsize=11)
            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.set_ylim(bottom=0, top=max(means) * 1.2)

            # Add value labels on bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + stds[list(bars).index(bar)] + 0.02,
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

            # Add grid
            ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Remove last subplot (unused)
        fig.delaxes(axes[-1])

        plt.tight_layout()
        output_path = self.results_dir / 'metric_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved to: {output_path}")

    def plot_score_distributions(self):
        """Plot distribution of composite scores for each model"""
        print("\nGenerating score distribution plot...")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        dfs = [self.base_df, self.sft_df, self.grpo_df]
        titles = ['Base Model', 'SFT Model', 'GRPO Model']
        colors = ['#3498db', '#2ecc71', '#e74c3c']

        for idx, (df, title, color) in enumerate(zip(dfs, titles, colors)):
            ax = axes[idx]

            # Plot histogram
            n, bins, patches = ax.hist(df['composite'], bins=20, alpha=0.7,
                                       color=color, edgecolor='black', linewidth=1.2)

            # Add mean line
            mean_score = df['composite'].mean()
            ax.axvline(mean_score, color='darkred', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_score:.3f}', alpha=0.8)

            ax.set_xlabel('Composite Score', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)

        plt.tight_layout()
        output_path = self.results_dir / 'score_distributions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved to: {output_path}")

    def plot_improvement_heatmap(self):
        """Plot heatmap showing percentage improvements"""
        print("\nGenerating improvement heatmap...")

        # Calculate improvements
        improvements = {
            'SFT vs Base': {
                'Format Exact': (self.sft_df['format_exact'].mean() - self.base_df['format_exact'].mean())
                               / max(self.base_df['format_exact'].mean(), 0.001) * 100,
                'Answer Exact': (self.sft_df['answer_exact'].mean() - self.base_df['answer_exact'].mean())
                               / max(self.base_df['answer_exact'].mean(), 0.001) * 100,
                'Composite': (self.sft_df['composite'].mean() - self.base_df['composite'].mean())
                            / max(self.base_df['composite'].mean(), 0.001) * 100,
            },
            'GRPO vs SFT': {
                'Format Exact': (self.grpo_df['format_exact'].mean() - self.sft_df['format_exact'].mean())
                               / max(self.sft_df['format_exact'].mean(), 0.001) * 100,
                'Answer Exact': (self.grpo_df['answer_exact'].mean() - self.sft_df['answer_exact'].mean())
                               / max(self.sft_df['answer_exact'].mean(), 0.001) * 100,
                'Composite': (self.grpo_df['composite'].mean() - self.sft_df['composite'].mean())
                            / max(self.sft_df['composite'].mean(), 0.001) * 100,
            }
        }

        improv_df = pd.DataFrame(improvements).T

        fig, ax = plt.subplots(figsize=(10, 6))

        # Create heatmap
        sns.heatmap(improv_df, annot=True, fmt='.1f', cmap='RdYlGn',
                   center=0, cbar_kws={'label': 'Improvement (%)'},
                   linewidths=1, linecolor='white', ax=ax)

        ax.set_title('Percentage Improvements', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('', fontsize=12)
        ax.set_ylabel('', fontsize=12)

        # Adjust tick labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=11)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=11)

        plt.tight_layout()
        output_path = self.results_dir / 'improvement_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved to: {output_path}")

    def generate_summary_text(self):
        """Generate a text summary of results"""
        print("\nGenerating text summary...")

        summary = f"""
{'='*60}
MODEL COMPARISON SUMMARY
{'='*60}

Number of samples: {len(self.base_df)}

COMPOSITE SCORES:
- Base Model:   {self.base_df['composite'].mean():.3f} ± {self.base_df['composite'].std():.3f}
- SFT Model:    {self.sft_df['composite'].mean():.3f} ± {self.sft_df['composite'].std():.3f}
- GRPO Model:   {self.grpo_df['composite'].mean():.3f} ± {self.grpo_df['composite'].std():.3f}

FORMAT COMPLIANCE (Exact):
- Base Model:   {self.base_df['format_exact'].mean():.1%}
- SFT Model:    {self.sft_df['format_exact'].mean():.1%}
- GRPO Model:   {self.grpo_df['format_exact'].mean():.1%}

ANSWER ACCURACY (Exact):
- Base Model:   {self.base_df['answer_exact'].mean():.1%}
- SFT Model:    {self.sft_df['answer_exact'].mean():.1%}
- GRPO Model:   {self.grpo_df['answer_exact'].mean():.1%}

IMPROVEMENTS:
SFT vs Base:    {(self.sft_df['composite'].mean() - self.base_df['composite'].mean()) / max(self.base_df['composite'].mean(), 0.001) * 100:.1f}%
GRPO vs SFT:    {(self.grpo_df['composite'].mean() - self.sft_df['composite'].mean()) / max(self.sft_df['composite'].mean(), 0.001) * 100:.1f}%
GRPO vs Base:   {(self.grpo_df['composite'].mean() - self.base_df['composite'].mean()) / max(self.base_df['composite'].mean(), 0.001) * 100:.1f}%

{'='*60}
"""

        output_path = self.results_dir / 'summary.txt'
        with open(output_path, 'w') as f:
            f.write(summary)

        print(summary)
        print(f"✓ Saved to: {output_path}")

    def generate_all_plots(self):
        """Generate all visualizations"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)

        self.plot_metric_comparison()
        self.plot_score_distributions()
        self.plot_improvement_heatmap()
        self.generate_summary_text()

        print("\n" + "="*60)
        print("VISUALIZATION COMPLETED!")
        print("="*60)
        print(f"\nAll plots saved to: {self.results_dir}")
        print("  - metric_comparison.png")
        print("  - score_distributions.png")
        print("  - improvement_heatmap.png")
        print("  - summary.txt")


if __name__ == "__main__":
    import sys

    # Get results directory from command line or use default
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Use default path
        BASE_DIR = Path(__file__).parent.parent
        results_dir = str(BASE_DIR / "evaluation_results")

    visualizer = ResultVisualizer(results_dir)
    visualizer.generate_all_plots()
