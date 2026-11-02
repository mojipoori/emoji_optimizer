import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tbparse import SummaryReader
import os

def create_ridge_plot(log_dir, max_iterations=50):
    """Create ridge plot using seaborn and tbparse"""
    
    # Load data using tbparse
    reader = SummaryReader(log_dir, pivot=True)
    df = reader.histograms
    
    if df.empty:
        print("No histogram data found!")
        return
    
    print(f"Found histogram data with columns: {df.columns.tolist()}")
    print(f"Data shape: {df.shape}")
    
    # Filter for gradient histograms and limit iterations
    if 'Gradients/abs_ln/counts' in df.columns and 'Gradients/abs_ln/limits' in df.columns:
        # Filter data
        df = df[df.index <= max_iterations]
        
        # Rename columns for easier access
        df = df.rename(columns={
            'Gradients/abs_ln/counts': 'counts',
            'Gradients/abs_ln/limits': 'limits'
        })
        
        # Set background
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        
        # Choose color palettes for the distributions
        pal = sns.color_palette("Oranges", 20)[5:-5]
        
        # Initialize the FacetGrid object (stacking multiple plots)
        g = sns.FacetGrid(df, row='step', hue='step', aspect=1, height=.4, palette=pal)
        
        def plot_subplots(x, color, label, data):
            ax = plt.gca()
            ax.text(0, .08, label, fontweight="bold", color=color, fontsize=10,
                    ha="left", va="center", transform=ax.transAxes)
            counts = data['counts'].iloc[0]
            limits = data['limits'].iloc[0]
            x, y = SummaryReader.histogram_to_bins(counts, limits, limits[0], limits[-1], 200)
            # Draw the densities in a few steps
            sns.lineplot(x=x, y=y, clip_on=False, color="w", lw=1)
            ax.fill_between(x, y, color=color)
        
        # Plot each subplots with df[df['step']==i]
        g.map_dataframe(plot_subplots, None)
        
        # Add a bottom line for each subplot
        # passing color=None to refline() uses the hue mapping
        g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
        # Set the subplots to overlap (i.e., height of each distribution)
        g.figure.subplots_adjust(hspace=-.9)
        # Remove axes details that don't play well with overlap
        g.set_titles("")
        g.set(yticks=[], xlabel="", ylabel="")
        g.despine(bottom=True, left=True)
        
        # Set x-axis ticks (1 unit intervals)
        plt.xticks(range(int(plt.xlim()[0]), int(plt.xlim()[1]) + 1))
        
        # Set y-axis ticks (every 3 steps)
        steps = sorted(df.index.unique())
        y_ticks = steps[::3]  # Every 3rd step
        plt.yticks(range(len(y_ticks)), [f'Step {s}' for s in y_ticks])
        
        plt.tight_layout()
        plt.show()
        
    else:
        print("Gradient histogram columns not found!")
        print(f"Available columns: {df.columns.tolist()}")

def main():
    # Set the log directory
    log_dir = "runs/resnet20-adamw-lr0.01-betas0.9-0.99-wdecay0.0005-eps1e-08-run0-seed8-warmupsteplr--linT250-linS0.001-slrSS110-slrGam0.1-20251017-225040-resetFalse"
    
    # Create ridge plot
    create_ridge_plot(log_dir, max_iterations=50)

if __name__ == '__main__':
    main()