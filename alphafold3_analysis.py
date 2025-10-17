"""
AlphaFold 3 Prediction Analysis Script

Usage:
# Default: analyze all metrics and save heatmaps
python alphafold3_analysis.py /path/to/folder --prefix model_mouse

# Customize tick spacing for heatmaps
python alphafold3_analysis.py /path/to/folder --prefix model_name --tickspacing 50

# Display plots interactively
python alphafold3_analysis.py /path/to/folder --prefix model_name --showplot

"""

import json
import numpy as np
import pandas as pd
import string
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import argparse
import sys


# Display settings
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_data(json_file_path):
    """Load JSON data from a file."""
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        return data
    except Exception as e:
        print("Error reading the JSON file:", e)
        return None


def number_to_column_label(num):
    """Convert a number to Excel-style column label (1->A, 2->B, etc.)."""
    letters = ''
    while num > 0:
        num, remainder = divmod(num - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def calculate_chain_lengths(token_res_ids):
    """Calculate chain lengths from token residue IDs."""
    chain_lengths = []
    current_chain_length = 1
    previous_residue = token_res_ids[0]

    for residue in token_res_ids[1:]:
        if residue == previous_residue + 1:
            current_chain_length += 1
        else:
            chain_lengths.append(current_chain_length)
            current_chain_length = 1
        previous_residue = residue

    chain_lengths.append(current_chain_length)
    return chain_lengths


def get_file_paths(sample_folder, prefix):
    """Get full_data (confidences.json) and summary_confidences file paths."""
    full_data_path = os.path.join(sample_folder, f'{prefix}_confidences.json')
    summary_confidences_path = os.path.join(sample_folder, f'{prefix}_summary_confidences.json')
    model_path = os.path.join(sample_folder, f'{prefix}_model.cif')

    # Check if files exist
    if not os.path.exists(full_data_path):
        print(f"Warning: {full_data_path} not found")
        full_data_path = None
    if not os.path.exists(summary_confidences_path):
        print(f"Warning: {summary_confidences_path} not found")
        summary_confidences_path = None
    if not os.path.exists(model_path):
        print(f"Warning: {model_path} not found")
        model_path = None

    return full_data_path, summary_confidences_path, model_path


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_global_confidence(summary_confidences_path):
    """Analyze and display global confidence measures and ipTM matrix."""
    print("\n" + "="*80)
    print("GLOBAL CONFIDENCE MEASURES AND ipTM MATRIX")
    print("="*80)

    data = extract_data(summary_confidences_path)
    if data is None:
        return

    # Extract data
    chain_iptm = data.get('chain_iptm')
    chain_pair_iptm = data.get('chain_pair_iptm')
    fraction_disordered = data.get('fraction_disordered', 'N/A')
    has_clash = data.get('has_clash', 'N/A')
    iptm = data.get('iptm', 'N/A')
    num_recycles = data.get('num_recycles', 'N/A')
    ptm = data.get('ptm', 'N/A')
    ranking_score = data.get('ranking_score', 'N/A')
    chain_ptm = data.get('chain_ptm')

    # Display statistics
    print(f"\nFile path: {summary_confidences_path}\n")
    print("Global confidence measures:")
    print(f"  fraction_disordered: {fraction_disordered}")
    print(f"  has_clash: {has_clash}")
    print(f"  iptm: {iptm}")
    print(f"  num_recycles: {num_recycles}")
    print(f"  ptm: {ptm}")
    print(f"  ranking_score: {ranking_score}")

    # Chain statistics
    df_chain = pd.DataFrame({'chain_ptm': chain_ptm, 'chain_iptm': chain_iptm})
    df_chain.index = [number_to_column_label(i) for i in range(1, len(df_chain.index) + 1)]

    print("\nEntity ptm and ipTM average:")
    print(df_chain)

    # ipTM Matrix
    df_iptm = pd.DataFrame(chain_pair_iptm)
    df_iptm.index = [number_to_column_label(i) for i in range(1, len(df_iptm.index) + 1)]
    df_iptm.columns = [number_to_column_label(i) for i in range(1, len(df_iptm.columns) + 1)]

    print("\nipTM matrix:")
    print(df_iptm)


def analyze_pae_heatmap(full_data_path, tick_spacing=None, reres=False):
    """Generate Predicted Alignment Error (PAE) heatmap."""
    print("\n" + "="*80)
    print("PREDICTED ALIGNMENT ERROR (PAE) HEATMAP")
    print("="*80)

    data = extract_data(full_data_path)
    if data is None:
        return None

    pae_data = data['pae']
    token_res_ids = data['token_res_ids']

    print(f"\nFile path: {full_data_path}")
    print(f"Number of residues: {len(token_res_ids)}")

    chain_lengths = calculate_chain_lengths(token_res_ids)
    print(f"Entity lengths: {chain_lengths}")

    total_residues = sum(chain_lengths)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pae_data, cmap='viridis', cbar_kws={'label': 'Expected Position Error (Ångströms)'}, 
                square=True, ax=ax)

    if len(chain_lengths) > 1:
        chain_boundaries = np.cumsum(chain_lengths)
        chain_labels = [number_to_column_label(i+1) for i in range(len(chain_lengths))]
        chain_centers = [(start + end)/2 for start, end in zip([0]+list(chain_boundaries[:-1]), chain_boundaries)]

        for boundary in chain_boundaries[:-1]:
            ax.axhline(boundary, color='white', linewidth=1.2)
            ax.axvline(boundary, color='white', linewidth=1.2)
        
        # Chain labels on top
        for label, center in zip(chain_labels, chain_centers):
            ax.text(center, -3, label, ha='center', va='bottom', fontsize=12, color='black', clip_on=False)
        
        # Chain labels on left side (further left to avoid y-axis ticks)
        x_offset = -0.10 * total_residues
        for label, center in zip(chain_labels, chain_centers):
            ax.text(x_offset, center, label, ha='right', va='center', 
                   fontsize=12, color='black', clip_on=False)

    ax.set_title('Predicted Alignment Error (PAE) Heatmap', y=1.05)
    ax.set_xlabel('Scored residue')
    ax.set_ylabel('Aligned residue')

    # Handle tick spacing
    if reres and len(chain_lengths) > 1:
        tick_positions = []
        tick_labels = []
        start = 0
        for length in chain_lengths:
            if tick_spacing is not None:
                # Start at 1, then use the spacing
                chain_ticks = [1] + list(range(tick_spacing, length + 1, tick_spacing))
            else:
                num_ticks = 10
                step = max(1, length // num_ticks)
                chain_ticks = list(range(1, length + 1, step))
            
            # Convert to global positions
            tick_positions.extend([t + start for t in chain_ticks])
            tick_labels.extend([str(t) for t in chain_ticks])
            start += length
        
        ax.set_xticks(np.array(tick_positions) - 1)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        ax.set_yticks(np.array(tick_positions) - 1)
        ax.set_yticklabels(tick_labels)
    else:
        # Use custom tick spacing if provided, otherwise auto-calculate
        if tick_spacing is not None:
            tick_positions = [1] + list(range(tick_spacing, total_residues + 1, tick_spacing))
        else:
            num_ticks = 10
            tick_step = max(1, total_residues // num_ticks)
            tick_positions = list(range(1, total_residues + 1, tick_step))

        tick_labels = [str(i) for i in tick_positions]
        ax.set_xticks(np.array(tick_positions) - 1)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        ax.set_yticks(np.array(tick_positions) - 1)
        ax.set_yticklabels(tick_labels)

    plt.tight_layout()
    return fig


def analyze_contact_probability(full_data_path, tick_spacing=None, reres=False):
    """Generate Contact Probability heatmap."""
    print("\n" + "="*80)
    print("CONTACT PROBABILITY HEATMAP")
    print("="*80)

    data = extract_data(full_data_path)
    if data is None:
        return None

    contact_data = data['contact_probs']
    token_res_ids = data['token_res_ids']

    print(f"\nFile path: {full_data_path}")
    print(f"Number of residues: {len(token_res_ids)}")

    chain_lengths = calculate_chain_lengths(token_res_ids)
    print(f"Entity lengths: {chain_lengths}")

    total_residues = sum(chain_lengths)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(contact_data, cmap='viridis', cbar_kws={'label': 'Contact Probability'}, 
                square=True, ax=ax)

    if len(chain_lengths) > 1:
        chain_boundaries = np.cumsum(chain_lengths)
        chain_labels = [number_to_column_label(i+1) for i in range(len(chain_lengths))]
        chain_centers = [(start + end)/2 for start, end in zip([0]+list(chain_boundaries[:-1]), chain_boundaries)]
        
        for boundary in chain_boundaries[:-1]:
            ax.axhline(boundary, color='white', linewidth=1.2)
            ax.axvline(boundary, color='white', linewidth=1.2)
        
        # Chain labels on top
        for label, center in zip(chain_labels, chain_centers):
            ax.text(center, -3, label, ha='center', va='bottom', fontsize=12, color='black')
        
        # Chain labels on left side
        x_offset = -0.10 * total_residues
        for label, center in zip(chain_labels, chain_centers):
            ax.text(x_offset, center, label, ha='right', va='center', 
                   fontsize=12, color='black')

    ax.set_title('Contact Probability Heatmap', y=1.05)
    ax.set_xlabel('Residue id')
    ax.set_ylabel('Residue id')

    # Handle tick spacing
    if reres and len(chain_lengths) > 1:
        tick_positions = []
        tick_labels = []
        start = 0
        for length in chain_lengths:
            if tick_spacing is not None:
                # Start at 1, then use the spacing
                chain_ticks = [1] + list(range(tick_spacing, length + 1, tick_spacing))
            else:
                num_ticks = 10
                step = max(1, length // num_ticks)
                chain_ticks = list(range(1, length + 1, step))
            
            # Convert to global positions
            tick_positions.extend([t + start for t in chain_ticks])
            tick_labels.extend([str(t) for t in chain_ticks])
            start += length
        
        ax.set_xticks(np.array(tick_positions) - 1)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        ax.set_yticks(np.array(tick_positions) - 1)
        ax.set_yticklabels(tick_labels)
    else:
        # Use custom tick spacing if provided, otherwise auto-calculate
        if tick_spacing is not None:
            tick_positions = [1] + list(range(tick_spacing, total_residues + 1, tick_spacing))
        else:
            num_ticks = 10
            tick_step = max(1, total_residues // num_ticks)
            tick_positions = list(range(1, total_residues + 1, tick_step))

        tick_labels = [str(i) for i in tick_positions]
        ax.set_xticks(np.array(tick_positions) - 1)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        ax.set_yticks(np.array(tick_positions) - 1)
        ax.set_yticklabels(tick_labels)

    plt.tight_layout()
    return fig


def save_figures(sample_folder, prefix, figures):
    """Save all generated figures."""
    print("\n" + "="*80)
    print("SAVING ANALYSIS HEATMAPS")
    print("="*80 + "\n")

    figure_names = ['pae_heatmap', 'contact_probs_heatmap']

    for fig, name in zip(figures, figure_names):
        if fig is not None:
            path = os.path.join(sample_folder, f"{prefix}_{name}.png")
            fig.savefig(path, dpi=600, bbox_inches='tight')
            print(f"Saved {name} at {path}")


# ============================================================================
# COMMAND LINE ARGUMENT PARSING
# ============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze AlphaFold 3 prediction confidence measures and generate visualizations.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python alphafold3_analysis.py /path/to/output/folder --prefix dram1_mouse

  # Custom tick spacing for heatmaps
  python alphafold3_analysis.py /path/to/output/folder --prefix dram1_mouse --tickspacing 50

  # Display plots interactively
  python alphafold3_analysis.py /path/to/output/folder --prefix dram1_mouse --showplot
        """
    )

    parser.add_argument(
        'folder',
        type=str,
        help='Path to the folder containing AlphaFold 3 output files'
    )

    parser.add_argument(
        '--prefix',
        type=str,
        required=True,
        help='Prefix for the input files (e.g., "dram1_mouse" for files like "dram1_mouse_confidences.json")'
    )

    parser.add_argument(
        '--tickspacing',
        type=int,
        default=None,
        help='Spacing between tick marks on heatmap axes (default: auto-calculated based on data size)'
    )

    parser.add_argument(
        '--showplot',
        action='store_true',
        help='Display plots interactively (default: only save them)'
    )
    parser.add_argument(
        '--reres',
        action='store_true',
        help='Restart residue numbering for x and y axes at 1 for each chain'
    )

    return parser.parse_args()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()

    sample_folder = args.folder
    prefix = args.prefix
    tick_spacing = args.tickspacing
    display_plots = args.showplot

    # Validate folder exists
    if not os.path.exists(sample_folder):
        print(f"ERROR: Sample folder does not exist: {sample_folder}")
        sys.exit(1)

    print("="*80)
    print("ALPHAFOLD 3 PREDICTION ANALYSIS")
    print(f"\nSample folder: {sample_folder}")
    print(f"File prefix: {prefix}")
    if tick_spacing:
        print(f"Tick spacing: {tick_spacing}")

    # Get file paths
    full_data_path, summary_confidences_path, model_path = get_file_paths(sample_folder, prefix)

    if not all([full_data_path, summary_confidences_path]):
        print("\nERROR: Could not find all required files.")
        print(f"  {prefix}_confidences.json: {'Found' if full_data_path else 'NOT FOUND'}")
        print(f"  {prefix}_summary_confidences.json: {'Found' if summary_confidences_path else 'NOT FOUND'}")
        sys.exit(1)

    print(f"\nFound files:")
    print(f"  {prefix}_confidences.json: {full_data_path}")
    print(f"  {prefix}_summary_confidences.json: {summary_confidences_path}")
    if model_path:
        print(f"  {prefix}_model.cif: {model_path}")

    # Run analyses
    analyze_global_confidence(summary_confidences_path)
    fig1 = analyze_pae_heatmap(full_data_path, tick_spacing=tick_spacing, reres=args.reres)
    fig2 = analyze_contact_probability(full_data_path, tick_spacing=tick_spacing, reres=args.reres)

    # Save figures
    save_figures(sample_folder, prefix, [fig1, fig2])

    # Show all plots if requested
    if display_plots:
        plt.show()
    else:
        print("\nPlots not displayed (use --showplot to show them interactively)")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
