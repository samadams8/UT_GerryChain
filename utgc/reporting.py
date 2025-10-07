import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def save_visualization(partition, step, results, counties=None, municipalities=None):
    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    partition.plot(ax=ax, cmap='tab20c')
    if municipalities is not None:
        municipalities.boundary.plot(ax=ax, color='black', linewidth=0.25, alpha=0.5)
    if counties is not None:
        counties.boundary.plot(ax=ax, color='black', linewidth=1, alpha=0.5)
    title = f"Step {step}: Muni Splits: {results.get('split_munis_count', 0)}, County Splits: {results.get('split_counties_count', 0)}"
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    plt.savefig(f"results/step_{step:05d}.png", dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()


def save_results(results, available_elections):
    print("Saving results...")
    os.makedirs("results", exist_ok=True)
    with open("results/ensemble_results.json", "w") as f:
        json.dump(results, f, indent=2)

    summary_data = []
    for result in results:
        summary_row = {
            "step": result["step"],
            "vote_share_agg": result.get("vote_share_agg", "none"),
            "split_counties_count": result.get("split_counties_count", 0),
            "split_counties_extra_parts": result.get("split_counties_extra_parts", 0),
            "split_munis_count": result.get("split_munis_count", 0),
            "split_munis_extra_parts": result.get("split_munis_extra_parts", 0),
        }

        for metric_key in ["mean_median", "partisan_bias", "efficiency_gap", "partisan_gini"]:
            if metric_key in result:
                summary_row[metric_key] = result[metric_key]

        for election in available_elections:
            if "Republican_agg_share_by_district" not in result:
                rep_total_key = f"{election}_Republican_total"
                if rep_total_key in result:
                    summary_row[rep_total_key] = result[rep_total_key]
                for metric_name in ["efficiency_gap", "mean_median", "partisan_bias"]:
                    key = f"{election}_{metric_name}"
                    if key in result:
                        summary_row[key] = result[key]
                rep_wins_key = f"{election}_Republican_wins"
                if rep_wins_key in result:
                    summary_row[rep_wins_key] = result[rep_wins_key]
                margin_pct_key = f"{election}_margin_pct_by_district"
                if margin_pct_key in result and isinstance(result[margin_pct_key], list) and len(result[margin_pct_key]) > 0:
                    valid = [x for x in result[margin_pct_key] if x is not None]
                    if len(valid) > 0:
                        summary_row[f"{election}_avg_margin_pct"] = float(sum(valid) / len(valid))

        key = "Republican_agg_share_by_district"
        if "Republican_agg_seats" in result:
            summary_row["Republican_agg_seats"] = int(result["Republican_agg_seats"]) if result["Republican_agg_seats"] is not None else None

        if key in result and isinstance(result[key], list) and len(result[key]) > 0:
            for idx, share in enumerate(result[key], start=1):
                col_name = f"Republican_agg_share_d{idx}"
                summary_row[col_name] = None if share is None else float(share)

        summary_data.append(summary_row)

    summary_df = pd.DataFrame(summary_data)
    district_cols = [c for c in summary_df.columns if c.startswith("Republican_agg_share_d")]
    non_district_cols = [c for c in summary_df.columns if c not in district_cols]
    summary_df = summary_df[non_district_cols + district_cols]
    summary_df.to_csv("results/ensemble_summary.csv", index=False)

    print(f"Results saved to results/ directory")
    print(f"Summary statistics:")
    print(f"  Munis split (avg count): {summary_df['split_munis_count'].mean():.2f}")
    print(f"  Munis extra parts (avg total): {summary_df['split_munis_extra_parts'].mean():.2f}")
    print(f"  Counties split (avg count): {summary_df['split_counties_count'].mean():.2f}")
    print(f"  Counties extra parts (avg total): {summary_df['split_counties_extra_parts'].mean():.2f}")

    if "Republican_agg_share_by_district" not in summary_df.columns:
        for election in available_elections:
            rep_col = f"{election}_Republican_total"
            if rep_col in summary_df.columns:
                print(f"  {election} - Average Republican votes: {summary_df[rep_col].mean():.0f}")
    else:
        if "Republican_agg_seats" in summary_df.columns:
            print(f"  Aggregated Republican seats (avg): {summary_df['Republican_agg_seats'].mean():.2f}")
        if "mean_median" in summary_df.columns:
            print(f"  Mean-median: {summary_df['mean_median'].mean():.3f}")
        if "partisan_bias" in summary_df.columns:
            print(f"  Partisan bias: {summary_df['partisan_bias'].mean():.3f}")
        if "efficiency_gap" in summary_df.columns:
            print(f"  Efficiency gap: {summary_df['efficiency_gap'].mean():.3f}")
        if "partisan_gini" in summary_df.columns:
            print(f"  Partisan Gini: {summary_df['partisan_gini'].mean():.3f}")


def create_partisan_histogram_plots(summary_df):
    metrics = {
        'mean_median': 'Mean-Median Difference',
        'partisan_bias': 'Partisan Bias',
        'efficiency_gap': 'Efficiency Gap',
        'partisan_gini': 'Partisan Gini'
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, (col, title) in enumerate(metrics.items()):
        if col in summary_df.columns:
            ax = axes[i]
            data = summary_df[col].dropna()
            if len(data) > 0:
                if col == 'partisan_bias':
                    share_cols = [col for col in summary_df.columns if col.startswith('Republican_agg_share_d')]
                    num_districts = len(share_cols)
                    unique_vals = sorted(data.unique())
                    bin_edges = []
                    for val in unique_vals:
                        bin_edges.extend([val - 1/(num_districts*2), val + 1/(num_districts*2)])
                    bin_edges = sorted(list(set(bin_edges)))
                    ax.hist(data, bins=bin_edges, alpha=0.7, color='#6B7280', edgecolor='white', linewidth=0.8)
                    ax.set_xticks(unique_vals)
                    ax.set_xlim([min(unique_vals) - 0.5, max(unique_vals) + 0.5])
                else:
                    # Use adaptive bins to avoid small-range errors
                    ax.hist(data, bins='auto', alpha=0.7, color='#6B7280', edgecolor='white', linewidth=0.8)
                ax.set_title(f'Distribution of {title}', fontsize=12, fontweight='bold')
                ax.set_xlabel(title)
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                mean_val = data.mean()
                median_val = data.median()
                ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.3f}')
                ax.axvline(median_val, color='orange', linestyle='--', alpha=0.8, label=f'Median: {median_val:.3f}')
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, f'No data for {title}', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Distribution of {title}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/ensemble_partisan_histograms.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_split_histogram_plots(summary_df):
    metrics = {
        'split_munis_count': 'Municipality Splits',
        'split_munis_extra_parts': 'Municipality Extra Parts',
        'split_counties_count': 'County Splits',
        'split_counties_extra_parts': 'County Extra Parts'
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, (col, title) in enumerate(metrics.items()):
        if col in summary_df.columns:
            ax = axes[i]
            data = summary_df[col].dropna()
            if len(data) > 0:
                min_val = int(data.min())
                max_val = int(data.max())
                bin_min = max(0, min_val - 1)
                bin_max = max_val + 1
                bins = [i - 0.5 for i in range(bin_min, bin_max + 1)]
                ax.hist(data, bins=bins, alpha=0.7, color='#6B7280', edgecolor='white', linewidth=0.8)
                ax.set_title(f'Distribution of {title}', fontsize=12, fontweight='bold')
                ax.set_xlabel(title)
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                ax.set_xticks(range(bin_min, bin_max + 1))
                mean_val = data.mean()
                median_val = data.median()
                ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.1f}')
                ax.axvline(median_val, color='orange', linestyle='--', alpha=0.8, label=f'Median: {median_val:.1f}')
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, f'No data for {title}', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Distribution of {title}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/ensemble_split_histograms.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_shares_and_seats_plots(summary_df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    share_cols = [col for col in summary_df.columns if col.startswith('Republican_agg_share_d')]
    if share_cols:
        share_data = []
        for col in share_cols:
            district_num = col.split('_')[-1]
            shares = summary_df[col].dropna()
            for share in shares:
                share_data.append({'District': f'District {district_num[1:]}', 'Republican Share': share})
        if share_data:
            import seaborn as sns
            import pandas as pd
            share_df = pd.DataFrame(share_data)
            sns.violinplot(data=share_df, x='District', y='Republican Share', ax=ax1, hue='District', palette='vlag', legend=False)
            cmap = sns.color_palette("vlag", as_cmap=True)
            min_share = share_df['Republican Share'].min()
            max_share = share_df['Republican Share'].max()
            colors = []
            for district in share_df['District'].unique():
                district_data = share_df[share_df['District'] == district]['Republican Share']
                median_share = district_data.median()
                color = cmap(median_share)
                colors.append(color)
            for i, patch in enumerate(ax1.collections):
                if hasattr(patch, 'set_facecolor'):
                    patch.set_facecolor(colors[i % len(colors)])
            ax1.set_title('Distribution of Republican Vote Shares by District', fontsize=12, fontweight='bold')
            ax1.set_xlabel('District')
            ax1.set_ylabel('Republican Vote Share')
            ax1.axhline(0.5, color='black', linestyle='--', alpha=0.7, label='50% Threshold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
        else:
            ax1.text(0.5, 0.5, 'No share data available', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Republican Vote Shares by District', fontsize=12, fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'No Republican vote share data found', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Republican Vote Shares by District', fontsize=12, fontweight='bold')

    if 'Republican_agg_seats' in summary_df.columns:
        seats_data = summary_df['Republican_agg_seats'].dropna()
        if len(seats_data) > 0:
            total_districts = len([col for col in summary_df.columns if col.startswith('Republican_agg_share_d')])
            if total_districts == 0:
                max_seats = int(seats_data.max()) + 1
                bins = range(0, max_seats + 1)
            else:
                bins = range(0, total_districts + 1)
            bin_edges = [i - 0.5 for i in range(total_districts + 2)]
            ax2.hist(seats_data, bins=bin_edges, alpha=0.7, color='#6B7280', edgecolor='white', linewidth=0.8)
            ax2.set_title('Distribution of Republican Seats', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Number of Republican Seats')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
            mean_val = seats_data.mean()
            median_val = seats_data.median()
            ax2.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.1f}')
            ax2.axvline(median_val, color='orange', linestyle='--', alpha=0.8, label=f'Median: {median_val:.1f}')
            ax2.legend()
            ax2.set_xlim(-0.5, total_districts + 0.5)
            ax2.set_xticks(range(total_districts + 1))
        else:
            ax2.text(0.5, 0.5, 'No seat data available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Distribution of Republican Seats', fontsize=12, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No Republican seat data found', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Distribution of Republican Seats', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/ensemble_shares_and_seats.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_plots(summary_df):
    print("Creating ensemble summary plots...")
    plt.style.use('default')
    sns.set_palette("Blues")
    create_partisan_histogram_plots(summary_df)
    create_split_histogram_plots(summary_df)
    create_shares_and_seats_plots(summary_df)


