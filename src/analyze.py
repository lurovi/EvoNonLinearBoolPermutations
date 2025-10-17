import sys
import json
import os
import random
from pathlib import Path
from typing import List, Tuple
from matplotlib import pyplot as plt
import matplotlib.colorbar as colorbar
import matplotlib.colors as mcolors
import seaborn as sns
import fastplot

import numpy as np
import pandas as pd

from stat_test import is_mannwhitneyu_passed


# =====================================
# Helper functions
# =====================================


def results_folder_name(pop_size: int, gen: int, folder_name: str) -> str:
    return f"{folder_name}/pop{pop_size}_gen{gen}/"


def truth_tables_history(results_folder: str, n_bits: int, seed_index: int, pressure: int, torus_dim: int, radius: int, cmp_rate: float) -> pd.DataFrame:
    s = f'ea_n_bits_{n_bits}_seed_{seed_index}_pressure_{pressure}_torus_{torus_dim}_radius_{radius}_cmp_{str(cmp_rate).replace(".", "d")}.csv'
    s = os.path.join(results_folder, s)
    return pd.read_csv(s, delimiter=',', decimal='.')


def programs_history(results_folder: str, n_bits: int, seed_index: int, pressure: int, torus_dim: int, radius: int, cmp_rate: float, min_length: int, max_length: int, pipeline_iter_step: int, init_bin_size: int) -> pd.DataFrame:
    s = f'ea_programs_n_bits_{n_bits}_seed_{seed_index}_pressure_{pressure}_torus_{torus_dim}_radius_{radius}_cmp_{str(cmp_rate).replace(".", "d")}_len_{min_length}_{max_length}_pipeiter_{pipeline_iter_step}_initbin_{init_bin_size}.csv'
    s = os.path.join(results_folder, s)
    return pd.read_csv(s, delimiter=',', decimal='.')


def global_program(results_folder: str, n_bits: int, seed_index: int, pressure: int, torus_dim: int, radius: int, cmp_rate: float, min_length: int, max_length: int, pipeline_iter_step: int, init_bin_size: int) -> list[tuple[str, list]]:
    # BE CAREFUL WITH THIS FUNCTION, IT USES EVAL
    s = f'best_ea_programs_n_bits_{n_bits}_seed_{seed_index}_pressure_{pressure}_torus_{torus_dim}_radius_{radius}_cmp_{str(cmp_rate).replace(".", "d")}_len_{min_length}_{max_length}_pipeiter_{pipeline_iter_step}_initbin_{init_bin_size}.txt'
    s = os.path.join(results_folder, s)
    with open(s, 'r') as f:
        pr = f.read()
    program = eval(pr)
    return program


# =====================================
# Aggregating results files together
# =====================================

def persist_dict_with_aggregated_metric_for_truth_tables_for_all_generations(
    results_folder: str,
    n_bits: list[int],
    seed_indexes: list[int],
    pressure: int,
    torus_dim: int,
    radius: list[int],
    cmp_rate: list[float],
    persist: bool
) -> dict:
    # KEYS:
    # n_bits (5, 6, 7, 8, 9, etc.),
    # metric (best_fitness, granular_best_fitness, best_resiliency, best_algebraic_degree, best_max_autocorrelation_coefficient, real_global_moran_I),
    # method (baseline, torus2_radius1_cmp0.25, torus2_radius2_cmp0.5, etc.),
    # aggregation (median, q1, q3)
    # VALUE:
    # list of float as long as the number of generations (including initialization), each float is the "aggregation" over the same generation across the other seeds 
    data = {}
    
    loaded_history = {}
    for nb in n_bits:
        # baseline
        for seed in seed_indexes:
            df = truth_tables_history(results_folder, nb, seed, pressure, 0, 0, 0.0)
            df['granular_best_fitness'] = df['best_fitness'].copy()
            df['best_fitness'] = df['best_fitness'].apply(lambda x: int(x))
            loaded_history[f'{nb}_{seed}_{pressure}_{0}_{0}_{0.0}'] = df
        # torus methods
        for r in radius:
            for c in cmp_rate:
                for seed in seed_indexes:
                    df = truth_tables_history(results_folder, nb, seed, 0, torus_dim, r, c)
                    df['granular_best_fitness'] = df['best_fitness'].copy()
                    df['best_fitness'] = df['best_fitness'].apply(lambda x: int(x))
                    loaded_history[f'{nb}_{seed}_{0}_{torus_dim}_{r}_{c}'] = df
    
    
    for nb in n_bits:
        data[str(nb)] = {}
        for metric in ['best_fitness', 'granular_best_fitness', 'best_resiliency', 'best_algebraic_degree', 'best_max_autocorrelation_coefficient', 'real_global_moran_I']:
            data[str(nb)][metric] = {}
            # baseline
            method = 'baseline'
            all_seeds_values = []
            for seed in seed_indexes:
                df = loaded_history[f'{nb}_{seed}_{pressure}_{0}_{0}_{0.0}']
                all_seeds_values.append(df[metric].to_list())
            all_seeds_values = np.array(all_seeds_values)
            median_values = np.median(all_seeds_values, axis=0).tolist()
            q1_values = np.percentile(all_seeds_values, 25, axis=0).tolist()
            q3_values = np.percentile(all_seeds_values, 75, axis=0).tolist()
            data[str(nb)][metric][method] = {
                'median': median_values,
                'q1': q1_values,
                'q3': q3_values
            }
            # torus methods
            for r in radius:
                for c in cmp_rate:
                    method = f'torus{torus_dim}_radius{r}_cmp{str(c)}'
                    all_seeds_values = []
                    for seed in seed_indexes:
                        df = loaded_history[f'{nb}_{seed}_{0}_{torus_dim}_{r}_{c}']
                        all_seeds_values.append(df[metric].to_list())
                    all_seeds_values = np.array(all_seeds_values)
                    median_values = np.median(all_seeds_values, axis=0).tolist()
                    q1_values = np.percentile(all_seeds_values, 25, axis=0).tolist()
                    q3_values = np.percentile(all_seeds_values, 75, axis=0).tolist()
                    data[str(nb)][metric][method] = {
                        'median': median_values,
                        'q1': q1_values,
                        'q3': q3_values
                    }

    if persist:
        with open(os.path.join('../analysis/', 'aggregated_metrics_over_generations_truth_tables.json'), 'w') as f:
            json.dump(data, f, indent=4)
    return data


def persist_dict_with_distribution_metric_for_truth_tables_for_all_repetitions_fixed_generation(
    results_folder: str,
    n_bits: list[int],
    gens: list[int],
    seed_indexes: list[int],
    pressure: int,
    torus_dim: int,
    radius: list[int],
    cmp_rate: list[float],
    persist: bool
) -> dict:
    # KEYS:
    # n_bits (5, 6, 7, 8, 9, etc.),
    # metric (best_fitness, granular_best_fitness, best_resiliency, best_algebraic_degree, best_max_autocorrelation_coefficient, real_global_moran_I),
    # method (baseline, torus2_radius1_cmp0.25, torus2_radius2_cmp0.5, etc.),
    # generation (100 - 1, 500 - 1, 1000 - 1),
    # VALUE:
    # list of float as long as the number of repetitions, each float is taken by the cell in the given repetition corresponding to <metric, gen> 
    data = {}

    loaded_history = {}
    for nb in n_bits:
        # baseline
        for seed in seed_indexes:
            df = truth_tables_history(results_folder, nb, seed, pressure, 0, 0, 0.0)
            df['granular_best_fitness'] = df['best_fitness'].copy()
            df['best_fitness'] = df['best_fitness'].apply(lambda x: int(x))
            loaded_history[f'{nb}_{seed}_{pressure}_{0}_{0}_{0.0}'] = df
        # torus methods
        for r in radius:
            for c in cmp_rate:
                for seed in seed_indexes:
                    df = truth_tables_history(results_folder, nb, seed, 0, torus_dim, r, c)
                    df['granular_best_fitness'] = df['best_fitness'].copy()
                    df['best_fitness'] = df['best_fitness'].apply(lambda x: int(x))
                    loaded_history[f'{nb}_{seed}_{0}_{torus_dim}_{r}_{c}'] = df

    for nb in n_bits:
        data[str(nb)] = {}
        for metric in ['best_fitness', 'granular_best_fitness', 'best_resiliency', 'best_algebraic_degree', 'best_max_autocorrelation_coefficient', 'real_global_moran_I']:
            data[str(nb)][metric] = {}
            # baseline
            method = 'baseline'
            data[str(nb)][metric][method] = {}
            for g in gens:
                all_seeds_values = []
                for seed in seed_indexes:
                    df = loaded_history[f'{nb}_{seed}_{pressure}_{0}_{0}_{0.0}']
                    all_seeds_values.append(df[metric].to_list()[g])
                data[str(nb)][metric][method][str(g)] = all_seeds_values
            # torus methods
            for r in radius:
                for c in cmp_rate:
                    method = f'torus{torus_dim}_radius{r}_cmp{str(c)}'
                    data[str(nb)][metric][method] = {}
                    for g in gens:
                        all_seeds_values = []
                        for seed in seed_indexes:
                            df = loaded_history[f'{nb}_{seed}_{0}_{torus_dim}_{r}_{c}']
                            all_seeds_values.append(df[metric].to_list()[g])
                        data[str(nb)][metric][method][str(g)] = all_seeds_values
    if persist:
        with open(os.path.join('../analysis/', 'distribution_metrics_fixed_generation_truth_tables.json'), 'w') as f:
            json.dump(data, f, indent=4)
    return data




# =====================================
# Plots
# =====================================


def lineplot_grid_over_generations_cellular_truth_tables(
    data: dict,
    metric: str,
    cmp_rate: float,
    palette: dict,
    save_png: bool,
    dpi: int,
    PLOT_ARGS: dict
):
    
    plot = fastplot.plot(None, None, mode='callback',
                         callback=lambda plt: my_callback_lineplot_grid_over_generations_cellular_truth_tables(plt, data, metric, cmp_rate, palette),
                         style='latex', **PLOT_ARGS)
    if save_png:
        plot.savefig(f'../analysis/img/cellular_lineplot_{metric}_cmp{str(cmp_rate).replace(".", "d")}.png', dpi=dpi)
    plot.savefig(f'../analysis/img/cellular_lineplot_{metric}_cmp{str(cmp_rate).replace(".", "d")}.pdf', dpi=dpi)


def my_callback_lineplot_grid_over_generations_cellular_truth_tables(plt, data: dict, metric: str, cmp_rate: float, palette: dict):
    metric_alias = {'granular_best_fitness': r'$\hat{\overline{\ell}}$', 'best_fitness': r'$\overline{\ell}$', 'best_resiliency': r'$r$', 'best_algebraic_degree': r'$d$', 'best_max_autocorrelation_coefficient': r'$a$', 'real_global_moran_I': r'$I$'}
    
    n, m = 3, 4
    radius = ["1", "2", "3"]
    fig, ax = plt.subplots(n, m, figsize=(15, 8), layout='constrained', squeeze=False)
    num_gen = 1000
    x = list(range(num_gen))
    
    n_bits = np.array(list(range(5, 16 + 1))).reshape(n, m)
    for i in range(n):
        for j in range(m):
            nb = str(n_bits[i, j])
            curr_data = data[nb][metric]
            ax[i, j].set_title(f'$n = {nb}$', fontsize=16)
            # baseline + torus methods
            # keep track of min/max values (use q1/q3 if available) so we can add a small padding
            local_min = np.inf
            local_max = -np.inf

            method = 'baseline'
            y = np.array(curr_data[method]['median'])
            q1 = np.array(curr_data[method]['q1'])
            q3 = np.array(curr_data[method]['q3'])
            ax[i, j].plot(x, y, label=method, color=palette["0"], linestyle='-', linewidth=1.2, markersize=10)
            ax[i, j].fill_between(x, q1, q3, color=palette["0"], alpha=0.2)
            local_min = min(local_min, float(np.nanmin(q1)), float(np.nanmin(y)))
            local_max = max(local_max, float(np.nanmax(q3)), float(np.nanmax(y)))

            for r in radius:
                method = f'torus2_radius{r}_cmp{str(cmp_rate)}'
                y = np.array(curr_data[method]['median'])
                q1 = np.array(curr_data[method]['q1'])
                q3 = np.array(curr_data[method]['q3'])
                ax[i, j].plot(x, y, label=method, color=palette[r], linestyle='-', linewidth=1.2, markersize=10)
                ax[i, j].fill_between(x, q1, q3, color=palette[r], alpha=0.2)
                local_min = min(local_min, float(np.nanmin(q1)), float(np.nanmin(y)))
                local_max = max(local_max, float(np.nanmax(q3)), float(np.nanmax(y)))

            # set x limits/ticks
            ax[i, j].set_xlim(0, num_gen)
            ax[i, j].set_xticks([0, num_gen // 2, num_gen])

            # apply a small padding on the y axis so lines aren't clipped at the top/bottom border
            if metric == "real_global_moran_I":
                ax[i, j].set_ylim(-0.1, 0.5)
                ax[i, j].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
            else:
                if np.isfinite(local_min) and np.isfinite(local_max):
                    rng = local_max - local_min
                    if rng <= 0:
                        # flat line: provide a sensible padding
                        pad = 1.0
                    else:
                        # for integer/discrete metrics provide at least 1 unit of headroom
                        pad = max(1.0, 0.02 * rng)
                    ax[i, j].set_ylim(local_min - pad, local_max + pad)
            
            ax[i, j].tick_params(axis='both', which='both', reset=False, bottom=False, top=False, left=False, right=False)
            
            if i == n - 1:
                ax[i, j].set_xlabel('Generation', fontsize=18)
            else:
                ax[i, j].tick_params(labelbottom=False)
                ax[i, j].set_xticklabels([])
            
            if j == 0:
                ax[i, j].set_ylabel(metric_alias[metric], labelpad=10 if i != 0 else (22 if metric in ('best_fitness', 'granular_best_fitness') else 10), fontsize=18)
            #else:
                #ax[i, j].tick_params(labelleft=False)
                #ax[i, j].set_yticklabels([])            
            
            if i == n - 1 and j == m - 1:
                ax[i, j].tick_params(pad=7)

            ax[i, j].grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)

 
def boxplot_grid_cellular_truth_tables(
    data: dict,
    metric: str,
    gen: int,
    palette_cmp: dict,
    save_png: bool,
    dpi: int,
    PLOT_ARGS: dict
):
    metric_alias = {'granular_best_fitness': r'$\hat{\overline{\ell}}$', 'best_fitness': r'$\overline{\ell}$', 'best_resiliency': r'$r$', 'best_algebraic_degree': r'$d$', 'best_max_autocorrelation_coefficient': r'$a$', 'real_global_moran_I': r'$I$'}
    
    n_bits = list(range(5, 16 + 1))
    dataframes_dict = {}
    significance_dict = {}
    for nb in n_bits:
        signif = {}
        nb_str = str(nb)
        gen_str = str(gen)
        curr_data = data[nb_str][metric]
        curr_dataframe_dict = {"Method": [], r"$p$": [], metric_alias[metric]: []}
        # baseline
        base_values = curr_data['baseline'][gen_str]
        curr_dataframe_dict["Method"].extend([r'\notoroid'] * len(base_values))
        curr_dataframe_dict[r"$p$"].extend([str(0.0)] * len(base_values))
        curr_dataframe_dict[metric_alias[metric]].extend(base_values)
        # torus methods
        for r in [1, 2, 3]:
            for c in [0.25, 0.5, 0.75, 1.0]:
                method = r'\toroid{' + str(2) + '}{' + str(r) + '}'
                if method not in signif:
                    signif[method] = {}
                cellular_values = curr_data[f'torus2_radius{r}_cmp{c}'][gen_str]
                curr_dataframe_dict["Method"].extend([method] * len(cellular_values))
                curr_dataframe_dict[r"$p$"].extend([str(c)] * len(cellular_values))
                curr_dataframe_dict[metric_alias[metric]].extend(cellular_values)
                is_passed, _ = is_mannwhitneyu_passed(base_values, cellular_values, alternative='less', alpha=0.05)
                signif[method][str(c)] = is_passed
        df = pd.DataFrame(curr_dataframe_dict)
        dataframes_dict[nb_str] = df
        significance_dict[nb_str] = signif
    for nb in n_bits:
        print(f"n={nb}")
        print(significance_dict[str(nb)])
        print()
    plot = fastplot.plot(None, None, mode='callback',
                         callback=lambda plt: my_callback_boxplot_grid_cellular_truth_tables(plt, dataframes_dict, significance_dict, metric_alias[metric], palette_cmp),
                         style='latex', **PLOT_ARGS)
    if save_png:
        plot.savefig(f'../analysis/img/cellular_boxplot_{metric}_gen{str(gen)}.png', dpi=dpi)
    plot.savefig(f'../analysis/img/cellular_boxplot_{metric}_gen{str(gen)}.pdf', dpi=dpi)


def my_callback_boxplot_grid_cellular_truth_tables(plt, data: dict[str, pd.DataFrame], significance_dict: dict[str, dict[str, dict[str, bool]]], y_title: str, palette_cmp: dict[str, str]):
    n, m = 4, 3
    fig, ax = plt.subplots(n, m, figsize=(15, 10), layout='constrained', squeeze=False)
    n_bits = np.array(list(range(5, 16 + 1))).reshape(n, m)
    for i in range(n):
        for j in range(m):
            nb = str(n_bits[i, j])
            df = data[nb]
            ax[i, j].set_title(f'$n = {nb}$', fontsize=16)
            # Grid and ticks
            ax[i, j].grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)
            ax[i, j].tick_params(axis='both', which='both', reset=False, bottom=False, top=False, left=False, right=False)
            # determine local data min/max to add padding and avoid clipping of boxes/whiskers
            try:
                col = df[y_title]
                # drop NaNs
                col = col.dropna()
                if len(col) > 0:
                    local_min = float(col.min())
                    local_max = float(col.max())
                else:
                    local_min = np.nan
                    local_max = np.nan
            except Exception:
                local_min = np.nan
                local_max = np.nan

            sns.boxplot(data=df, x="Method", y=y_title, hue=r"$p$", palette=palette_cmp, ax=ax[i, j], showfliers=False,
                        legend=False, fliersize=0.0, log_scale=None)

            # if we computed sensible local bounds, add a small padding to prevent clipping
            if np.isfinite(local_min) and np.isfinite(local_max):
                rng = local_max - local_min
                if rng <= 0:
                    pad = 1.0
                else:
                    # for metrics that are integer-like ensure at least 0.5-1 unit padding
                    pad = max(0.5, 0.02 * rng)
                ax[i, j].set_ylim(local_min - pad, local_max + pad)

            # --- significance annotations: draw single '*' per True in significance_dict (below the box)
            # significance_dict is structured as significance_dict[nb][method][p_str] = bool
            # compute method/hue positions consistent with seaborn's layout
            try:
                signif = significance_dict.get(nb, {}) if significance_dict else {}
                methods = list(dict.fromkeys(df['Method'].tolist()))
                hues = list(dict.fromkeys(df[r"$p$"].tolist()))
                n_methods = len(methods)
                n_hues = len(hues) if len(hues) > 0 else 1
                xticks = np.arange(n_methods)
                box_total_width = 0.8
                single_width = box_total_width / n_hues

                for method_label, hue_dict in signif.items():
                    for hue_str, passed in hue_dict.items():
                        if not passed:
                            continue
                        # find method index and hue index
                        try:
                            mi = methods.index(method_label)
                        except ValueError:
                            continue
                        try:
                            hi = hues.index(str(hue_str))
                        except ValueError:
                            # try numeric matching if one is numeric type
                            try:
                                hi = hues.index(hue_str)
                            except Exception:
                                continue

                        # x coordinate: group center + offset for hue (data coords)
                        offset = (hi - (n_hues - 1) / 2.0) * single_width
                        x_data = float(xticks[mi] + offset)

                        # attempt to place the star at a fixed vertical axes fraction so all stars
                        # appear at the same height across subplots
                        try:
                            # transform data x to display coords, then to axes coords
                            x_disp, _ = ax[i, j].transData.transform((x_data, 0))
                            x_axes = ax[i, j].transAxes.inverted().transform((x_disp, 0))[0]
                            y_axes_fixed = 0.03
                            ax[i, j].text(x_axes, y_axes_fixed, r'\textbf{*}', transform=ax[i, j].transAxes,
                                          ha='center', va='center', fontsize=20, color='black', clip_on=False)
                        except Exception:
                            # fallback: place using data coords near bottom of axis
                            vals = df[(df['Method'] == method_label) & (df[r"$p$"] == str(hue_str))][y_title].dropna()
                            if len(vals) == 0:
                                continue
                            y0, y1 = ax[i, j].get_ylim()
                            yrange = max((y1 - y0), 1e-6)
                            stagger = (hi - (n_hues - 1) / 2.0) * 0.02 * yrange
                            y_coord = y0 + 0.02 * yrange + stagger
                            ax[i, j].text(x_data, y_coord, r'\textbf{*}', ha='center', va='bottom', fontsize=20, color='black', clip_on=False)
            except Exception:
                # keep plotting even if annotations fail
                pass
            
            
            
            if i == n - 1:
                ax[i, j].set_xlabel('Method', fontsize=18)
            else:
                ax[i, j].tick_params(labelbottom=False)
                ax[i, j].set_xticklabels([])
                ax[i, j].set_xlabel('')
            
            if j == 0:
                ax[i, j].set_ylabel(y_title, labelpad=10 if i == n - 1 else (15 if i == 0 else 17), fontsize=18)
            else:
                # empty y labels for
                ax[i, j].set_ylabel('')
                #ax[i, j].tick_params(labelleft=False)
                #ax[i, j].set_yticklabels([])            
            
            if i == n - 1 and j == m - 1:
                ax[i, j].tick_params(pad=7)

            ax[i, j].grid(True, axis='y', which='major', color='gray', linestyle='--', linewidth=0.5)
    

def create_legend(label_color_dict, PLOT_ARGS, title=None):
    fastplot.plot(None, f'legend.pdf', mode='callback',
                  callback=lambda plt: create_legend_callback(plt, label_color_dict, title),
                  style='latex', **PLOT_ARGS)


def create_legend_callback(plt, label_color_dict, title=None):
    """
    Create a legend from a dictionary of labels and colors.

    Parameters:
    - label_color_dict: dict
        Dictionary where keys are labels and values are colors.
    - ax: matplotlib.axes.Axes, optional
        The axis to which the legend will be added. If None, uses the current axis.
    - loc: str, optional
        Location of the legend (default: 'upper right').
    - title: str, optional
        Title of the legend.
    """
    plt.tight_layout(pad=0)
    fig, ax = plt.subplots(figsize=(6, 1))  # Small figure with only the legend
    fig.set_size_inches(5.0, 0.3)
    ax.axis("off")  # Hide axes

    # Create legend handles
    handles = [plt.Line2D([0], [0], color=color, lw=4, label=label) for label, color in label_color_dict.items()]

    # Create and center the legend
    ax.legend(
        handles=handles,
        loc="center",
        title=title,
        ncol=len(label_color_dict),  # Arrange items in one row
        frameon=False  # Remove box around legend
    )

# =====================================
# Tables
# =====================================





# =====================================
# Main
# =====================================

def main():
    preamble = r'''
                \usepackage{amsmath}
                \usepackage{libertine}
                \usepackage{xspace}

                \newcommand{\moranI}{$I$\xspace}
                \newcommand{\toroid}[2]{$\mathcal{T}^{#1}_{#2}$\xspace}
                \newcommand{\notoroid}{$\mathcal{T}^{0}$\xspace}

                '''

    PLOT_ARGS = {'rcParams': {'text.latex.preamble': preamble, 'pdf.fonttype': 42, 'ps.fonttype': 42}}
    palette = {'0': "#B60000", '1': "#06BC4F", '2': "#b10984", '3': "#5620bc"}
    palette_toroid = {r'\notoroid': "#B60000", r'\toroid{2}{1}': "#06BC4F", r'\toroid{2}{2}': "#b10984", r'\toroid{2}{3}': "#5620bc"}
    palette_cmp = {'0.0': "#B60000", '0.25': "#BCB8F7", '0.5': "#594fe7", '0.75': "#2d1bcc", '1.0': "#020270"}
    palette_p = {r'$p = 0.25$': "#BCB8F7", r'$p = 0.5$': "#594fe7", r'$p = 0.75$': "#2d1bcc", r'$p = 1.0$': "#020270"}
    
    
    n_bits = list(range(5, 16 + 1))
    seed_indexes = list(range(1, 30 + 1))
    pressure = 4
    pop_size = 100
    n_iter = 1000
    torus_dim = 2
    radius = [1, 2, 3]
    cmp_rate = [0.25, 0.5, 0.75, 1.0]
    gens = [200 - 1, 400 - 1, 500 - 1, 1000 - 1]
    results_folder = results_folder_name(pop_size, n_iter, '../results')
    persist = True
    
    # _ = persist_dict_with_aggregated_metric_for_truth_tables_for_all_generations(
    #      results_folder,
    #      n_bits,
    #      seed_indexes,
    #      pressure,
    #      torus_dim,
    #      radius,
    #      cmp_rate,
    #      persist
    # )
    # _ = persist_dict_with_distribution_metric_for_truth_tables_for_all_repetitions_fixed_generation(
    #      results_folder,
    #      n_bits,
    #      gens,
    #      seed_indexes,
    #      pressure,
    #      torus_dim,
    #      radius,
    #      cmp_rate,
    #      persist
    # )
    create_legend(palette_p, PLOT_ARGS)
    quit()
    with open('../analysis/aggregated_metrics_over_generations_truth_tables_pop100_gen1000.json', 'r') as f:
        data = json.load(f)
        
    with open('../analysis/distribution_metrics_fixed_generation_truth_tables_pop100_gen1000.json', 'r') as f:
        data_box = json.load(f)

    for metric in ['best_fitness', 'real_global_moran_I']:
        for cr in [0.25, 0.5, 0.75, 1.0]:
            lineplot_grid_over_generations_cellular_truth_tables(
                data,
                metric=metric,
                cmp_rate=cr,
                palette=palette,
                save_png=False,
                dpi=800,
                PLOT_ARGS=PLOT_ARGS
            )
    quit()
    boxplot_grid_cellular_truth_tables(
        data=data_box,
        metric='best_fitness',
        gen=999,
        palette_cmp=palette_cmp,
        save_png=False,
        dpi=800,
        PLOT_ARGS=PLOT_ARGS
    )


if __name__ == "__main__":
    main()
