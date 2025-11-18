import statistics
import sys
import json
import os
import random
from pathlib import Path
from typing import List, Tuple
import matplotlib.colorbar as colorbar
import matplotlib.colors as mcolors
import seaborn as sns

import numpy as np
import pandas as pd

from stat_test import is_mannwhitneyu_passed

from matplotlib import pyplot as plt
preamble = r'''
            \usepackage{amsmath}
            \usepackage{libertine}
            \usepackage{xspace}

            \newcommand{\moranI}{$I$\xspace}
            \newcommand{\toroid}[2]{$\mathcal{T}^{#1}_{#2}$\xspace}
            \newcommand{\notoroid}{$\mathcal{T}^{0}$\xspace}

            '''
plt.rcParams.update({
    "text.usetex": True, "font.family": "serif", "font.serif": "Computer Modern Roman",
    "text.latex.preamble": preamble,  'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False,
})


# =====================================
# Helper functions
# =====================================

def truth_tables_history(results_folder: str, pop_size: int, gen: int, dupl_retry: int, n_bits: int, seed_index: int, pressure: int, torus_dim: int, radius: int, cmp_rate: float) -> pd.DataFrame:
    s = f'ea_n_bits_{n_bits}_seed_{seed_index}_pressure_{pressure}_torus_{torus_dim}_radius_{radius}_cmp_{str(cmp_rate).replace(".", "d")}.csv'
    s = os.path.join(results_folder, f"pop{pop_size}_gen{gen}_duplretry{dupl_retry}", s)
    return pd.read_csv(s, delimiter=',', decimal='.')


def programs_history(results_folder: str, pop_size: int, gen: int, dupl_retry: int, n_bits: int, seed_index: int, pressure: int, torus_dim: int, radius: int, cmp_rate: float, min_length: int, max_length: int, pipeline_iter_step: int, init_bin_size: int) -> pd.DataFrame:
    s = f'ea_programs_n_bits_{n_bits}_seed_{seed_index}_pressure_{pressure}_torus_{torus_dim}_radius_{radius}_cmp_{str(cmp_rate).replace(".", "d")}_len_{min_length}_{max_length}_pipeiter_{pipeline_iter_step}_initbin_{init_bin_size}.csv'
    s = os.path.join(results_folder, f"pop{pop_size}_gen{gen}_duplretry{dupl_retry}", s)
    return pd.read_csv(s, delimiter=',', decimal='.')


def global_program(results_folder: str, pop_size: int, gen: int, dupl_retry: int, n_bits: int, seed_index: int, pressure: int, torus_dim: int, radius: int, cmp_rate: float, min_length: int, max_length: int, pipeline_iter_step: int, init_bin_size: int) -> list[tuple[str, list]]:
    # BE CAREFUL WITH THIS FUNCTION, IT USES EVAL
    s = f'best_ea_programs_n_bits_{n_bits}_seed_{seed_index}_pressure_{pressure}_torus_{torus_dim}_radius_{radius}_cmp_{str(cmp_rate).replace(".", "d")}_len_{min_length}_{max_length}_pipeiter_{pipeline_iter_step}_initbin_{init_bin_size}.txt'
    s = os.path.join(results_folder, f"pop{pop_size}_gen{gen}_duplretry{dupl_retry}", s)
    with open(s, 'r') as f:
        pr = f.read()
    program = eval(pr)
    return program


# =====================================
# Aggregating results files together
# =====================================

def load_history(
    results_folder: str,
    n_bits: list[int],
    seed_indexes: list[int],
    pop_size: int,
    gen: int,
    dupl_retry: int,
    pressure: int,
    torus_dim: int,
    radius: list[int],
    cmp_rate: list[float]
) -> dict:
    loaded_history = {}
    for nb in n_bits:
        # baseline
        for seed in seed_indexes:
            print(f'Loading history for n_bits={nb}, seed={seed}, baseline')
            df = truth_tables_history(
                results_folder=results_folder,
                pop_size=pop_size,
                gen=gen,
                dupl_retry=0,
                n_bits=nb,
                seed_index=seed,
                pressure=pressure,
                torus_dim=0,
                radius=0,
                cmp_rate=0.0
            )
            df['granular_best_fitness'] = df['best_fitness'].copy()
            df['best_fitness'] = df['best_fitness'].apply(lambda x: int(x))
            loaded_history[f'{pop_size}_{gen}_{0}_{nb}_{seed}_{pressure}_{0}_{0}_{0.0}'] = df
        # baseline10retry
        for seed in seed_indexes:
            print(f'Loading history for n_bits={nb}, seed={seed}, baseline10retry')
            df = truth_tables_history(
                results_folder=results_folder,
                pop_size=pop_size,
                gen=gen,
                dupl_retry=dupl_retry,
                n_bits=nb,
                seed_index=seed,
                pressure=pressure,
                torus_dim=0,
                radius=0,
                cmp_rate=0.0
            )
            df['granular_best_fitness'] = df['best_fitness'].copy()
            df['best_fitness'] = df['best_fitness'].apply(lambda x: int(x))
            loaded_history[f'{pop_size}_{gen}_{dupl_retry}_{nb}_{seed}_{pressure}_{0}_{0}_{0.0}'] = df
        # torus methods
        for r in radius:
            for c in cmp_rate:
                for seed in seed_indexes:
                    print(f'Loading history for n_bits={nb}, seed={seed}, torus{torus_dim}_radius{r}_cmp{c}')
                    df = truth_tables_history(
                        results_folder=results_folder,
                        pop_size=pop_size,
                        gen=gen,
                        dupl_retry=0,
                        n_bits=nb,
                        seed_index=seed,
                        pressure=0,
                        torus_dim=torus_dim,
                        radius=r,
                        cmp_rate=c
                    )
                    df['granular_best_fitness'] = df['best_fitness'].copy()
                    df['best_fitness'] = df['best_fitness'].apply(lambda x: int(x))
                    loaded_history[f'{pop_size}_{gen}_{0}_{nb}_{seed}_{0}_{torus_dim}_{r}_{c}'] = df
    return loaded_history


def load_history_programs(
    results_folder: str,
    n_bits: list[int],
    seed_indexes: list[int],
    pop_size: int,
    gen: int,
    pressure: int,
    init_bin_size: int,
    lengths: list[tuple[int, int]],
    pipeline_iter_steps: list[int],
) -> dict:
    loaded_history = {}
    for nb in n_bits:
        # baseline
        for seed in seed_indexes:
            print(f'Loading history for n_bits={nb}, seed={seed}, baseline')
            df = truth_tables_history(
                results_folder=results_folder,
                pop_size=pop_size,
                gen=gen,
                dupl_retry=0,
                n_bits=nb,
                seed_index=seed,
                pressure=pressure,
                torus_dim=0,
                radius=0,
                cmp_rate=0.0
            )
            df['granular_best_fitness'] = df['best_fitness'].copy()
            df['best_fitness'] = df['best_fitness'].apply(lambda x: int(x))
            loaded_history[f'{pop_size}_{gen}_{0}_{nb}_{seed}_{pressure}_{0}_{0}_{0}_{0}'] = df
        # programs
        for min_, max_ in lengths:
            for p in pipeline_iter_steps:
                for seed in seed_indexes:
                    print(f'Loading history for n_bits={nb}, seed={seed}, init_bin_size={init_bin_size}, len=({min_},{max_}), pipe_iter_step={p}')
                    df = programs_history(
                        results_folder=results_folder,
                        pop_size=pop_size,
                        gen=gen,
                        dupl_retry=0,
                        n_bits=nb,
                        seed_index=seed,
                        pressure=pressure,
                        torus_dim=0,
                        radius=0,
                        cmp_rate=0.0,
                        min_length=min_,
                        max_length=max_,
                        pipeline_iter_step=p,
                        init_bin_size=init_bin_size
                    )
                    df['granular_best_fitness'] = df['best_fitness'].copy()
                    df['best_fitness'] = df['best_fitness'].apply(lambda x: int(x))
                    loaded_history[f'{pop_size}_{gen}_{0}_{nb}_{seed}_{pressure}_{init_bin_size}_{min_}_{max_}_{p}'] = df
    return loaded_history

## TRUTH TABLES AGGRAGEGATED METRICS

def persist_dict_with_aggregated_metric_for_truth_tables_for_all_generations(
    results_folder: str,
    pop_size: int,
    gen: int,
    dupl_retry: int,
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
    # method (baseline, baseline10retry, torus2_radius1_cmp0.25, torus2_radius2_cmp0.5, etc.),
    # aggregation (median, q1, q3)
    # VALUE:
    # list of float as long as the number of generations (including initialization), each float is the "aggregation" over the same generation across the other seeds 
    data = {}
    
    loaded_history = load_history(
        results_folder=results_folder,
        pop_size=pop_size,
        gen=gen,
        dupl_retry=dupl_retry,
        n_bits=n_bits,
        seed_indexes=seed_indexes,
        pressure=pressure,
        torus_dim=torus_dim,
        radius=radius,
        cmp_rate=cmp_rate
    )

    for nb in n_bits:
        data[str(nb)] = {}
        for metric in ['best_fitness', 'granular_best_fitness', 'best_resiliency', 'best_algebraic_degree', 'best_max_autocorrelation_coefficient', 'real_global_moran_I', 'pop_med_fitness', 'pop_q1_fitness', 'pop_q3_fitness']:
            data[str(nb)][metric] = {}
            # baseline
            method = 'baseline'
            all_seeds_values = []
            for seed in seed_indexes:
                df = loaded_history[f'{pop_size}_{gen}_{0}_{nb}_{seed}_{pressure}_{0}_{0}_{0.0}']
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
            # baseline10retry
            method = f'baseline{dupl_retry}retry'
            all_seeds_values = []
            for seed in seed_indexes:
                df = loaded_history[f'{pop_size}_{gen}_{dupl_retry}_{nb}_{seed}_{pressure}_{0}_{0}_{0.0}']
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
                        df = loaded_history[f'{pop_size}_{gen}_{0}_{nb}_{seed}_{0}_{torus_dim}_{r}_{c}']
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
    pop_size: int,
    gen: int,
    dupl_retry: int,
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
    # method (baseline, baseline10retry, torus2_radius1_cmp0.25, torus2_radius2_cmp0.5, etc.),
    # generation (100 - 1, 500 - 1, 1000 - 1),
    # VALUE:
    # list of float as long as the number of repetitions, each float is taken by the cell in the given repetition corresponding to <metric, gen> 
    data = {}

    loaded_history = load_history(
        results_folder=results_folder,
        pop_size=pop_size,
        gen=gen,
        dupl_retry=dupl_retry,
        n_bits=n_bits,
        seed_indexes=seed_indexes,
        pressure=pressure,
        torus_dim=torus_dim,
        radius=radius,
        cmp_rate=cmp_rate
    )

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
                    df = loaded_history[f'{pop_size}_{gen}_{0}_{nb}_{seed}_{pressure}_{0}_{0}_{0.0}']
                    all_seeds_values.append(df[metric].to_list()[g])
                data[str(nb)][metric][method][str(g)] = all_seeds_values
            # baseline10retry
            method = f'baseline{dupl_retry}retry'
            data[str(nb)][metric][method] = {}
            for g in gens:
                all_seeds_values = []
                for seed in seed_indexes:
                    df = loaded_history[f'{pop_size}_{gen}_{dupl_retry}_{nb}_{seed}_{pressure}_{0}_{0}_{0.0}']
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
                            df = loaded_history[f'{pop_size}_{gen}_{0}_{nb}_{seed}_{0}_{torus_dim}_{r}_{c}']
                            all_seeds_values.append(df[metric].to_list()[g])
                        data[str(nb)][metric][method][str(g)] = all_seeds_values
    if persist:
        with open(os.path.join('../analysis/', 'distribution_metrics_fixed_generation_truth_tables.json'), 'w') as f:
            json.dump(data, f, indent=4)
    return data


## PROGRAMS AGGRAGEGATED METRICS

def persist_dict_with_aggregated_metric_for_programs_for_all_generations(
    results_folder: str,
    pop_size: int,
    gen: int,
    n_bits: list[int],
    seed_indexes: list[int],
    pressure: int,
    init_bin_size: int,
    lengths: list[tuple[int, int]],
    pipeline_iter_steps: list[int],
    persist: bool
) -> dict:
    # KEYS:
    # n_bits (5, 6, 7, 8, 9, etc.),
    # metric (best_fitness, granular_best_fitness, best_resiliency, best_algebraic_degree, best_max_autocorrelation_coefficient, real_global_moran_I),
    # method (baseline, binsize16_length_2_5_p100, binsize16_length_2_10_p100, binsize16_length_2_10_p200, etc.),
    # aggregation (median, q1, q3)
    # VALUE:
    # list of float as long as the number of generations (including initialization), each float is the "aggregation" over the same generation across the other seeds 
    data = {}
    
    loaded_history = load_history_programs(
        results_folder=results_folder,
        pop_size=pop_size,
        gen=gen,
        n_bits=n_bits,
        seed_indexes=seed_indexes,
        pressure=pressure,
        init_bin_size=init_bin_size,
        lengths=lengths,
        pipeline_iter_steps=pipeline_iter_steps
    )

    for nb in n_bits:
        data[str(nb)] = {}
        for metric in ['best_fitness', 'granular_best_fitness', 'best_resiliency', 'best_algebraic_degree', 'best_max_autocorrelation_coefficient', 'real_global_moran_I', 'pop_med_fitness', 'pop_q1_fitness', 'pop_q3_fitness']:
            data[str(nb)][metric] = {}
            # baseline
            method = 'baseline'
            all_seeds_values = []
            for seed in seed_indexes:
                df = loaded_history[f'{pop_size}_{gen}_{0}_{nb}_{seed}_{pressure}_{0}_{0}_{0}_{0}']
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
            # programs
            for min_, max_ in lengths:
                for p in pipeline_iter_steps:
                    method = f'binsize{init_bin_size}_length_{min_}_{max_}_p{p}'
                    all_seeds_values = []
                    for seed in seed_indexes:
                        df = loaded_history[f'{pop_size}_{gen}_{0}_{nb}_{seed}_{pressure}_{init_bin_size}_{min_}_{max_}_{p}']
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
        with open(os.path.join('../analysis/', 'aggregated_metrics_over_generations_programs.json'), 'w') as f:
            json.dump(data, f, indent=4)
    return data


def persist_dict_with_distribution_metric_for_programs_for_all_repetitions_fixed_generation(
    results_folder: str,
    pop_size: int,
    gen: int,
    n_bits: list[int],
    gens: list[int],
    seed_indexes: list[int],
    pressure: int,
    init_bin_size: int,
    lengths: list[tuple[int, int]],
    pipeline_iter_steps: list[int],
    persist: bool
) -> dict:
    # KEYS:
    # n_bits (5, 6, 7, 8, 9, etc.),
    # metric (best_fitness, granular_best_fitness, best_resiliency, best_algebraic_degree, best_max_autocorrelation_coefficient, real_global_moran_I),
    # method (baseline, binsize16_length_2_5_p100, binsize16_length_2_10_p100, binsize16_length_2_10_p200, etc.),
    # generation (100 - 1, 500 - 1, 1000 - 1),
    # VALUE:
    # list of float as long as the number of repetitions, each float is taken by the cell in the given repetition corresponding to <metric, gen> 
    data = {}

    loaded_history = load_history_programs(
        results_folder=results_folder,
        pop_size=pop_size,
        gen=gen,
        n_bits=n_bits,
        seed_indexes=seed_indexes,
        pressure=pressure,
        init_bin_size=init_bin_size,
        lengths=lengths,
        pipeline_iter_steps=pipeline_iter_steps
    )

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
                    df = loaded_history[f'{pop_size}_{gen}_{0}_{nb}_{seed}_{pressure}_{0}_{0}_{0}_{0}']
                    all_seeds_values.append(df[metric].to_list()[g])
                data[str(nb)][metric][method][str(g)] = all_seeds_values
            # programs
            for min_, max_ in lengths:
                for p in pipeline_iter_steps:
                    method = f'binsize{init_bin_size}_length_{min_}_{max_}_p{p}'
                    data[str(nb)][metric][method] = {}
                    for g in gens:
                        all_seeds_values = []
                        for seed in seed_indexes:
                            df = loaded_history[f'{pop_size}_{gen}_{0}_{nb}_{seed}_{pressure}_{init_bin_size}_{min_}_{max_}_{p}']
                            all_seeds_values.append(df[metric].to_list()[g])
                        data[str(nb)][metric][method][str(g)] = all_seeds_values
    if persist:
        with open(os.path.join('../analysis/', 'distribution_metrics_fixed_generation_programs.json'), 'w') as f:
            json.dump(data, f, indent=4)
    return data


# =====================================
# Plots TRUTH TABLES
# =====================================


def lineplot_grid_over_generations_cellular_truth_tables(
    data: dict,
    metric: str,
    cmp_rate: float,
    palette: dict,
    save_png: bool,
    dpi: int
):
    
    # plot = fastplot.plot(None, None, mode='callback',
    #                      callback=lambda plt: my_callback_lineplot_grid_over_generations_cellular_truth_tables(plt, data, metric, cmp_rate, palette),
    #                      style='latex', **PLOT_ARGS)
    plot = my_callback_lineplot_grid_over_generations_cellular_truth_tables(data, metric, cmp_rate, palette)
    if save_png:
        plot.savefig(f'../analysis/img/cellular_lineplot_{metric}_cmp{str(cmp_rate).replace(".", "d")}.png', dpi=dpi)
    plot.savefig(f'../analysis/img/cellular_lineplot_{metric}_cmp{str(cmp_rate).replace(".", "d")}.pdf', dpi=dpi)
    plt.clf()
    plt.cla()
    plt.close()


def my_callback_lineplot_grid_over_generations_cellular_truth_tables(data: dict, metric: str, cmp_rate: float, palette: dict):
    metric_alias = {'pop_med_fitness': r'Median $\ \ \tilde{\overline{\ell}}$', 'granular_best_fitness': r'$\tilde{\overline{\ell}}$', 'best_fitness': r'$\overline{\ell}$', 'best_resiliency': r'$r$', 'best_algebraic_degree': r'$d$', 'best_max_autocorrelation_coefficient': r'$a$', 'real_global_moran_I': r'$I$'}
    
    n, m = 3, 3
    radius = ["1", "2", "3"]
    fig, ax = plt.subplots(n, m, figsize=(16, 10), layout='constrained', squeeze=False)
    # Add a bit more white space on the left with constrained layout
    # rect = [left, bottom, right, top] in figure coordinates
    fig.get_layout_engine().set(w_pad=4/72, h_pad=4/72, hspace=0.05, wspace=0.05, rect=[0.01, 0, 0.98, 1]) # type: ignore

    num_gen = 1000
    x = list(range(num_gen))

    n_bits = np.array(list(range(8, 16 + 1))).reshape(n, m)
    for i in range(n):
        for j in range(m):
            nb = str(n_bits[i, j])
            curr_data = data[nb][metric]
            ax[i, j].set_title(f'$n = {nb}$', fontsize=24)
            # baseline + torus methods
            # keep track of min/max values (use q1/q3 if available) so we can add a small padding
            local_min = np.inf
            local_max = -np.inf

            method = 'baseline10retry'
            y = np.array(curr_data[method]['median'])
            q1 = np.array(curr_data[method]['q1']) if metric != "pop_med_fitness" else np.array(data[nb]['pop_q1_fitness'][method]['median'])
            q3 = np.array(curr_data[method]['q3']) if metric != "pop_med_fitness" else np.array(data[nb]['pop_q3_fitness'][method]['median'])
            ax[i, j].plot(x, y, label=method, color=palette["0"], linestyle='-', linewidth=2.0, markersize=10)
            ax[i, j].fill_between(x, q1, q3, color=palette["0"], alpha=0.15)
            local_min = min(local_min, float(np.nanmin(q1)), float(np.nanmin(y)))
            local_max = max(local_max, float(np.nanmax(q3)), float(np.nanmax(y)))

            for r in radius:
                method = f'torus2_radius{r}_cmp{str(cmp_rate)}'
                y = np.array(curr_data[method]['median'])
                q1 = np.array(curr_data[method]['q1']) if metric != "pop_med_fitness" else np.array(data[nb]['pop_q1_fitness'][method]['median'])
                q3 = np.array(curr_data[method]['q3']) if metric != "pop_med_fitness" else np.array(data[nb]['pop_q3_fitness'][method]['median'])
                ax[i, j].plot(x, y, label=method, color=palette[r], linestyle='-', linewidth=2.0, markersize=10)
                ax[i, j].fill_between(x, q1, q3, color=palette[r], alpha=0.15)
                local_min = min(local_min, float(np.nanmin(q1)), float(np.nanmin(y)))
                local_max = max(local_max, float(np.nanmax(q3)), float(np.nanmax(y)))

            # set x limits/ticks
            ax[i, j].set_xlim(0, num_gen)
            ax[i, j].set_xticks([0, num_gen // 2, num_gen])

            # apply a small padding on the y axis so lines aren't clipped at the top/bottom border
            if metric == "real_global_moran_I":
                ax[i, j].set_ylim(-0.1, 0.5)
                ax[i, j].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
            elif metric == 'best_fitness':
                if nb == '8':
                    ax[i, j].set_ylim(110, 117)
                    #ax[i, j].set_yticks([250, 260, 270, 280, 290, 300])
                elif nb == '9':
                    ax[i, j].set_ylim(230, 237)
                    #ax[i, j].set_yticks([500, 520, 540, 560, 580, 600])
                elif nb == '10':
                    ax[i, j].set_ylim(472, 481)
                    #ax[i, j].set_yticks([1000, 1040, 1080, 1120, 1160, 1200])
                elif nb == '11':
                    ax[i, j].set_ylim(968, 977)
                    #ax[i, j].set_yticks([2000, 2080, 2160, 2240, 2320, 2400])
                elif nb == '12':
                    ax[i, j].set_ylim(1962, 1973)
                    #ax[i, j].set_yticks([4000, 4160, 4320, 4480, 4640, 4800])
                elif nb == '13':
                    ax[i, j].set_ylim(3966, 3981)
                    #ax[i, j].set_yticks([8000, 8320, 8640, 8960, 9280, 9600])
                elif nb == '14':
                    ax[i, j].set_ylim(7998, 8013)
                    #ax[i, j].set_yticks([16000, 16640, 17280, 17920, 18560, 19200])
                elif nb == '15':
                    ax[i, j].set_ylim(16088, 16115)
                    #ax[i, j].set_yticks([32000, 33280, 34560, 35840, 37120, 38400])
                elif nb == '16':
                    ax[i, j].set_ylim(32328, 32359)
                    #ax[i, j].set_yticks([64000, 66560, 69120, 71680, 74240, 76800])
            elif metric == 'pop_med_fitness':
                if nb == '8':
                    ax[i, j].set_ylim(110, 117)
                    #ax[i, j].set_yticks([250, 260, 270, 280, 290, 300])
                elif nb == '9':
                    ax[i, j].set_ylim(230, 238)
                    #ax[i, j].set_yticks([500, 520, 540, 560, 580, 600])
                elif nb == '10':
                    ax[i, j].set_ylim(472, 482)
                    #ax[i, j].set_yticks([1000, 1040, 1080, 1120, 1160, 1200])
                elif nb == '11':
                    ax[i, j].set_ylim(968, 978)
                    #ax[i, j].set_yticks([2000, 2080, 2160, 2240, 2320, 2400])
                elif nb == '12':
                    ax[i, j].set_ylim(1962, 1974)
                    #ax[i, j].set_yticks([4000, 4160, 4320, 4480, 4640, 4800])
                elif nb == '13':
                    ax[i, j].set_ylim(3966, 3980)
                    #ax[i, j].set_yticks([8000, 8320, 8640, 8960, 9280, 9600])
                elif nb == '14':
                    ax[i, j].set_ylim(7998, 8014)
                    #ax[i, j].set_yticks([16000, 16640, 17280, 17920, 18560, 19200])
                elif nb == '15':
                    ax[i, j].set_ylim(16088, 16114)
                    #ax[i, j].set_yticks([32000, 33280, 34560, 35840, 37120, 38400])
                elif nb == '16':
                    ax[i, j].set_ylim(32328, 32358)
                    #ax[i, j].set_yticks([64000, 66560, 69120, 71680, 74240, 76800])
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
            ax[i, j].tick_params(axis='y', labelsize=14)
            if i == n - 1:
                ax[i, j].set_xlabel('Generation', fontsize=24)
                ax[i, j].tick_params(axis='x', labelsize=18)
            else:
                ax[i, j].tick_params(labelbottom=False)
                ax[i, j].set_xticklabels([])
            
            if j == 0:
                ax[i, j].set_ylabel(metric_alias[metric], labelpad=10 if i == n - 1 else (15 if i == 0 else 17), fontsize=24)
                #ax[i, j].set_ylabel(metric_alias[metric], labelpad=10 if i != 0 else (22 if metric in ('best_fitness', 'granular_best_fitness') else 10), fontsize=18)
            #else:
                #ax[i, j].tick_params(labelleft=False)
                #ax[i, j].set_yticklabels([])

            ax[i, j].grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)
    return fig

 
def boxplot_grid_cellular_truth_tables(
    data: dict,
    baseline_vs_baseline10retry: bool,
    metric: str,
    gen: int,
    palette_cmp: dict,
    save_png: bool,
    dpi: int
):
    metric_alias = {'granular_best_fitness': r'$\tilde{\overline{\ell}}$', 'best_fitness': r'$\overline{\ell}$', 'best_resiliency': r'$r$', 'best_algebraic_degree': r'$d$', 'best_max_autocorrelation_coefficient': r'$a$', 'real_global_moran_I': r'$I$'}

    n_bits = list(range(8, 16 + 1))
    dataframes_dict = {}
    significance_dict = {}
    
    if baseline_vs_baseline10retry:
        dataframes_dict = {r"$n$": [], "Method": [], metric_alias[metric]: []}
        print("Comparing baseline vs. baseline10retry")
        for nb in n_bits:
            signif = {}
            nb_str = str(nb)
            gen_str = str(gen)
            curr_data = data[nb_str][metric]
            # baseline
            base_values = curr_data['baseline'][gen_str]
            dataframes_dict[r"$n$"].extend([nb] * len(base_values))
            dataframes_dict["Method"].extend([r'Vanilla GA'] * len(base_values))
            dataframes_dict[metric_alias[metric]].extend(base_values)
            # baseline10retry
            retry_values = curr_data['baseline10retry'][gen_str]
            dataframes_dict[r"$n$"].extend([nb] * len(retry_values))
            dataframes_dict["Method"].extend([r'Diversity Preserving GA'] * len(retry_values))
            dataframes_dict[metric_alias[metric]].extend(retry_values)
            is_passed, _ = is_mannwhitneyu_passed(base_values, retry_values, alternative='less', alpha=0.05)
            signif[r'Diversity Preserving GA'] = is_passed
            is_passed, _ = is_mannwhitneyu_passed(retry_values, base_values, alternative='less', alpha=0.05)
            signif[r'Vanilla GA'] = is_passed
            df = pd.DataFrame(dataframes_dict)
            significance_dict[nb_str] = signif
        for nb in n_bits:
            print(f"n={nb}")
            print(significance_dict[str(nb)])
            print()
        # # retrieve max values from metric_alias[metric] to scale y axis properly
        # y_axes_fixed = df[metric_alias[metric]].max()
        # # divide by y_axes_fixed to have all plots with same y axis
        # df[metric_alias[metric]] /= y_axes_fixed
        
        # plot = fastplot.plot(None, None, mode='callback',
        #                      callback=lambda plt: my_callback_boxplot_grid_cellular_truth_tables(plt, {"0": df}, significance_dict, metric_alias[metric], palette_cmp),
        #                      style='latex', **PLOT_ARGS)
        # if save_png:
        #     plot.savefig(f'../analysis/img/cellular_boxplot_baseline_vs_baseline10retry_{metric}_gen{str(gen)}.png', dpi=dpi)
        # plot.savefig(f'../analysis/img/cellular_boxplot_baseline_vs_baseline10retry_{metric}_gen{str(gen)}.pdf', dpi=dpi)
        return
    
    
    for nb in n_bits:
        signif = {}
        nb_str = str(nb)
        gen_str = str(gen)
        curr_data = data[nb_str][metric]
        curr_dataframe_dict = {"Method": [], r"$p$": [], metric_alias[metric]: []}
        # baseline10retry
        base_values = curr_data['baseline10retry'][gen_str]
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
                print(f'Computing significance for n={nb}, method={method}, p={c}')
                print(f'  baseline10retry values: {base_values}')
                print(f'  cellular values: {cellular_values}')
                is_passed, _ = is_mannwhitneyu_passed(base_values, cellular_values, alternative='less', alpha=0.05)
                print(f'    baseline10retry < {method} (p={c}): {is_passed}')
                print()
                signif[method][str(c)] = is_passed
        df = pd.DataFrame(curr_dataframe_dict)
        dataframes_dict[nb_str] = df
        significance_dict[nb_str] = signif
    for nb in n_bits:
        print(f"n={nb}")
        print(significance_dict[str(nb)])
        print()
    
    # plot = fastplot.plot(None, None, mode='callback',
    #                      callback=lambda plt: my_callback_boxplot_grid_cellular_truth_tables(plt, dataframes_dict, significance_dict, metric_alias[metric], palette_cmp),
    #                      style='latex', **PLOT_ARGS)
    plot = my_callback_boxplot_grid_cellular_truth_tables(dataframes_dict, significance_dict, metric_alias[metric], palette_cmp)
    plot.savefig(f'../analysis/img/cellular_boxplot_{metric}_gen{str(gen)}.pdf', dpi=dpi)
    if save_png:
        plot.savefig(f'../analysis/img/cellular_boxplot_{metric}_gen{str(gen)}.png', dpi=dpi)
    plt.clf()
    plt.cla()
    plt.close()


def my_callback_boxplot_grid_cellular_truth_tables(data: dict[str, pd.DataFrame], significance_dict: dict[str, dict[str, dict[str, bool]]], y_title: str, palette_cmp: dict[str, str]): 
    n, m = 3, 3
    fig, ax = plt.subplots(n, m, figsize=(15, 10), layout='tight', squeeze=False)
    n_bits = np.array(list(range(8, 16 + 1))).reshape(n, m)
    for i in range(n):
        for j in range(m):
            nb = str(n_bits[i, j])
            df = data[nb]
            ax[i, j].set_title(f'$n = {nb}$', fontsize=24)
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
                        legend=False, fliersize=2.0, log_scale=None)
            
            # increase thickness of median and quartile lines and also of the box edges
            for line in ax[i, j].artists + ax[i, j].lines:
                line.set_linewidth(1.8)
                line.set_color('black')

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
                            y_axes_fixed = 0.1
                            ax[i, j].text(x_axes, y_axes_fixed, r'\textbf{*}', transform=ax[i, j].transAxes,
                                          ha='center', va='center', fontsize=32, color='black', clip_on=False)
                        except Exception:
                            # fallback: place using data coords near bottom of axis
                            vals = df[(df['Method'] == method_label) & (df[r"$p$"] == str(hue_str))][y_title].dropna()
                            if len(vals) == 0:
                                continue
                            y0, y1 = ax[i, j].get_ylim()
                            yrange = max((y1 - y0), 1e-6)
                            stagger = (hi - (n_hues - 1) / 2.0) * 0.02 * yrange
                            y_coord = y0 + 0.02 * yrange + stagger
                            ax[i, j].text(x_data, y_coord, r'\textbf{*}', ha='center', va='bottom', fontsize=32, color='black', clip_on=False)
            except Exception:
                # keep plotting even if annotations fail
                pass

            ax[i, j].tick_params(axis='y', labelsize=14)
            if i == n - 1:
                ax[i, j].set_xlabel('Method', fontsize=24)
                # increase font size of x tick labels for bottom row
                ax[i, j].tick_params(axis='x', labelsize=22)
            else:
                ax[i, j].tick_params(labelbottom=False)
                ax[i, j].set_xticklabels([])
                ax[i, j].set_xlabel('')
            
            if j == 0:
                ax[i, j].set_ylabel(y_title, labelpad=10 if i == n - 1 else (15 if i == 0 else 17), fontsize=24)
            else:
                # empty y labels for
                ax[i, j].set_ylabel('')
                #ax[i, j].tick_params(labelleft=False)
                #ax[i, j].set_yticklabels([])

            ax[i, j].grid(True, axis='y', which='major', color='gray', linestyle='--', linewidth=0.5)
    return fig


def create_legend(label_color_dict, title=None):
    # fastplot.plot(None, f'legend.pdf', mode='callback',
    #               callback=lambda plt: create_legend_callback(plt, label_color_dict, title),
    #               style='latex', **PLOT_ARGS)
    plot = create_legend_callback(label_color_dict, title)
    plot.savefig(f'../analysis/img/legend.png', dpi=800)
    plot.savefig(f'../analysis/img/legend.pdf', dpi=800)
    plt.clf()
    plt.cla()
    plt.close()


def create_legend_callback(label_color_dict, title=None):
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
    return fig


def heatmap(results_folder: str, n_bits: int, gens: list[int], dupl_retry: int, radius: list[int], cmp_rate: float, pop_size: int, num_gen: int, pressure: int, torus_dim: int, pop_shape: tuple[int, ...], seed_index: int, save_png: bool, dpi: int):
    all_histories = {}
    methods_names = [f'baseline{dupl_retry}retry'] + [f'torus{torus_dim}_radius{r}_cmp{str(cmp_rate)}' for r in radius]
    methods_alias = [r'\notoroid'] + [r'\toroid{' + str(torus_dim) + '}{' + str(r) + '}' for r in radius]
    temp = truth_tables_history(
        results_folder=results_folder,
        pop_size=pop_size,
        n_bits=n_bits,
        gen=num_gen,
        dupl_retry=dupl_retry,
        pressure=pressure,
        torus_dim=0,
        radius=0,
        cmp_rate=0.0,
        seed_index=seed_index
    )['pop_fitness_list'].to_list()
    all_histories[f'baseline{dupl_retry}retry'] = [np.array([float(elem) for elem in temp[i].split(" ")]).reshape(*pop_shape) for i in gens]
    for r in radius:
        method_name = f'torus{torus_dim}_radius{r}_cmp{str(cmp_rate)}'
        temp = truth_tables_history(
            results_folder=results_folder,
            pop_size=pop_size,
            n_bits=n_bits,
            gen=num_gen,
            dupl_retry=0,
            pressure=0,
            torus_dim=torus_dim,
            radius=r,
            cmp_rate=cmp_rate,
            seed_index=seed_index
        )['pop_fitness_list'].to_list()
        all_histories[method_name] = [np.array([float(elem) for elem in temp[i].split(" ")]).reshape(*pop_shape) for i in gens]

    all_errs = []
    for key in all_histories:
        for elem in all_histories[key]:
            all_errs.extend(elem.flatten().tolist())
    min_ = min(all_errs)
    max_ = max(all_errs) # np.percentile(all_errs, 90)
    print("Min: ", min_)
    print("Max: ", max_)

    # plot = fastplot.plot(None, None, mode='callback',
    #                      callback=lambda plt: my_callback_heatmap(plt, vmin=min_, vmax=max_, all_histories=all_histories, gens=gens, methods_names=methods_names, methods_alias=methods_alias),
    #                      style='latex', **PLOT_ARGS)
    plot = my_callback_heatmap(vmin=min_, vmax=max_, all_histories=all_histories, gens=gens, methods_names=methods_names, methods_alias=methods_alias)
    if save_png:
        plot.savefig(f'../analysis/img/heatmap.png', dpi=dpi)
    plot.savefig(f'../analysis/img/heatmap.pdf', dpi=dpi)
    plt.clf()
    plt.cla()
    plt.close()
    return min_, max_


def my_callback_heatmap(vmin: float, vmax: float, all_histories: dict, gens: list[int], methods_names: list[str], methods_alias: list[str]):
    n, m = len(all_histories), len(gens)
    fig, ax = plt.subplots(n, m, figsize=(10, 10), layout='tight', squeeze=False)

    met_i = 0
    for i in range(n):
        method = methods_names[met_i]
        alias = methods_alias[met_i]
        iter_i = 0

        for j in range(m):
            pop = all_histories[method][iter_i]

            _ = ax[i, j].imshow(pop, cmap='inferno', vmin=vmin, vmax=vmax)
            ax[i, j].tick_params(axis='both', which='both', reset=False, bottom=False, top=False, left=False, right=False)
            ax[i, j].tick_params(labelbottom=False)
            ax[i, j].tick_params(labelleft=False)

            if i == 0:
                ax[i, j].set_title(f"Gen. {gens[iter_i]}" if gens[iter_i] < 999 else f"Gen. {gens[iter_i] + 1}", fontsize=18)

            iter_i += 1

        ax[i, 0].set_ylabel(alias, fontsize=18)
        met_i += 1
    return fig


def make_colorbar(vmin, vmax):
    #fastplot.plot(None, f'../analysis/img/colorbar.pdf', mode='callback', callback=lambda plt: my_callback_colorbar(plt, vmin, vmax), style='latex', **PLOT_ARGS)
    plot = my_callback_colorbar(vmin, vmax)
    plot.savefig(f'../analysis/img/colorbar.png', dpi=800)
    plot.savefig(f'../analysis/img/colorbar.pdf', dpi=800)
    plt.clf()
    plt.cla()
    plt.close()

def my_callback_colorbar(vmin, vmax):
    fig, ax = plt.subplots(figsize=(1.1, 10), layout='constrained')
    #fig.subplots_adjust(bottom=0.5)
    # Add a bit more white space on the left with constrained layout
    # rect = [left, bottom, right, top] in figure coordinates
    fig.get_layout_engine().set(w_pad=4/72, h_pad=4/72, hspace=0.05, wspace=0.05, rect=[0.01, 0, 0.95, 1]) # type: ignore

    cmap = plt.get_cmap('inferno')
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    cb = colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label(r'$\tilde{\overline{\ell}}$', fontsize=18, rotation=270, labelpad=15)
    return fig


# =====================================
# Plots PROGRAMS
# =====================================

def boxplot_grid_shuffle_programs(
    data: dict,
    metric: str,
    gen: int,
    palette_length: dict,
    save_png: bool,
    dpi: int
):
    metric_alias = {'granular_best_fitness': r'$\tilde{\overline{\ell}}$', 'best_fitness': r'$\overline{\ell}$', 'best_resiliency': r'$r$', 'best_algebraic_degree': r'$d$', 'best_max_autocorrelation_coefficient': r'$a$', 'real_global_moran_I': r'$I$'}

    n_bits = [9, 10, 11, 12, 13, 14] # list(range(8, 16 + 1))
    init_bin_size = 1
    dataframes_dict = {}
    significance_dict = {}

    for nb in n_bits:
        signif = {}
        nb_str = str(nb)
        gen_str = str(gen)
        curr_data = data[nb_str][metric]
        curr_dataframe_dict = {"Step": [], r"Length": [], metric_alias[metric]: []}
        # baseline
        base_values = curr_data['baseline'][gen_str]
        curr_dataframe_dict["Step"].extend(['GA'] * len(base_values))
        curr_dataframe_dict[r"Length"].extend(['GA'] * len(base_values))
        curr_dataframe_dict[metric_alias[metric]].extend(base_values)
        # programs
        for p in [10, 50, 100]:# [100, 200, 500]:
            for min_, max_ in [(2, 100), (2, 500), (2, 1000)]:#[(2, 5), (2, 10), (2, 20)]:
                method = f'{p}'
                if method not in signif:
                    signif[method] = {}
                cellular_values = curr_data[f'binsize{init_bin_size}_length_{min_}_{max_}_p{p}'][gen_str]
                curr_dataframe_dict["Step"].extend([method] * len(cellular_values))
                curr_dataframe_dict[r"Length"].extend([f'{min_}_{max_}'] * len(cellular_values))
                curr_dataframe_dict[metric_alias[metric]].extend(cellular_values)
                print(f'Computing significance for n={nb}, step={method}, length={min_}_{max_}')
                print(f'  baseline values: {base_values}')
                print(f'  cellular values: {cellular_values}')
                is_passed, _ = is_mannwhitneyu_passed(base_values, cellular_values, alternative='less', alpha=0.05)
                print(f'    baseline < step {method} (length={min_}_{max_}): {is_passed}')
                print()
                signif[method][f'{min_}_{max_}'] = is_passed
        df = pd.DataFrame(curr_dataframe_dict)
        dataframes_dict[nb_str] = df
        significance_dict[nb_str] = signif
    for nb in n_bits:
        print(f"n={nb}")
        print(significance_dict[str(nb)])
        print()

    plot = my_callback_boxplot_grid_shuffle_programs(dataframes_dict, significance_dict, metric_alias[metric], palette_length)
    if save_png:
        plot.savefig(f'../analysis/img/programs_boxplot_{metric}_gen{str(gen)}.png', dpi=dpi)
    plot.savefig(f'../analysis/img/programs_boxplot_{metric}_gen{str(gen)}.pdf', dpi=dpi)
    plt.clf()
    plt.cla()
    plt.close()


def my_callback_boxplot_grid_shuffle_programs(data: dict[str, pd.DataFrame], significance_dict: dict[str, dict[str, dict[str, bool]]], y_title: str, palette_length: dict[str, str]): 
    n, m = 2, 3
    fig, ax = plt.subplots(n, m, figsize=(15, 10), layout='tight', squeeze=False)
    n_bits = np.array([9, 10, 11, 12, 13, 14]).reshape(n, m)
    for i in range(n):
        for j in range(m):
            nb = str(n_bits[i, j])
            df = data[nb]
            ax[i, j].set_title(f'$n = {nb}$', fontsize=24)
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

            sns.boxplot(data=df, x="Step", y=y_title, hue=r"Length", palette=palette_length, ax=ax[i, j], showfliers=False,
                        legend=False, fliersize=2.0, log_scale=None)

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
                methods = list(dict.fromkeys(df['Step'].tolist()))
                hues = list(dict.fromkeys(df[r'Length'].tolist()))
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
                                          ha='center', va='center', fontsize=28, color='black', clip_on=False)
                        except Exception:
                            # fallback: place using data coords near bottom of axis
                            vals = df[(df['Step'] == method_label) & (df[r"Length"] == str(hue_str))][y_title].dropna()
                            if len(vals) == 0:
                                continue
                            y0, y1 = ax[i, j].get_ylim()
                            yrange = max((y1 - y0), 1e-6)
                            stagger = (hi - (n_hues - 1) / 2.0) * 0.02 * yrange
                            y_coord = y0 + 0.02 * yrange + stagger
                            ax[i, j].text(x_data, y_coord, r'\textbf{*}', ha='center', va='bottom', fontsize=28, color='black', clip_on=False)
            except Exception:
                # keep plotting even if annotations fail
                pass

            ax[i, j].tick_params(axis='y', labelsize=14)
            if i == n - 1:
                ax[i, j].set_xlabel('Step', fontsize=24)
                # increase font size of x tick labels for bottom row
                ax[i, j].tick_params(axis='x', labelsize=22)
            else:
                ax[i, j].tick_params(labelbottom=False)
                ax[i, j].set_xticklabels([])
                ax[i, j].set_xlabel('')
            
            if j == 0:
                ax[i, j].set_ylabel(y_title, labelpad=10 if i == n - 1 else (15 if i == 0 else 17), fontsize=24)
            else:
                # empty y labels for
                ax[i, j].set_ylabel('')
                #ax[i, j].tick_params(labelleft=False)
                #ax[i, j].set_yticklabels([])

            ax[i, j].grid(True, axis='y', which='major', color='gray', linestyle='--', linewidth=0.5)
    return fig



# =====================================
# Tables
# =====================================


## TRUTH TABLES

def print_table_max_and_med_non_linearity(data: dict, dupl_retry: int, gen: int, torus_dim: int, radius: int, cmp_rate: float):
    best_known_nonlinearities = {
        5: 12, 6: 26, 7: 56, 8: 116, 9: 240, 10: 492, 11: 992, 12: 2010, 13: 4036, 14: 8120, 15: 16272, 16: 32638
    }
    n_bits = list(range(8, 16 + 1))
    # First row
    table_string = ""
    for nb in n_bits:
        table_string += f"{nb} & "
        table_string += f"{best_known_nonlinearities[nb]} & "
        
        values = data[str(nb)]['best_fitness'][f'baseline{dupl_retry}retry'][str(gen)]
        max_nl = max(values)
        med_nl = round(statistics.median(values), 2)
        table_string += f"{max_nl} & {med_nl} & "
        
        method = f'torus{torus_dim}_radius{radius}_cmp{str(cmp_rate)}'
        values = data[str(nb)]['best_fitness'][method][str(gen)]
        max_nl = max(values)
        med_nl = round(statistics.median(values), 2)
        table_string += f"{max_nl} & {med_nl} & "
        
        table_string = table_string[:-2] + r" \\ " + "\n"
    print(table_string)
    
## PROGRAMS

# =====================================
# Main
# =====================================

def main_programs():
    palette = {'GA': "#360983", '10': "#BCEC0E", '50': "#a61111", '100': "#14D4DB"}
    
    palette_length = {'GA': "#360983", '2_100': "#ECF26E", '2_500': "#A1A726", '2_1000': "#555905"}
    palette_l = {'GA': "#360983", r'2_100': "#ECF26E", r'2_500': "#A1A726", r'2_1000': "#555905"}

    n_bits = [9, 10, 11, 12, 13, 14] # list(range(8, 16 + 1))
    seed_indexes = list(range(1, 30 + 1))
    pressure = 3
    pop_size = 50
    n_iter = 10000
    init_bin_size = 1
    lengths = [(2, 100), (2, 500), (2, 1000)]
    pipeline_iter_steps = [10, 50, 100]
    gens = [200 - 1, 400 - 1, 500 - 1, 1000 - 1]
    results_folder = '../results/'
    persist = True
    
    # _ = persist_dict_with_aggregated_metric_for_programs_for_all_generations(
    #      results_folder,
    #      pop_size,
    #      n_iter,
    #      n_bits,
    #      seed_indexes,
    #      pressure,
    #      init_bin_size,
    #      lengths,
    #      pipeline_iter_steps,
    #      persist
    # )
    # _ = persist_dict_with_distribution_metric_for_programs_for_all_repetitions_fixed_generation(
    #      results_folder,
    #      pop_size,
    #      n_iter,
    #      n_bits,
    #      gens,
    #      seed_indexes,
    #      pressure,
    #      init_bin_size,
    #      lengths,
    #      pipeline_iter_steps,
    #      persist
    # )
    # create_legend(palette_p)
    # quit()
    with open('../analysis/aggregated_metrics_over_generations_programs_pop50_gen10000.json', 'r') as f:
        data = json.load(f)

    with open('../analysis/distribution_metrics_fixed_generation_programs_pop50_gen10000.json', 'r') as f:
        data_box = json.load(f)

    # vmin, vmax = heatmap(
    #     results_folder=results_folder,
    #     n_bits=16,
    #     gens=[1, 5, 10, 500, 999],
    #     dupl_retry=dupl_retry,
    #     radius=[1, 2, 3],
    #     cmp_rate=0.5,
    #     pop_size=pop_size,
    #     num_gen=n_iter,
    #     pressure=pressure,
    #     torus_dim=torus_dim,
    #     pop_shape=(10, 10),
    #     seed_index=49,
    #     save_png=False,
    #     dpi=800,
    # )
    # make_colorbar(vmin, vmax)
    # print_table_max_and_med_non_linearity(
    #     data=data_box,
    #     dupl_retry=dupl_retry,
    #     gen=999,
    #     torus_dim=torus_dim,
    #     radius=1,
    #     cmp_rate=0.5
    # )
    # quit()
    # for metric in ['best_fitness', 'pop_med_fitness', 'real_global_moran_I']:
    #     for cr in [0.5]:
    #         lineplot_grid_over_generations_cellular_truth_tables(
    #             data,
    #             metric=metric,
    #             cmp_rate=cr,
    #             palette=palette,
    #             save_png=False,
    #             dpi=800
    #         )
    #quit()
    boxplot_grid_shuffle_programs(
        data=data_box,
        metric='best_fitness',
        gen=999,
        palette_length=palette_length,
        save_png=False,
        dpi=800
    )


def main_truth_tables():
    palette = {'0': "#B60000", '1': "#06BC4F", '2': "#b10984", '3': "#5620bc"}
    palette_toroid = {r'\notoroid': "#B60000", r'\toroid{2}{1}': "#06BC4F", r'\toroid{2}{2}': "#b10984", r'\toroid{2}{3}': "#5620bc"}
    
    palette_cmp = {'0.0': "#B60000", '0.25': "#BCB8F7", '0.5': "#594fe7", '0.75': "#2d1bcc", '1.0': "#020270"}
    palette_p = {r'$p = 0.25$': "#BCB8F7", r'$p = 0.5$': "#594fe7", r'$p = 0.75$': "#2d1bcc", r'$p = 1.0$': "#020270"}

    n_bits = list(range(8, 16 + 1))
    seed_indexes = list(range(1, 50 + 1))
    pressure = 4
    pop_size = 100
    n_iter = 1000
    dupl_retry = 10
    torus_dim = 2
    radius = [1, 2, 3]
    cmp_rate = [0.25, 0.5, 0.75, 1.0]
    gens = [200 - 1, 400 - 1, 500 - 1, 1000 - 1]
    results_folder = '../results/'
    persist = True
    
    # _ = persist_dict_with_aggregated_metric_for_truth_tables_for_all_generations(
    #      results_folder,
    #      pop_size,
    #      n_iter,
    #      dupl_retry,
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
    #      pop_size,
    #      n_iter,
    #      dupl_retry,
    #      n_bits,
    #      gens,
    #      seed_indexes,
    #      pressure,
    #      torus_dim,
    #      radius,
    #      cmp_rate,
    #      persist
    # )
    # create_legend(palette_toroid)
    # quit()
    with open('../analysis/aggregated_metrics_over_generations_truth_tables_pop100_gen1000.json', 'r') as f:
        data = json.load(f)
        
    with open('../analysis/distribution_metrics_fixed_generation_truth_tables_pop100_gen1000.json', 'r') as f:
        data_box = json.load(f)

    vmin, vmax = heatmap(
        results_folder=results_folder,
        n_bits=16,
        gens=[1, 5, 10, 500, 999],
        dupl_retry=dupl_retry,
        radius=[1, 2, 3],
        cmp_rate=0.5,
        pop_size=pop_size,
        num_gen=n_iter,
        pressure=pressure,
        torus_dim=torus_dim,
        pop_shape=(10, 10),
        seed_index=49,
        save_png=True,
        dpi=800,
    )
    make_colorbar(vmin, vmax)
    # print_table_max_and_med_non_linearity(
    #     data=data_box,
    #     dupl_retry=dupl_retry,
    #     gen=999,
    #     torus_dim=torus_dim,
    #     radius=1,
    #     cmp_rate=0.5
    # )
    # quit()
    # for metric in ['best_fitness', 'pop_med_fitness', 'real_global_moran_I']:
    #     for cr in [0.5]:
    #         lineplot_grid_over_generations_cellular_truth_tables(
    #             data,
    #             metric=metric,
    #             cmp_rate=cr,
    #             palette=palette,
    #             save_png=True,
    #             dpi=800
    #         )
    quit()
    boxplot_grid_cellular_truth_tables(
        data=data_box,
        baseline_vs_baseline10retry=False,
        metric='best_fitness',
        gen=999,
        palette_cmp=palette_cmp,
        save_png=True,
        dpi=800
    )




if __name__ == "__main__":
    main_truth_tables()
    # main_programs()
