"""
Publication-quality figure generation.

Unified style: FONT_SIZE=11, DPI=300, poverty=#d73027, non-poverty=#2166ac.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Style configuration
FONT_SIZE = 11
DPI = 300
C_POVERTY = '#d73027'
C_NONPOV = '#2166ac'
C_PRE = '#4575b4'
C_DURING = '#fee090'
C_POST = '#d73027'

FP_COLORS = {'pre': C_PRE, 'during': C_DURING, 'post': C_POST}
FP_LABELS = {
    'pre': 'Pre (2000-2012)',
    'during': 'During (2013-2019)',
    'post': 'Post (2020-2023)',
}


def setup_style():
    """Set publication-quality matplotlib defaults."""
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'font.size': FONT_SIZE,
        'figure.dpi': DPI,
        'axes.labelsize': FONT_SIZE,
        'axes.titlesize': FONT_SIZE + 1,
        'xtick.labelsize': FONT_SIZE - 1,
        'ytick.labelsize': FONT_SIZE - 1,
        'legend.fontsize': FONT_SIZE - 2,
        'figure.figsize': (7, 5),
        'axes.grid': True,
        'grid.alpha': 0.2,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
    })


def fig1_dual_scaling(panel, results_pooled, save_path=None, year=2015):
    """
    Figure 1: Dual log-log scatter plots.
    Left: Non-agricultural GDP vs Population (superlinear).
    Right: Agricultural output vs Population (sublinear).
    """
    setup_style()
    d = panel[panel['Year'] == year].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, ycol, eq, title in zip(axes,
        ['NonAgriGDP', 'GrossOutputValue'],
        ['14a', '14b'],
        ['(a) Non-agricultural GDP', '(b) Agricultural output']):

        mask = (d['PopNum'] > 0) & (d[ycol] > 0)
        dd = d[mask]
        log_x = np.log10(dd['PopNum'])
        log_y = np.log10(dd[ycol])
        is_pov = dd['IsPoverty'].fillna(False)

        ax.scatter(log_x[~is_pov], log_y[~is_pov], s=8, alpha=0.3,
                   c=C_NONPOV, label='Non-poverty', rasterized=True)
        ax.scatter(log_x[is_pov], log_y[is_pov], s=8, alpha=0.5,
                   c=C_POVERTY, label='Poverty', rasterized=True)

        if eq in results_pooled:
            r = results_pooled[eq]
            b = r.betas['PopNum']
            a_log10 = r.intercept / np.log(10)
            x_range = np.linspace(log_x.min(), log_x.max(), 100)
            ax.plot(x_range, a_log10 + b * x_range, 'k-', linewidth=2,
                    label=f'$\\beta$ = {b:.3f}')

        ax.set_xlabel(r'log$_{10}$(Population, $10^4$ persons)')
        ax.set_ylabel(f'log$_{{10}}$({ycol}, $10^4$ CNY)')
        ax.set_title(title)
        ax.legend(loc='lower right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def fig2_control_variable_effect(results_pooled, save_path=None):
    """
    Figure 2: β comparison before/after controlling for cultivated area.
    """
    setup_style()
    eqs = {
        '14a': 'NonAgri\n(no control)',
        '14d': 'NonAgri\n(+ land)',
        '14b': 'Agri\n(no control)',
        '14c': 'Agri\n(+ land)',
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [C_NONPOV, C_NONPOV, C_POVERTY, C_POVERTY]
    hatches = ['', '//', '', '//']

    for i, (eq_key, label) in enumerate(eqs.items()):
        if eq_key in results_pooled:
            r = results_pooled[eq_key]
            b = r.betas['PopNum']
            se = r.se['PopNum']
            ax.bar(i, b, yerr=1.96*se, color=colors[i], alpha=0.7,
                   hatch=hatches[i], edgecolor='black', linewidth=0.5,
                   capsize=5)

    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1)
    ax.set_xticks(range(len(eqs)))
    ax.set_xticklabels(eqs.values())
    ax.set_ylabel('$\\beta$ (Population exponent)')
    ax.set_title('Effect of Controlling for Cultivated Area')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def fig3_scaling_timeseries(ts_df, save_path=None):
    """
    Figure 3: β₁(t) and β₂(t) time series with 95% CI bands.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    for eq, color, label in [('14a', C_NONPOV, 'Eq 14a: Non-agri GDP'),
                              ('14b', C_POVERTY, 'Eq 14b: Agri output')]:
        sub = ts_df[(ts_df['Equation'] == eq) & (ts_df['Variable'] == 'PopNum')]
        if len(sub) > 0:
            ax.plot(sub['Year'], sub['Beta'], '-o', color=color,
                    markersize=4, label=label)
            ax.fill_between(sub['Year'], sub['CI_lower'], sub['CI_upper'],
                           alpha=0.15, color=color)

    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvspan(2013, 2020, alpha=0.08, color='green',
               label='Poverty alleviation campaign')
    ax.set_xlabel('Year')
    ax.set_ylabel('Scaling exponent $\\beta$')
    ax.set_title('Scaling Exponents Over Time (95% CI)')
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def fig4_saai_maps(saai_df, years=None, save_path=None):
    """
    Figure 4: SAAI spatial maps for selected years.
    Requires CountyLng, CountyLat columns.
    """
    setup_style()
    if years is None:
        years = [2000, 2010, 2015, 2020, 2023]

    # Filter to years that have spatial data
    has_coords = saai_df.dropna(subset=['CountyLng', 'CountyLat', 'SAAI'])
    years = [y for y in years if y in has_coords['Year'].values]
    if not years:
        print("No years with spatial data available for SAAI maps")
        return None

    n = len(years)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 5))
    if n == 1:
        axes = [axes]

    norm = Normalize(vmin=-3, vmax=3)
    cmap = plt.cm.RdBu_r

    for ax, yr in zip(axes, years):
        d = has_coords[has_coords['Year'] == yr]
        sc = ax.scatter(d['CountyLng'], d['CountyLat'], c=d['SAAI'],
                       cmap=cmap, norm=norm, s=3, alpha=0.7, rasterized=True)
        ax.set_title(f'{int(yr)}')
        ax.set_xlabel('Longitude')
        if yr == years[0]:
            ax.set_ylabel('Latitude')
        ax.set_aspect('equal')

    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                 label='SAAI (ξ)', shrink=0.8)
    fig.suptitle('SAAI Spatial Distribution', fontsize=14, y=1.02)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def fig5_effective_potential_surface(veff_ts, save_path=None):
    """
    Figure 5: V_eff(ξ,t) heatmap showing potential landscape evolution.
    """
    setup_style()
    years = sorted(veff_ts.keys())
    xi_common = np.linspace(-4, 4, 200)

    V_matrix = np.full((len(years), len(xi_common)), np.nan)
    for i, yr in enumerate(years):
        xi_grid, veff = veff_ts[yr]
        V_matrix[i, :] = np.interp(xi_common, xi_grid, veff,
                                    left=np.nan, right=np.nan)

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.pcolormesh(xi_common, years, V_matrix, shading='auto',
                       cmap='viridis_r',
                       vmin=0, vmax=np.nanpercentile(V_matrix, 95))
    plt.colorbar(im, ax=ax, label='$V_{eff}(\\xi)$')
    ax.set_xlabel('SAAI ($\\xi$)')
    ax.set_ylabel('Year')
    ax.set_title('Effective Potential Landscape Evolution')
    ax.axhline(y=2013, color='white', linestyle='--', alpha=0.7)
    ax.axhline(y=2020, color='white', linestyle=':', alpha=0.7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def fig6_drift_diffusion(km_results, save_path=None):
    """
    Figure 6: Drift A(ξ) and diffusion B(ξ) for three FP periods.
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for period, km in km_results.items():
        mask = km.reliable
        if mask.sum() == 0:
            continue
        xi = km.bin_centers[mask]

        axes[0].plot(xi, km.A[mask], '-o', color=FP_COLORS[period],
                    markersize=3, label=FP_LABELS[period])
        axes[0].fill_between(xi,
            km.A[mask] - 1.96*km.A_se[mask],
            km.A[mask] + 1.96*km.A_se[mask],
            alpha=0.15, color=FP_COLORS[period])

        axes[1].plot(xi, km.B[mask], '-o', color=FP_COLORS[period],
                    markersize=3, label=FP_LABELS[period])

    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    axes[0].set_xlabel('SAAI ($\\xi$)')
    axes[0].set_ylabel('Drift $A(\\xi)$')
    axes[0].set_title('(a) Drift coefficient')
    axes[0].legend()

    axes[1].set_xlabel('SAAI ($\\xi$)')
    axes[1].set_ylabel('Diffusion $B(\\xi)$')
    axes[1].set_title('(b) Diffusion coefficient')
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def fig_beta_timeseries_publication(ts_df, save_path=None):
    """
    Publication figure: β₁₄ₐ vs β₁₄b time series + CV universality check.
    Left: β(t) with 95% CI; Right: CV of residuals over time.
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Left panel: β time series ---
    ax = axes[0]
    for eq, color, label in [('14a', C_NONPOV, r'$\beta_{14a}$ Non-agri GDP'),
                              ('14b', C_POVERTY, r'$\beta_{14b}$ Agri output')]:
        sub = ts_df[(ts_df['Equation'] == eq) & (ts_df['Variable'] == 'PopNum')].sort_values('Year')
        if len(sub) == 0:
            continue
        ax.plot(sub['Year'], sub['Beta'], '-o', color=color, markersize=4, label=label)
        ax.fill_between(sub['Year'], sub['CI_lower'], sub['CI_upper'],
                        alpha=0.15, color=color)

    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.7, label=r'$\beta=1$')
    ax.axvline(x=2013, color='green', linestyle=':', alpha=0.5, label='2013 Campaign start')
    ax.axvline(x=2020, color='red', linestyle=':', alpha=0.5, label='2020 Full exit')
    ax.set_xlabel('Year')
    ax.set_ylabel(r'Scaling exponent $\beta$')
    ax.set_title('(a) Scaling exponents over time')
    ax.legend(fontsize=8)

    # --- Right panel: CV = std(residual) / mean(Y) proxy ---
    # We use SE/|β| as a proxy for coefficient of variation stability
    ax2 = axes[1]
    for eq, color, label in [('14a', C_NONPOV, 'CV (Eq 14a)'),
                              ('14b', C_POVERTY, 'CV (Eq 14b)')]:
        sub = ts_df[(ts_df['Equation'] == eq) & (ts_df['Variable'] == 'PopNum')].sort_values('Year')
        if len(sub) == 0:
            continue
        cv = sub['SE'] / sub['Beta'].abs()
        ax2.plot(sub['Year'], cv, '-s', color=color, markersize=3, label=label)

    ax2.axvline(x=2013, color='green', linestyle=':', alpha=0.5)
    ax2.axvline(x=2020, color='red', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('CV (SE / |β|)')
    ax2.set_title('(b) Coefficient of variation over time')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def fig_sectoral_timeseries(ts_df, save_path=None):
    """
    Sectoral β time series: Primary, Secondary, Tertiary GDP ~ PopNum.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    sector_map = {
        '14j': ('#2ca02c', r'$\beta_{primary}$ (Primary GDP)'),
        '14h': ('#ff7f0e', r'$\beta_{secondary}$ (Secondary GDP)'),
        '14i': ('#9467bd', r'$\beta_{tertiary}$ (Tertiary GDP)'),
    }
    for eq, (color, label) in sector_map.items():
        sub = ts_df[(ts_df['Equation'] == eq) & (ts_df['Variable'] == 'PopNum')].sort_values('Year')
        if len(sub) == 0:
            continue
        ax.plot(sub['Year'], sub['Beta'], '-o', color=color, markersize=4, label=label)
        ax.fill_between(sub['Year'], sub['CI_lower'], sub['CI_upper'],
                        alpha=0.15, color=color)

    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.7, label=r'$\beta=1$')
    ax.axvline(x=2013, color='green', linestyle=':', alpha=0.5, label='2013')
    ax.axvline(x=2020, color='red', linestyle=':', alpha=0.5, label='2020')
    ax.set_xlabel('Year')
    ax.set_ylabel(r'Scaling exponent $\beta$')
    ax.set_title('Sectoral Scaling Exponents Over Time (95% CI)')
    ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def fig_urbanization_continuous(quantile_results, interaction_results, save_path=None):
    """
    Continuous urbanization analysis.
    Left: 20-quantile β vs urbanization median + β=1 crossover.
    Right: interaction coefficient β₁ with CI.
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Left: quantile scatter ---
    ax = axes[0]
    qt = quantile_results['table']
    gs = quantile_results['group_stats']
    # NonAgri sector
    for sector, color, marker in [('NonAgri', C_NONPOV, 'o'),
                                   ('Secondary', '#4393c3', 's'),
                                   ('Tertiary', '#762a83', '^')]:
        rows = qt[qt['Sector'] == sector].copy()
        if len(rows) == 0:
            continue
        # Merge median urbanization ratio
        rows = rows.merge(gs[['median_ratio']].reset_index(), left_on='Group', right_on='UrbGroup', how='left')
        ax.errorbar(rows['median_ratio'], rows['Beta'], yerr=1.96*rows['SE'],
                    fmt=marker, color=color, markersize=5, capsize=3,
                    label=sector, alpha=0.8)

    # Crossover line
    if interaction_results is not None and 'crossover_u' in interaction_results:
        u_star = interaction_results['crossover_u']
        if 0 < u_star < 1.5:
            ax.axvline(x=u_star, color='orange', linestyle='--', alpha=0.8,
                       label=f'Crossover u*={u_star:.2f}')

    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Urbanization ratio (NonAgriGDP/RegGDP)')
    ax.set_ylabel(r'$\beta$ (Population exponent)')
    ax.set_title('(a) β vs urbanization level (quantiles)')
    ax.legend(fontsize=8)

    # --- Right: interaction β₁ ---
    ax2 = axes[1]
    if interaction_results is not None:
        labels = ['β₀ (base)', 'β₁ (interaction)']
        vals = [interaction_results['beta0'], interaction_results['beta1']]
        ses = [interaction_results['se_beta0'], interaction_results['se_beta1']]
        colors = [C_NONPOV, C_POVERTY]
        x_pos = np.arange(len(labels))
        for i in range(len(labels)):
            ax2.bar(i, vals[i], yerr=1.96*ses[i], color=colors[i], alpha=0.7,
                    edgecolor='black', linewidth=0.5, capsize=5)
            ax2.text(i, vals[i] + 1.96*ses[i] + 0.01,
                     f'{vals[i]:.4f}', ha='center', fontsize=9)
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels)
        ax2.set_ylabel('Coefficient value')
        ax2.set_title('(b) Interaction regression: ln Y = (β₀+β₁u) ln N')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def fig_period_urbanization_heatmap(period_results, save_path=None):
    """
    Heatmap: 3 time periods × 3 urbanization groups, one panel per sector.
    β values as color + annotation with significance stars.
    """
    setup_style()
    sectors = ['Secondary', 'Tertiary', 'NonAgri']
    periods = list(period_results.keys())
    groups = ['Low', 'Medium', 'High']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, sector in zip(axes, sectors):
        matrix = np.full((len(periods), len(groups)), np.nan)
        annot = [['' for _ in groups] for _ in periods]

        for i, period in enumerate(periods):
            tbl = period_results[period]['table']
            for j, grp in enumerate(groups):
                row = tbl[(tbl['Sector'] == sector) & (tbl['Group'] == grp)]
                if len(row) > 0:
                    b = row.iloc[0]['Beta']
                    p = row.iloc[0]['Wald_beta1_p']
                    matrix[i, j] = b
                    stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                    annot[i][j] = f'{b:.3f}{stars}'

        im = ax.imshow(matrix, cmap='RdYlBu_r', aspect='auto',
                        vmin=0.7, vmax=1.3)
        for i in range(len(periods)):
            for j in range(len(groups)):
                ax.text(j, i, annot[i][j], ha='center', va='center',
                        fontsize=10, fontweight='bold',
                        color='white' if abs(matrix[i, j] - 1) > 0.15 else 'black')

        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups)
        ax.set_yticks(range(len(periods)))
        ax.set_yticklabels(periods)
        ax.set_title(sector)
        ax.set_xlabel('Urbanization group')
        if sector == sectors[0]:
            ax.set_ylabel('Period')

    fig.colorbar(im, ax=axes, label='β (PopNum)', shrink=0.8)
    fig.suptitle('Period × Urbanization × Sector: β Heatmap (stars = Wald β≠1)',
                  fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def fig_sectoral_crossover(sector_interactions, save_path=None):
    """
    Sectoral crossover plot: β(u) = β₀ + β₁·u for Primary/Secondary/Tertiary.
    Three lines with 95% CI shading + β=1 reference + crossover verticals.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(9, 6))

    u_grid = np.linspace(0, 1.2, 200)
    colors = {'Primary': '#2ca02c', 'Secondary': '#ff7f0e', 'Tertiary': '#9467bd'}
    linestyles = {'Primary': '--', 'Secondary': '-.', 'Tertiary': '-'}

    for sector, res in sector_interactions.items():
        b0 = res['beta0']
        b1 = res['beta1']
        cov = res['cov_matrix']
        beta_u = b0 + b1 * u_grid

        # SE(β(u)) = sqrt(Var(β₀) + u²·Var(β₁) + 2u·Cov(β₀,β₁))
        # indices: 0=const, 1=lnN (β₀), 2=u·lnN (β₁)
        var_b0 = cov[1, 1]
        var_b1 = cov[2, 2]
        cov_b0b1 = cov[1, 2]
        se_u = np.sqrt(var_b0 + u_grid**2 * var_b1 + 2 * u_grid * cov_b0b1)

        color = colors.get(sector, 'gray')
        ls = linestyles.get(sector, '-')
        ax.plot(u_grid, beta_u, ls, color=color, linewidth=2.5, label=sector)
        ax.fill_between(u_grid, beta_u - 1.96 * se_u, beta_u + 1.96 * se_u,
                         color=color, alpha=0.12)

        # Crossover vertical
        u_star = res['crossover_u']
        if 0 < u_star < 1.2:
            ax.axvline(x=u_star, color=color, linestyle=':', alpha=0.6, linewidth=1)
            ax.text(u_star + 0.01, 0.55, f'u*={u_star:.2f}',
                    color=color, fontsize=9, rotation=90, va='bottom')

    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='β=1')
    ax.set_xlabel('Urbanization ratio u (NonAgriGDP / RegGDP)')
    ax.set_ylabel('β(u) = β₀ + β₁·u')
    ax.set_title('Sectoral Crossover: β(u) by Industry')
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1.2)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def fig_quantile_lowess(quantile_results, save_path=None):
    """
    20-quantile β scatter with SE errorbars + LOWESS smooth + β=1 reference.
    One panel per sector (NonAgri, Tertiary, Secondary).
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

    setup_style()
    sectors = ['NonAgri', 'Tertiary', 'Secondary']
    colors = {'NonAgri': C_NONPOV, 'Tertiary': '#9467bd', 'Secondary': '#ff7f0e'}

    tbl = quantile_results['table']
    gs = quantile_results['group_stats'].reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for ax, sector in zip(axes, sectors):
        rows = tbl[tbl['Sector'] == sector].copy()
        if len(rows) == 0:
            ax.set_title(f'{sector} (no data)')
            continue
        rows = rows.merge(gs[['UrbGroup', 'median_ratio']], left_on='Group',
                           right_on='UrbGroup', how='left')
        x = rows['median_ratio'].values
        y = rows['Beta'].values
        se = rows['SE'].values

        color = colors.get(sector, 'gray')
        ax.errorbar(x, y, yerr=1.96 * se, fmt='o', color=color,
                     markersize=5, capsize=3, alpha=0.7, label='Quantile β')

        # LOWESS smooth
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() >= 5:
            smoothed = sm_lowess(y[valid], x[valid], frac=0.4, return_sorted=True)
            ax.plot(smoothed[:, 0], smoothed[:, 1], '-', color=color,
                    linewidth=2.5, label='LOWESS')

        ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.7,
                    label='β=1')
        ax.set_xlabel('Urbanization ratio (median)')
        ax.set_ylabel('β (PopNum)')
        ax.set_title(sector)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle('20-Quantile β with LOWESS Smoothing (95% CI)',
                  fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def fig_landau_summary(landau_ts, saai_df=None, save_path=None):
    """
    Figure 7 (paper): Landau potential fit summary.
    Left: Landau coefficients a(t), h(t) with bistable shading.
    Right: Order parameter m(t) for poverty vs non-poverty counties.
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Left: Landau coefficients ---
    ax = axes[0]
    years = landau_ts['Year'].values
    a_vals = landau_ts['a'].values
    ax.plot(years, a_vals, '-o', color=C_NONPOV, markersize=4, label='$a(t)$')
    if 'h' in landau_ts.columns:
        ax.plot(years, landau_ts['h'].values, '-s', color=C_POVERTY,
                markersize=4, label='$h(t)$')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvspan(2013, 2020, alpha=0.08, color='green',
               label='Poverty alleviation')
    # Shade bistable region (a < 0)
    bistable_mask = a_vals < 0
    for i in range(len(years) - 1):
        if bistable_mask[i]:
            ax.axvspan(years[i], years[i + 1], alpha=0.06, color='red')
    ax.set_xlabel('Year')
    ax.set_ylabel('Landau coefficient')
    ax.set_title('(a) Landau coefficients $a(t)$ and $h(t)$')
    ax.legend(fontsize=9)

    # --- Right: Order parameter ---
    ax2 = axes[1]
    if saai_df is not None and 'IsPoverty' in saai_df.columns:
        for is_pov, color, label in [(True, C_POVERTY, 'Poverty counties'),
                                      (False, C_NONPOV, 'Non-poverty counties')]:
            sub = saai_df[saai_df['IsPoverty'] == is_pov]
            m_t = sub.groupby('Year')['SAAI'].mean().reset_index()
            ax2.plot(m_t['Year'], m_t['SAAI'], '-o', color=color,
                     markersize=4, label=label)
    elif 'barrier_height' in landau_ts.columns:
        ax2.plot(years, landau_ts['barrier_height'].values, '-o',
                 color=C_POVERTY, markersize=4, label='Barrier height $\\Delta V$')
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax2.axvspan(2013, 2020, alpha=0.08, color='green')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Order parameter $\\langle\\xi\\rangle$')
    ax2.set_title('(b) Order parameter by poverty status')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def fig7_early_warning(ews_all, ews_pov, save_path=None):
    """
    Figure 7: Early warning signals (variance, autocorrelation, skewness).
    """
    setup_style()
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    for ews, color, label in [(ews_all, C_NONPOV, 'All counties'),
                               (ews_pov, C_POVERTY, 'Poverty counties')]:
        axes[0].plot(ews['Year'], ews['variance'], '-o', color=color,
                     markersize=3, label=label)
        if 'autocorrelation' in ews.columns:
            axes[1].plot(ews['Year'], ews['autocorrelation'], '-o',
                        color=color, markersize=3, label=label)
        axes[2].plot(ews['Year'], ews['skewness'], '-o', color=color,
                     markersize=3, label=label)

    for ax in axes:
        ax.axvspan(2013, 2020, alpha=0.08, color='green')
        ax.legend()

    axes[0].set_ylabel('Variance $\\sigma^2(t)$')
    axes[0].set_title('Early Warning Signals')
    axes[1].set_ylabel('Lag-1 autocorrelation $\\rho(t)$')
    axes[2].set_ylabel('Skewness $\\gamma_1(t)$')
    axes[2].set_xlabel('Year')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig
