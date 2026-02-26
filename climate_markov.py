import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/kaggle/input/datasets/aryankarmore/worldsim/worldsim_final.csv')
targets = df[df['is_worldsim_target'] == 1].copy()

print(f'Full dataset:    {df.shape[0]:,} rows × {df.shape[1]} columns')
print(f'Target nations:  {targets["iso3"].nunique()} countries × {targets["Year"].nunique()} years (2000–2018)')
print(f'Total rows used: {targets.shape[0]}')
print()
print('Nations in simulation:')
nation_info = targets.groupby(['iso3','country']).size().reset_index().drop(columns=0)
for _, row in nation_info.iterrows():
    print(f'  {row["iso3"]}  →  {row["country"]}')

def classify_climate_state(row):
    if row['shocks_drought_count'] >= 1:
        return 'DROUGHT'
    elif row['shocks_flood_count'] >= 2:
        return 'FLOOD'
    elif row['shocks_heatwave_count'] >= 1 or row['water_scarcity_score'] >= 3:
        return 'HEAT_STRESS'
    else:
        return 'NORMAL'

# Apply to all target rows
targets['climate_state'] = targets.apply(classify_climate_state, axis=1)

# Summary table
summary = targets.groupby(['iso3', 'climate_state']).size().unstack(fill_value=0)
summary['TOTAL'] = summary.sum(axis=1)
print('Climate state frequency per nation (years 2000–2018):')
print()
print(summary.to_string())


nation_order = ['ETH', 'EGY', 'SAU', 'IND', 'CHN', 'BRA', 'NGA', 'AUS', 'DEU', 'USA']
nation_labels = {
    'ETH': 'Ethiopia', 'EGY': 'Egypt', 'SAU': 'Saudi Arabia', 'IND': 'India',
    'CHN': 'China', 'BRA': 'Brazil', 'NGA': 'Nigeria', 'AUS': 'Australia',
    'DEU': 'Germany', 'USA': 'United States'
}

state_to_num = {'NORMAL': 0, 'FLOOD': 1, 'HEAT_STRESS': 2, 'DROUGHT': 3}
num_to_color = {
    0: STATE_COLORS['NORMAL'],
    1: STATE_COLORS['FLOOD'],
    2: STATE_COLORS['HEAT_STRESS'],
    3: STATE_COLORS['DROUGHT'],
}

years = list(range(2000, 2019))

fig, axes = plt.subplots(10, 1, figsize=(16, 9), facecolor='#0d1117')
fig.suptitle('WorldSim Climate State Timeline  ·  2000–2018', 
             fontsize=14, color='#e6edf3', y=0.98, fontweight='bold')

for ax_idx, iso in enumerate(nation_order):
    ax = axes[ax_idx]
    sub = targets[targets['iso3'] == iso].sort_values('Year')
    
    for i, (_, row) in enumerate(sub.iterrows()):
        state = row['climate_state']
        color = STATE_COLORS[state]
        ax.barh(0, 1, left=i, color=color, height=0.7, linewidth=0)
    
    # Label
    ax.set_yticks([0])
    ax.set_yticklabels([f'{iso}  {nation_labels[iso]}'], fontsize=8.5, color='#e6edf3')
    ax.set_xlim(0, 19)
    ax.set_facecolor('#161b22')
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.spines[:].set_visible(False)
    
    if ax_idx == 9:  # Last row — show year labels
        ax.set_xticks(range(19))
        ax.set_xticklabels([str(y) for y in years], fontsize=7.5, color='#8b949e')
        ax.tick_params(axis='x', bottom=True, labelbottom=True)

# Legend
legend_patches = [mpatches.Patch(color=STATE_COLORS[s], label=s) for s in STATES]
fig.legend(handles=legend_patches, loc='lower center', ncol=4, 
           fontsize=9, fancybox=False, edgecolor='#30363d',
           framealpha=0.8, bbox_to_anchor=(0.5, -0.01))

plt.tight_layout(rect=[0, 0.04, 1, 0.97])
plt.savefig('plot_01_climate_timelines.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print('Saved: plot_01_climate_timelines.png')



def count_transitions(state_sequence):
    """Count 4x4 transition matrix from a sequence of state strings."""
    matrix = np.zeros((4, 4), dtype=int)
    for i in range(len(state_sequence) - 1):
        a = STATE_IDX[state_sequence[i]]
        b = STATE_IDX[state_sequence[i + 1]]
        matrix[a][b] += 1
    return matrix

def normalize_rows(matrix):
    """Convert count matrix to probability matrix (row-normalize)."""
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # avoid divide-by-zero
    return matrix / row_sums

# ── MAIN LOOP ── Three lines of logic, as promised
climate_matrices = {}
climate_count_matrices = {}

for iso in nation_order:
    sub = targets[targets['iso3'] == iso].sort_values('Year')
    sub['state'] = sub.apply(classify_climate_state, axis=1)
    matrix = count_transitions(sub['state'].values)
    climate_count_matrices[iso] = matrix
    climate_matrices[iso] = normalize_rows(matrix)

print('✅ Transition matrices computed for all 10 nations')
print()

# Print one as example
iso_example = 'ETH'
print(f'Example — Ethiopia ({iso_example}) Transition Probabilities:')
df_eth = pd.DataFrame(climate_matrices[iso_example], index=STATES, columns=STATES)
print(df_eth.round(3).to_string())
print()
print('Interpretation: ETH drought→drought persistence =', round(climate_matrices['ETH'][STATE_IDX['DROUGHT']][STATE_IDX['DROUGHT']], 3))
print('Compare — DEU drought→drought persistence =', round(climate_matrices['DEU'][STATE_IDX['DROUGHT']][STATE_IDX['DROUGHT']], 3))

# Custom colormap: dark bg → vivid green
cmap_custom = LinearSegmentedColormap.from_list(
    'worldsim', ['#161b22', '#0d4429', '#196c2e', '#2ea043', '#3fb950'], N=256
)

fig, axes = plt.subplots(2, 5, figsize=(18, 7.5), facecolor='#0d1117')
fig.suptitle('Climate Transition Probability Matrices  ·  Per Nation  ·  2000–2018',
             fontsize=13, color='#e6edf3', y=1.01, fontweight='bold')

short_labels = ['NOR', 'DRO', 'FLO', 'HTS']

for idx, iso in enumerate(nation_order):
    ax = axes[idx // 5][idx % 5]
    mat = climate_matrices[iso]
    cnt = climate_count_matrices[iso]
    
    im = ax.imshow(mat, cmap=cmap_custom, vmin=0, vmax=1, aspect='equal')
    
    # Annotate with probability + count
    for r in range(4):
        for c in range(4):
            prob = mat[r][c]
            count = cnt[r][c]
            text_color = '#e6edf3' if prob > 0.4 else '#8b949e'
            ax.text(c, r, f'{prob:.2f}\n({count})', ha='center', va='center',
                    fontsize=7.5, color=text_color, fontfamily='monospace')
    
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(short_labels, fontsize=7.5, color='#8b949e')
    ax.set_yticklabels(short_labels, fontsize=7.5, color='#8b949e')
    ax.set_facecolor('#161b22')
    ax.spines[:].set_color('#30363d')
    
    # Title with dominant climate character
    dominant = STATES[mat.diagonal().argmax()]
    dom_color = STATE_COLORS[dominant]
    ax.set_title(f'{iso}  ·  {nation_labels[iso]}', fontsize=9, 
                 color='#e6edf3', pad=6, fontweight='bold')
    
    # Diagonal highlight (persistence)
    for d in range(4):
        if mat[d][d] > 0:
            rect = plt.Rectangle((d-0.5, d-0.5), 1, 1, fill=False, 
                                  edgecolor='#ffa657', linewidth=1.5, linestyle='--')
            ax.add_patch(rect)

# Add axis labels
for ax_row in axes:
    for ax in ax_row:
        ax.set_xlabel('Next state →', fontsize=7, color='#6e7681')
        ax.set_ylabel('Current state', fontsize=7, color='#6e7681')

# Colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
sm = plt.cm.ScalarMappable(cmap=cmap_custom, norm=Normalize(0, 1))
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Transition probability', color='#8b949e', fontsize=8)
cbar.ax.yaxis.set_tick_params(color='#8b949e')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#8b949e', fontsize=7)
cbar.outline.set_edgecolor('#30363d')

# Legend for labels
legend_text = 'NOR=Normal  DRO=Drought  FLO=Flood  HTS=Heat/Scarcity  |  Dashed box = persistence (diagonal)'
fig.text(0.5, -0.01, legend_text, ha='center', fontsize=8, color='#6e7681')

plt.tight_layout(rect=[0, 0.02, 0.91, 1])
plt.savefig('plot_02_transition_matrices.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print('Saved: plot_02_transition_matrices.png')


fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='#0d1117')
fig.suptitle('Climate State Self-Persistence Probabilities  ·  P(same state next year)',
             fontsize=13, color='#e6edf3', y=1.02, fontweight='bold')

for state_idx, (state, ax) in enumerate(zip(['DROUGHT', 'HEAT_STRESS'], axes)):
    si = STATE_IDX[state]
    
    persistence = {}
    for iso in nation_order:
        persistence[iso] = climate_matrices[iso][si][si]
    
    sorted_items = sorted(persistence.items(), key=lambda x: x[1], reverse=True)
    isos = [x[0] for x in sorted_items]
    probs = [x[1] for x in sorted_items]
    
    bar_colors = []
    for iso, p in zip(isos, probs):
        if iso == 'DEU':
            bar_colors.append('#8b949e')  # grey baseline
        elif p > 0.5:
            bar_colors.append(STATE_COLORS[state])
        elif p > 0:
            bar_colors.append('#ffa657')  # amber for moderate
        else:
            bar_colors.append('#21262d')  # dark for zero
    
    bars = ax.barh(range(len(isos)), probs, color=bar_colors, height=0.65, 
                   linewidth=0, zorder=2)
    
    # Value labels
    for i, (bar, p) in enumerate(zip(bars, probs)):
        label_x = p + 0.02
        label = f'{p:.2f}' if p > 0 else 'NO DATA'
        ax.text(label_x, i, label, va='center', fontsize=9.5, 
                color='#e6edf3', fontfamily='monospace')
    
    ax.set_yticks(range(len(isos)))
    ax.set_yticklabels([f'{iso}  {nation_labels[iso]}' for iso in isos], fontsize=9)
    ax.set_xlim(0, 1.3)
    ax.set_xlabel(f'P({state} → {state})', fontsize=10, color='#8b949e')
    ax.set_title(f'{state} Persistence', fontsize=11, color=STATE_COLORS[state], 
                 pad=8, fontweight='bold')
    ax.axvline(0.5, color='#30363d', linewidth=1, linestyle='--', zorder=1)
    ax.text(0.5, len(isos)-0.3, '0.5 threshold', ha='center', fontsize=7.5, 
            color='#30363d')
    ax.grid(axis='x', alpha=0.3, zorder=0)
    ax.set_facecolor('#161b22')
    ax.spines[:].set_color('#30363d')

plt.tight_layout()
plt.savefig('plot_03_persistence_bars.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print('Saved: plot_03_persistence_bars.png')


fig, ax = plt.subplots(figsize=(14, 5.5), facecolor='#0d1117')
fig.suptitle('Climate State Distribution per Nation  ·  19 years (2000–2018)',
             fontsize=13, color='#e6edf3', fontweight='bold')

state_counts = targets.groupby(['iso3', 'climate_state']).size().unstack(fill_value=0)
# Reorder columns
for s in STATES:
    if s not in state_counts.columns:
        state_counts[s] = 0
state_counts = state_counts[STATES]
state_counts = state_counts.loc[nation_order]

# Convert to percentage
state_pct = state_counts.div(state_counts.sum(axis=1), axis=0) * 100

x = np.arange(len(nation_order))
bottom = np.zeros(len(nation_order))

for state in STATES:
    vals = state_pct[state].values
    bars = ax.bar(x, vals, bottom=bottom, color=STATE_COLORS[state], 
                  label=state, width=0.6, linewidth=0)
    
    # Label if > 10%
    for xi, (v, b) in enumerate(zip(vals, bottom)):
        if v > 10:
            ax.text(xi, b + v/2, f'{v:.0f}%', ha='center', va='center',
                    fontsize=8, color='#0d1117', fontweight='bold')
    bottom += vals

ax.set_xticks(x)
ax.set_xticklabels([f'{iso}\n{nation_labels[iso]}' for iso in nation_order], 
                   fontsize=8.5)
ax.set_ylabel('% of years in state', fontsize=10, color='#8b949e')
ax.set_ylim(0, 105)
ax.legend(loc='upper right', framealpha=0.8, edgecolor='#30363d', fontsize=9)
ax.spines[:].set_color('#30363d')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('plot_04_state_distribution.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print('Saved: plot_04_state_distribution.png')

shock_cols = ['shocks_drought_count', 'shocks_flood_count', 
              'shocks_heatwave_count', 'shocks_wildfire_count', 'shocks_storm_count']
shock_labels = ['Drought', 'Flood', 'Heatwave', 'Wildfire', 'Storm']

shock_totals = targets.groupby('iso3')[shock_cols].sum().loc[nation_order]
shock_totals.columns = shock_labels

cmap_shock = LinearSegmentedColormap.from_list(
    'shocks', ['#161b22', '#1a2744', '#1f6feb', '#58a6ff', '#cae8ff'], N=256
)

fig, ax = plt.subplots(figsize=(11, 6.5), facecolor='#0d1117')
fig.suptitle('Total Shock Events per Nation  ·  EM-DAT Records 2000–2018',
             fontsize=13, color='#e6edf3', fontweight='bold')

im = ax.imshow(shock_totals.values, cmap=cmap_shock, aspect='auto')

for r in range(len(nation_order)):
    for c in range(len(shock_labels)):
        val = shock_totals.values[r][c]
        text_color = '#0d1117' if val > shock_totals.values.max() * 0.6 else '#e6edf3'
        ax.text(c, r, str(int(val)), ha='center', va='center',
                fontsize=11, fontweight='bold', color=text_color)

ax.set_xticks(range(len(shock_labels)))
ax.set_xticklabels(shock_labels, fontsize=10.5)
ax.set_yticks(range(len(nation_order)))
ax.set_yticklabels([f'{iso}  {nation_labels[iso]}' for iso in nation_order], fontsize=9.5)
ax.spines[:].set_color('#30363d')

cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Total events (19 yrs)', color='#8b949e', fontsize=9)
cbar.ax.yaxis.set_tick_params(color='#8b949e')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#8b949e')
cbar.outline.set_edgecolor('#30363d')

plt.tight_layout()
plt.savefig('plot_05_shock_heatmap.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print('Saved: plot_05_shock_heatmap.png')


def simulate_markov(matrix, n_steps=1000, start_state=0):
    """Simulate n_steps of Markov chain, return state counts."""
    current = start_state
    counts = np.zeros(4)
    for _ in range(n_steps):
        probs = matrix[current]
        if probs.sum() == 0:
            probs = np.ones(4) / 4  # fallback uniform if row is zero
        current = np.random.choice(4, p=probs)
        counts[current] += 1
    return counts / counts.sum()

np.random.seed(42)

fig, axes = plt.subplots(2, 5, figsize=(18, 7), facecolor='#0d1117')
fig.suptitle('Validation: Simulated (1000yr) vs Empirical (19yr) State Distribution',
             fontsize=12, color='#e6edf3', y=1.01, fontweight='bold')

for idx, iso in enumerate(nation_order):
    ax = axes[idx // 5][idx % 5]
    
    # Empirical
    sub = targets[targets['iso3'] == iso]
    empirical = np.zeros(4)
    for s in STATES:
        empirical[STATE_IDX[s]] = (sub['climate_state'] == s).sum()
    empirical = empirical / empirical.sum()
    
    # Simulated
    simulated = simulate_markov(climate_matrices[iso])
    
    x = np.arange(4)
    w = 0.35
    
    bars_emp = ax.bar(x - w/2, empirical, w, label='Empirical', 
                      color=[STATE_COLORS[s] for s in STATES], alpha=0.9, linewidth=0)
    bars_sim = ax.bar(x + w/2, simulated, w, label='Simulated',
                      color=[STATE_COLORS[s] for s in STATES], alpha=0.4, 
                      hatch='///', linewidth=0)
    
    ax.set_xticks(x)
    ax.set_xticklabels(['NOR', 'DRO', 'FLO', 'HTS'], fontsize=7.5)
    ax.set_title(f'{iso}  {nation_labels[iso]}', fontsize=8.5, 
                 color='#e6edf3', fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Proportion', fontsize=7.5, color='#6e7681')
    ax.grid(axis='y', alpha=0.3)
    ax.set_facecolor('#161b22')
    ax.spines[:].set_color('#30363d')
    
    if idx == 0:
        ax.legend(fontsize=7, loc='upper right', framealpha=0.7, edgecolor='#30363d')

plt.tight_layout()
plt.savefig('plot_06_validation.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print('Saved: plot_06_validation.png')



