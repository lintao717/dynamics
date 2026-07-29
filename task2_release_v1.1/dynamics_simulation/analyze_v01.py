import pandas as pd
import numpy as np

df = pd.read_parquet('d:/舆情分析/Tibet_data_collector/data/exported/TIB-2025-001_uead.parquet')

print('=== TIME WINDOWS ===')
tw = df['time_window'].value_counts().sort_index()
print(f'Number of unique time windows: {len(tw)}')
print(tw.to_string())
print()

print('=== STATE COUNTS PER WINDOW ===')
for w in sorted(df['time_window'].unique()):
    sub = df[df['time_window'] == w]
    n_A = (sub['observed_active'] == 1).sum()
    n_D = ((sub['previously_active'] == 1) & (sub['observed_active'] == 0)).sum()
    n_react = sub['reactivated'].sum()
    n_shock = (sub['observed_action'] == 'external_shock').sum()
    n_total = len(sub)
    print(f'{w}: N={n_total:4d} A={n_A:3d} D={n_D:3d} react={n_react:3d} shock={n_shock:3d}')
print()

print('=== STANCE vs ACTIVE RATE ===')
for stance in df['public_stance'].unique():
    sub = df[df['public_stance'] == stance]
    if len(sub) > 10:
        active_rate = (sub['observed_active'] == 1).mean()
        print(f'{stance:25s}: n={len(sub):4d} arousal={sub["arousal_score"].mean():.3f} active_rate={active_rate:.3f}')
print()

print('=== REACTIVATION ===')
prev = df[df['previously_active'] == 1]
react_count = prev['reactivated'].sum()
print(f'Previously active rows: {len(prev)}')
print(f'Reactivated count: {react_count}')
print(f'Reactivation rate: {react_count/len(prev)*100:.1f}%')
print()

print('=== AGENT TRAJECTORIES ===')
traj_lens = df.groupby('agent_id').size()
print(f'Total unique agents: {len(traj_lens)}')
print(f'Mean posts/agent: {traj_lens.mean():.2f}')
for k in [2, 3, 5, 10]:
    print(f'Agents >= {k} posts: {(traj_lens >= k).sum()}')
print()

print('=== CORRELATIONS ===')
print(f'neighbor_active vs observed_active: {df["neighbor_active_count"].corr(df["observed_active"]):.4f}')
print(f'arousal vs observed_active: {df["arousal_score"].corr(df["observed_active"]):.4f}')
print(f'fatigue vs observed_active: {df["fatigue_feature_value"].corr(df["observed_active"]):.4f}')
print(f'shock_intensity vs observed_active: {df["shock_intensity"].corr(df["observed_active"]):.4f}')
print()

print('=== TIME WINDOW DATE RANGE ===')
print(f'Min: {df["time_window"].min()}')
print(f'Max: {df["time_window"].max()}')
unique_tw = sorted(df['time_window'].unique())
print(f'Windows: {len(unique_tw)}')
print(f'First 5: {unique_tw[:5]}')
print(f'Last 5: {unique_tw[-5:]}')
print()

print('=== TRANSITION COUNTS (agent-level) ===')
# Count agents that go from A to D (active in window t, inactive in t+1)
agent_windows = df.pivot_table(
    index='agent_id', columns='time_window', values='observed_active', aggfunc='first'
).fillna(0)
print(f'Agent-window matrix shape: {agent_windows.shape}')

# Count transitions
transitions_A_to_D = 0
transitions_D_to_A = 0
transitions_stay_A = 0
transitions_stay_D = 0
agents_with_transitions = 0

for agent_id, row in agent_windows.iterrows():
    vals = row.values
    has_transition = False
    for t in range(len(vals) - 1):
        if vals[t] == 1 and vals[t+1] == 0:
            transitions_A_to_D += 1
            has_transition = True
        elif vals[t] == 0 and vals[t+1] == 1:
            transitions_D_to_A += 1
            has_transition = True
        elif vals[t] == 1 and vals[t+1] == 1:
            transitions_stay_A += 1
        elif vals[t] == 0 and vals[t+1] == 0:
            transitions_stay_D += 1
    if has_transition:
        agents_with_transitions += 1

print(f'A->D transitions: {transitions_A_to_D}')
print(f'D->A transitions: {transitions_D_to_A}')
print(f'Stay A: {transitions_stay_A}')
print(f'Stay D: {transitions_stay_D}')
print(f'Agents with transitions: {agents_with_transitions}')

if transitions_A_to_D + transitions_stay_A > 0:
    p_A_to_D = transitions_A_to_D / (transitions_A_to_D + transitions_stay_A)
    print(f'Observed P(A->D): {p_A_to_D:.4f}')
if transitions_D_to_A + transitions_stay_D > 0:
    p_D_to_A = transitions_D_to_A / (transitions_D_to_A + transitions_stay_D)
    print(f'Observed P(D->A): {p_D_to_A:.4f}')
