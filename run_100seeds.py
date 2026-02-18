#!/usr/bin/env python3
"""100 runs with different random seeds, Precision 1, $10M vol, $3M TVL, ±10%."""

from simulation import *

tvl = D('3000000')
p_low, p_high = D('1800'), D('2200')
prec = 1
num_runs = 100

fees_std_list = []
fees_priv_list = []
priv_profit_list = []
vol_a_std_list = []
arb_fees_std_list = []
arb_fees_priv_list = []

for seed in range(num_runs):
    trades = generate_trades(total_volume=10_000_000, seed=seed)

    std = run_scenario(
        pool_a_range=(p_low, p_high),
        precision=prec,
        with_privileged=False,
        trades=trades,
        tvl=tvl,
    )
    priv = run_scenario(
        pool_a_range=(p_low, p_high),
        precision=prec,
        with_privileged=True,
        trades=trades,
        tvl=tvl,
    )

    fees_std = std['agg_fees_a'] + std['arb_fees_a']
    fees_priv = priv['agg_fees_a'] + priv['arb_fees_a']

    fees_std_list.append(fees_std)
    fees_priv_list.append(fees_priv)
    priv_profit_list.append(priv['priv_profit'])
    vol_a_std_list.append(std['volume_a_pct'])
    arb_fees_std_list.append(std['arb_fees_a'])
    arb_fees_priv_list.append(priv['arb_fees_a'])

    if (seed + 1) % 10 == 0:
        avg_so_far = sum(priv_profit_list) / len(priv_profit_list)
        print(f"  Seed {seed+1:>3}/{num_runs} done | "
              f"avg priv profit so far: ${avg_so_far:,.2f}", flush=True)

# Results
n = num_runs
avg = lambda lst: sum(lst) / n
mn = lambda lst: min(lst)
mx = lambda lst: max(lst)

print()
print("=" * 90)
print(f"100 RUNS  |  Precision 1  |  $10M Volume  |  $3M TVL  |  ±10% Range")
print("=" * 90)
print(f"{'Metric':<30} {'Average':>12} {'Min':>12} {'Max':>12}")
print("-" * 70)
print(f"{'Fees (no priv arb)':<30} ${avg(fees_std_list):>10,.2f} ${mn(fees_std_list):>10,.2f} ${mx(fees_std_list):>10,.2f}")
print(f"{'Fees (w/ priv arb)':<30} ${avg(fees_priv_list):>10,.2f} ${mn(fees_priv_list):>10,.2f} ${mx(fees_priv_list):>10,.2f}")
print(f"{'Fee Delta':<30} ${avg(fees_priv_list) - avg(fees_std_list):>10,.2f}")
print(f"{'Priv Arb Profit':<30} ${avg(priv_profit_list):>10,.2f} ${mn(priv_profit_list):>10,.2f} ${mx(priv_profit_list):>10,.2f}")
leak_pct = avg(priv_profit_list) / avg(fees_priv_list) * 100 if avg(fees_priv_list) > 0 else 0
print(f"{'Leak / Fees %':<30} {leak_pct:>11.2f}%")
print(f"{'Std Arb Fees (no priv)':<30} ${avg(arb_fees_std_list):>10,.2f} ${mn(arb_fees_std_list):>10,.2f} ${mx(arb_fees_std_list):>10,.2f}")
print(f"{'Std Arb Fees (w/ priv)':<30} ${avg(arb_fees_priv_list):>10,.2f} ${mn(arb_fees_priv_list):>10,.2f} ${mx(arb_fees_priv_list):>10,.2f}")
print(f"{'Vol A % (std)':<30} {avg(vol_a_std_list):>11.1f}% {mn(vol_a_std_list):>10.1f}% {mx(vol_a_std_list):>10.1f}%")
