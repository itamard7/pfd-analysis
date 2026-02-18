#!/usr/bin/env python3
"""10 seeds, random aggregator precision 1-5 per trade, $10M vol, $3M TVL, ±10%."""

from simulation import *
import random

tvl = D('3000000')
p_low, p_high = D('1800'), D('2200')
num_seeds = 10

fees_std_list = []
fees_priv_list = []
priv_profit_list = []
vol_a_std_list = []
arb_fees_std_list = []
arb_fees_priv_list = []

print(f"Running {num_seeds} seeds | $10M volume | $3M TVL | ±10% | Random precision 1-5 per trade")
print()

for seed in range(num_seeds):
    # Generate trades for this seed
    trades = generate_trades(total_volume=10_000_000, seed=seed)

    # Generate random per-trade precisions (1-5), same for std and priv
    rng = random.Random(seed + 10000)  # different seed than trade generation
    precisions = [rng.randint(1, 5) for _ in range(len(trades))]

    # Standard (no privileged arb)
    std = run_scenario(
        pool_a_range=(p_low, p_high),
        precision=precisions,
        with_privileged=False,
        trades=trades,
        tvl=tvl,
    )

    # Privileged (with privileged arb) — same trades and precisions
    priv = run_scenario(
        pool_a_range=(p_low, p_high),
        precision=precisions,
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

    leak_pct = (priv['priv_profit'] / fees_priv * 100) if fees_priv > 0 else 0
    print(f"  Seed {seed:>2}: Fees(std)=${fees_std:>10,.2f}  "
          f"Fees(priv)=${fees_priv:>10,.2f}  "
          f"Priv Profit=${priv['priv_profit']:>8,.2f}  "
          f"Leak={leak_pct:.2f}%  "
          f"Vol A%={std['volume_a_pct']:.1f}%",
          flush=True)

# Results
n = num_seeds
avg = lambda lst: sum(lst) / n
mn = lambda lst: min(lst)
mx = lambda lst: max(lst)

print()
print("=" * 90)
print(f"{num_seeds} SEEDS  |  Random Prec 1-5  |  $10M Volume  |  $3M TVL  |  ±10% Range")
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
