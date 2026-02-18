#!/usr/bin/env python3
"""$10M volume on $3M TVL, ±10% range, Precision 1 vs 10."""

from simulation import *

# Generate trade sequence — same for all runs
print("Generating trades ($10M volume)...")
trades = generate_trades(total_volume=10_000_000, seed=42)
buy_count = sum(1 for s, _ in trades if s == 'buy')
sell_count = sum(1 for s, _ in trades if s == 'sell')
buy_vol = sum(float(v) for s, v in trades if s == 'buy')
sell_vol = sum(float(v) for s, v in trades if s == 'sell')
print(f"  {len(trades)} trades | Buy: {buy_count} (${buy_vol:,.0f}) | Sell: {sell_count} (${sell_vol:,.0f})")
print()

tvl = D('3000000')
p_low, p_high = D('1800'), D('2200')  # ±10%

results = {}
for prec in [1, 10]:
    for mode in ['standard', 'privileged']:
        label = f"Prec {prec:>2} / {mode}"
        print(f"Running: {label} ...", end='', flush=True)
        res = run_scenario(
            pool_a_range=(p_low, p_high),
            precision=prec,
            with_privileged=(mode == 'privileged'),
            trades=trades,
            tvl=tvl,
        )
        results[(prec, mode)] = res
        print(f" done")

# ---- COMPARISON TABLE ----
print()
print("=" * 100)
print(f"$10M Volume  |  $3M TVL  |  ±10% Range  |  {len(trades)} trades")
print("=" * 100)
print(f"{'Test':<16} {'Fees (no priv)':>14} {'Fees (w/ priv)':>14} "
      f"{'Priv Profit':>12} {'Leak/Fees':>10} {'Vol A %':>8}")
print("-" * 80)

for prec in [1, 10]:
    std = results[(prec, 'standard')]
    priv = results[(prec, 'privileged')]

    fees_std = std['agg_fees_a'] + std['arb_fees_a']
    fees_priv = priv['agg_fees_a'] + priv['arb_fees_a']
    leak = priv['priv_profit']
    leak_pct = (leak / fees_priv * 100) if fees_priv > 0 else 0

    print(f"Precision {prec:<5} ${fees_std:>12,.2f} ${fees_priv:>12,.2f} "
          f"${leak:>10,.2f} {leak_pct:>9.2f}% "
          f"{std['volume_a_pct']:>7.1f}%")

# ---- DETAILED BREAKDOWN ----
print()
print("=" * 100)
print("DETAILED BREAKDOWN")
print("=" * 100)
print(f"{'Test':<16} {'Mode':<12} {'Agg Fees A':>12} {'Arb Fees A':>12} "
      f"{'Total':>12} {'Priv Profit':>12} {'Vol A %':>8} {'P_A':>10} {'P_B':>10}")
print("-" * 100)

for prec in [1, 10]:
    for mode in ['standard', 'privileged']:
        r = results[(prec, mode)]
        total = r['agg_fees_a'] + r['arb_fees_a']
        print(f"Precision {prec:<5} {mode:<12} "
              f"${r['agg_fees_a']:>10,.2f} ${r['arb_fees_a']:>10,.2f} "
              f"${total:>10,.2f} ${r['priv_profit']:>10,.2f} "
              f"{r['volume_a_pct']:>7.1f}% "
              f"${r['final_price_a']:>9,.2f} ${r['final_price_b']:>9,.2f}")
    print()
