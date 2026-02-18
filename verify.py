#!/usr/bin/env python3
"""Detailed step-by-step verification of individual trades."""

from simulation import *


def detailed_test(trade_side, trade_value, tvl, range_a, label):
    print('=' * 80)
    print(f'TEST: {label}')
    print(f'Trade: {trade_side} ETH, ${int(trade_value)} | TVL=${int(tvl):,} | Pool A range={range_a}')
    print('=' * 80)

    pl_a, pu_a = range_a
    pool_a = make_pool('A', D(str(pl_a)), D(str(pu_a)), D(str(tvl)), STANDARD_FEE)
    pool_b = make_pool('B', D('1800'), D('2200'), D(str(tvl)), STANDARD_FEE)

    print(f'\nINITIAL STATE:')
    print(f'  Pool A: L={float(pool_a.L):,.2f}, sqrtP={float(pool_a.sqrt_p):.8f}, '
          f'price={float(pool_a.price()):.6f}')
    print(f'  Pool B: L={float(pool_b.L):,.2f}, sqrtP={float(pool_b.sqrt_p):.8f}, '
          f'price={float(pool_b.price()):.6f}')

    # ---- STEP 1: AGGREGATOR (Precision 1) ----
    print(f'\n--- STEP 1: AGGREGATOR (Precision 1) ---')
    val = D(str(trade_value))

    # Manually check both options
    if trade_side == 'buy':
        eth_if_a, sp_a_new, fee_a = sim_buy_eth(
            pool_a.sqrt_p, pool_a.L, pool_a.fee_rate, pool_a.sqrt_p_upper, val)
        eth_if_b, sp_b_new, fee_b = sim_buy_eth(
            pool_b.sqrt_p, pool_b.L, pool_b.fee_rate, pool_b.sqrt_p_upper, val)

        print(f'  Option A (100% to A):')
        print(f'    fee = ${float(fee_a):.4f}, effective = ${float(val - fee_a):.4f}')
        print(f'    new_sqrtP = {float(sp_a_new):.8f}, new_price = {float(sp_a_new**2):.6f}')
        print(f'    ETH out = {float(eth_if_a):.8f}')
        # Manual check: new_sqrtP = old_sqrtP + effective / L
        manual_sp = float(pool_a.sqrt_p) + float(val - fee_a) / float(pool_a.L)
        print(f'    VERIFY sqrtP: old + eff/L = {float(pool_a.sqrt_p):.8f} + '
              f'{float(val-fee_a):.4f}/{float(pool_a.L):,.2f} = {manual_sp:.8f} '
              f'{"OK" if abs(manual_sp - float(sp_a_new)) < 1e-6 else "MISMATCH!"}')
        # Manual check: ETH out = L * (1/old - 1/new)
        manual_eth = float(pool_a.L) * (1/float(pool_a.sqrt_p) - 1/float(sp_a_new))
        print(f'    VERIFY ETH: L*(1/old - 1/new) = {manual_eth:.8f} '
              f'{"OK" if abs(manual_eth - float(eth_if_a)) < 1e-6 else "MISMATCH!"}')

        print(f'  Option B (100% to B):')
        print(f'    fee = ${float(fee_b):.4f}, effective = ${float(val - fee_b):.4f}')
        print(f'    new_sqrtP = {float(sp_b_new):.8f}, new_price = {float(sp_b_new**2):.6f}')
        print(f'    ETH out = {float(eth_if_b):.8f}')

        winner = 'A' if eth_if_a >= eth_if_b else 'B'
        print(f'  --> Aggregator picks Pool {winner} '
              f'({float(eth_if_a):.8f} vs {float(eth_if_b):.8f} ETH)')
    else:
        mid_p = (pool_a.price() + pool_b.price()) / 2
        eth_total = val / mid_p
        print(f'  Sell: ${float(val)} -> {float(eth_total):.8f} ETH at mid={float(mid_p):.4f}')

        usdc_if_a, sp_a_new, fee_a = sim_sell_eth(
            pool_a.sqrt_p, pool_a.L, pool_a.fee_rate, pool_a.sqrt_p_lower, eth_total)
        usdc_if_b, sp_b_new, fee_b = sim_sell_eth(
            pool_b.sqrt_p, pool_b.L, pool_b.fee_rate, pool_b.sqrt_p_lower, eth_total)

        print(f'  Option A (100% to A):')
        print(f'    fee = {float(fee_a):.8f} ETH, effective = {float(eth_total - fee_a):.8f} ETH')
        print(f'    new_sqrtP = {float(sp_a_new):.8f}, new_price = {float(sp_a_new**2):.6f}')
        print(f'    USDC out = ${float(usdc_if_a):.4f}')

        print(f'  Option B (100% to B):')
        print(f'    fee = {float(fee_b):.8f} ETH, effective = {float(eth_total - fee_b):.8f} ETH')
        print(f'    new_sqrtP = {float(sp_b_new):.8f}, new_price = {float(sp_b_new**2):.6f}')
        print(f'    USDC out = ${float(usdc_if_b):.4f}')

        winner = 'A' if usdc_if_a >= usdc_if_b else 'B'
        print(f'  --> Aggregator picks Pool {winner} '
              f'(${float(usdc_if_a):.4f} vs ${float(usdc_if_b):.4f} USDC)')

    # Execute
    fa_usdc_0 = pool_a.fees_usdc
    fa_eth_0 = pool_a.fees_eth

    vol_a, vol_b = aggregator_execute(trade_side, val, pool_a, pool_b, 1)

    print(f'\n  EXECUTED: ${vol_a:,.2f} to A, ${vol_b:,.2f} to B')
    print(f'  Pool A: price={float(pool_a.price()):.6f}, sqrtP={float(pool_a.sqrt_p):.8f}')
    print(f'  Pool B: price={float(pool_b.price()):.6f}, sqrtP={float(pool_b.sqrt_p):.8f}')
    print(f'  Pool A new fees: USDC +{float(pool_a.fees_usdc - fa_usdc_0):.4f}, '
          f'ETH +{float(pool_a.fees_eth - fa_eth_0):.8f}')

    spread = float((pool_a.price() - pool_b.price()) / pool_b.price() * 100)
    print(f'  Spread: {spread:.6f}%')

    # ---- STEP 2: STANDARD ARB ----
    print(f'\n--- STEP 2: STANDARD ARBITRAGE (0.3% both pools) ---')
    price_a = pool_a.price()
    price_b = pool_b.price()

    if price_a > price_b:
        print(f'  Price A ({float(price_a):.4f}) > Price B ({float(price_b):.4f})')
        print(f'  Direction: Buy ETH in B (cheap), sell in A (expensive)')
        # $1 test
        eth_1, _, _ = sim_buy_eth(pool_b.sqrt_p, pool_b.L, pool_b.fee_rate,
                                   pool_b.sqrt_p_upper, D('1'))
        usdc_1, _, _ = sim_sell_eth(pool_a.sqrt_p, pool_a.L, pool_a.fee_rate,
                                     pool_a.sqrt_p_lower, eth_1)
    elif price_b > price_a:
        print(f'  Price B ({float(price_b):.4f}) > Price A ({float(price_a):.4f})')
        print(f'  Direction: Buy ETH in A (cheap), sell in B (expensive)')
        eth_1, _, _ = sim_buy_eth(pool_a.sqrt_p, pool_a.L, pool_a.fee_rate,
                                   pool_a.sqrt_p_upper, D('1'))
        usdc_1, _, _ = sim_sell_eth(pool_b.sqrt_p, pool_b.L, pool_b.fee_rate,
                                     pool_b.sqrt_p_lower, eth_1)
    else:
        print(f'  Prices equal, no arb')
        eth_1 = ZERO
        usdc_1 = ZERO

    if price_a != price_b:
        print(f'  $1 test: spend $1 -> {float(eth_1):.10f} ETH -> ${float(usdc_1):.6f} USDC')
        print(f'  $1 profit: ${float(usdc_1 - 1):.6f} '
              f'({"PROFITABLE" if usdc_1 > 1 else "NOT profitable"})')
        print(f'  Need spread > ~0.6% for profit. Current spread = {abs(spread):.4f}%')

    fa_usdc_1 = pool_a.fees_usdc
    fa_eth_1 = pool_a.fees_eth
    profit_std, dir_std = find_and_execute_arb(pool_a, pool_b)

    spread2 = float((pool_a.price() - pool_b.price()) / pool_b.price() * 100)
    print(f'\n  Result: dir={dir_std}, profit=${float(profit_std):.4f}')
    print(f'  Pool A: price={float(pool_a.price()):.6f}')
    print(f'  Pool B: price={float(pool_b.price()):.6f}')
    print(f'  Pool A arb fees: USDC +{float(pool_a.fees_usdc - fa_usdc_1):.4f}, '
          f'ETH +{float(pool_a.fees_eth - fa_eth_1):.8f}')
    print(f'  Residual spread: {spread2:.6f}%')

    # ---- STEP 3: PRIVILEGED ARB ----
    print(f'\n--- STEP 3: PRIVILEGED ARBITRAGE (0% on A, 0.3% on B) ---')
    price_a = pool_a.price()
    price_b = pool_b.price()

    if price_a > price_b:
        print(f'  Price A ({float(price_a):.4f}) > Price B ({float(price_b):.4f})')
        print(f'  Direction: Buy in B (0.3%), sell in A (0%)')
        eth_1p, _, _ = sim_buy_eth(pool_b.sqrt_p, pool_b.L, pool_b.fee_rate,
                                    pool_b.sqrt_p_upper, D('1'))
        usdc_1p, _, _ = sim_sell_eth(pool_a.sqrt_p, pool_a.L, ZERO_FEE,
                                      pool_a.sqrt_p_lower, eth_1p)
    elif price_b > price_a:
        print(f'  Price B ({float(price_b):.4f}) > Price A ({float(price_a):.4f})')
        print(f'  Direction: Buy in A (0%), sell in B (0.3%)')
        eth_1p, _, _ = sim_buy_eth(pool_a.sqrt_p, pool_a.L, ZERO_FEE,
                                    pool_a.sqrt_p_upper, D('1'))
        usdc_1p, _, _ = sim_sell_eth(pool_b.sqrt_p, pool_b.L, pool_b.fee_rate,
                                      pool_b.sqrt_p_lower, eth_1p)
    else:
        eth_1p = ZERO
        usdc_1p = ZERO

    if price_a != price_b:
        print(f'  $1 test: spend $1 -> {float(eth_1p):.10f} ETH -> ${float(usdc_1p):.6f} USDC')
        print(f'  $1 profit: ${float(usdc_1p - 1):.6f} '
              f'({"PROFITABLE" if usdc_1p > 1 else "NOT profitable"})')
        print(f'  Need spread > ~0.3% for profit. Current spread = {abs(spread2):.4f}%')

    profit_priv, dir_priv = find_and_execute_arb(pool_a, pool_b, fee_a_override=ZERO_FEE)

    spread3 = float((pool_a.price() - pool_b.price()) / pool_b.price() * 100)
    print(f'\n  Result: dir={dir_priv}, profit=${float(profit_priv):.4f}')
    print(f'  Pool A: price={float(pool_a.price()):.6f}')
    print(f'  Pool B: price={float(pool_b.price()):.6f}')
    print(f'  Residual spread: {spread3:.6f}%')

    # ---- SUMMARY ----
    print(f'\n--- SUMMARY ---')
    print(f'  Std arb profit:  ${float(profit_std):.4f}')
    print(f'  Priv arb profit: ${float(profit_priv):.4f}')
    print(f'  Spread: {abs(spread):.4f}% -> {abs(spread2):.4f}% -> {abs(spread3):.4f}%')
    print()
    return profit_std, profit_priv


# ---- TEST 1: Buy $50K, TVL $500K, ±5% range (deep A) ----
print()
detailed_test('buy', 50000, 500000, (1900, 2100),
              'Buy $50K - Pool A deeper, should route to A, trigger arbs')

print('\n\n')

# ---- TEST 2: Sell $75K, TVL $500K, ±15% range (shallow A) ----
detailed_test('sell', 75000, 500000, (1700, 2300),
              'Sell $75K - Pool A shallower, should route to B, trigger arbs')
