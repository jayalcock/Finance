#!/usr/bin/env python3
from RentBuy import RentVsBuy
import numpy as np

# Create and run analysis
analysis = RentVsBuy()
results = analysis.run_analysis()

# Calculate net worth data directly without relying on plot
months = analysis.time_horizon_years * 12
property_values = results["property_values"]
mortgage_balances = results["mortgage_balances"]
selling_costs_pct = analysis.selling_closing_costs_percent

# Buyer's net worth: Home Equity - Potential Selling Costs
buy_net_worth = results["buy_equity"] - (selling_costs_pct * property_values)
# Renter's net worth: Total value of investments
rent_net_worth = results["rent_investments"]
diff = buy_net_worth - rent_net_worth

# Find crossover point - where diff changes from negative to positive
crossover_indices = np.where((diff[:-1] <= 0) & (diff[1:] > 0))[0]
print(f"Crossover indices: {crossover_indices}")

if len(crossover_indices) > 0:
    crossover_month = crossover_indices[0]
    print(f"First crossover at month: {crossover_month}")
    
    # Calculate precise break-even using linear interpolation
    before_diff = diff[crossover_month]
    after_diff = diff[crossover_month + 1]
    
    if after_diff - before_diff != 0:
        fraction = -before_diff / (after_diff - before_diff)
        precise_month = crossover_month + fraction
        break_even_year = precise_month / 12
        print(f"Precise break-even occurs at: {precise_month:.2f} months / {break_even_year:.2f} years")
    else:
        print("Could not calculate precise break-even (division by zero)")
else:
    print("No crossover found in analysis period")

# Print net worth comparison
print("\nNet worth comparison at specific months:")
check_months = [0, 48, 49, 50, 100, months]
for month in check_months:
    if month <= months:
        print(f"Month {month}: Buy=${buy_net_worth[month]:.2f}, Rent=${rent_net_worth[month]:.2f}, Diff=${diff[month]:.2f}")

print(f"\nFinal results from analysis:")
print(f"Break-even year in results: {results['break_even_year']}")
print(f"Buy final net worth: ${results['buy_final_net_worth']:,.2f}")
print(f"Rent final net worth: ${results['rent_final_net_worth']:,.2f}")
