#!/usr/bin/env python3
# Rent vs Buy Analysis for Vancouver, BC
# This script compares the financial outcome of renting versus buying property in Vancouver

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
from datetime import datetime

class RentVsBuy:
    def __init__(self):
        # Default parameters for Vancouver, BC (as of May 23, 2025)
        # Property parameters
        self.property_value = 1100000  # Average property price in CAD
        self.property_appreciation_rate = 0.05  # 5% annual appreciation
        
        # Buying parameters
        self.down_payment_percent = 0.20  # 20% down payment
        self.mortgage_rate = 0.045  # 4.5% mortgage rate
        self.mortgage_term_years = 25  # 25 year amortization
        self.property_tax_rate = 0.00311827  # Vancouver property tax rate (0.311827%)
        self.maintenance_percent = 0.01  # 1% of property value annually
        self.insurance_rate = 0.0035  # 0.35% of property value annually
        self.buying_closing_costs = 0.015 * self.property_value  # 1.5% of property value
        self.selling_closing_costs_percent = 0.08  # 8% (includes realtor fees, legal, etc.)
        self.is_first_time_buyer = True  # Whether buyer is a first-time home buyer
        self.is_new_construction = False  # Whether property is newly constructed
        
        # Renting parameters
        self.monthly_rent = 2800  # Average 1-bedroom rent in CAD
        self.rent_increase_rate = 0.04  # 4% annual increase (BC allows 2% + inflation)
        self.renter_insurance = 53.65 * 12  # $53.65/month for renter's insurance
        
        # Investment parameters
        self.investment_return_rate = 0.06  # 6% annual return on investments
          # Analysis parameters
        self.time_horizon_years = 15  # Compare over 15 years
        self.inflation_rate = 0.025  # 2.5% annual inflation
        self.marginal_tax_rate = 0.40  # 40% marginal tax rate
        
        # Strata/Condo fees
        self.strata_fee_monthly = 400.0  # Monthly strata/condo fee (CAD)
        
    def get_cmhc_premium_rate(self):
        """Determine the CMHC insurance premium rate based on down payment and property value."""
        # CMHC not available for homes > $1.5M or down payment >= 20%
        if self.property_value > 1_500_000 or self.down_payment_percent >= 0.20:
            return 0.0
        
        # Get base premium rate by down payment
        if self.down_payment_percent >= 0.15:
            base_rate = 0.028
        elif self.down_payment_percent >= 0.10:
            base_rate = 0.031
        elif self.down_payment_percent >= 0.05:
            base_rate = 0.04
        else:
            return 0.0  # Should not happen, but fallback
        
        # Add 20 basis points (0.002) for 30-year amortization for first-time buyers or new construction
        # As of December 15, 2024 rule
        if (self.mortgage_term_years > 25 and 
            (self.is_first_time_buyer or self.is_new_construction)):
            base_rate += 0.002
            
        return base_rate

    def calculate_cmhc_premium(self):
        """Calculate the CMHC insurance premium amount."""
        # For homes > $500,000, 5% on first $500k, 10% on remainder
        if self.property_value > 500_000:
            min_down = 0.05 * 500_000 + 0.10 * (self.property_value - 500_000)
        else:
            min_down = 0.05 * self.property_value
        # Only apply if down payment < 20% and property <= $1.5M
        if self.down_payment_percent < 0.20 and self.property_value <= 1_500_000:
            loan_amount = self.property_value - (self.property_value * self.down_payment_percent)
            premium_rate = self.get_cmhc_premium_rate()
            premium = loan_amount * premium_rate
            return premium
        else:
            return 0.0

    def calculate_mortgage_payment(self):
        """Calculate the monthly mortgage payment, including CMHC premium if applicable."""
        principal = self.property_value * (1 - self.down_payment_percent)
        cmhc_premium = self.calculate_cmhc_premium()
        total_principal = principal + cmhc_premium
        monthly_rate = self.mortgage_rate / 12
        num_payments = self.mortgage_term_years * 12
        if monthly_rate == 0:
            return total_principal / num_payments
        else:
            return total_principal * (monthly_rate * (1 + monthly_rate) ** num_payments) / ((1 + monthly_rate) ** num_payments - 1)

    def calculate_remaining_mortgage_balance(self, months):
        """Calculate the remaining mortgage balance after a number of months, including CMHC premium."""
        principal = self.property_value * (1 - self.down_payment_percent)
        cmhc_premium = self.calculate_cmhc_premium()
        total_principal = principal + cmhc_premium
        monthly_rate = self.mortgage_rate / 12
        num_payments = self.mortgage_term_years * 12
        payment = self.calculate_mortgage_payment()
        if monthly_rate == 0:
            return max(0, total_principal - (payment * months))
        else:
            return max(0, total_principal * ((1 + monthly_rate) ** num_payments - (1 + monthly_rate) ** months) / 
                     ((1 + monthly_rate) ** num_payments - 1))
    
    def calculate_mortgage_interest(self, month):
        """Calculate the mortgage interest for a specific month"""
        balance = self.calculate_remaining_mortgage_balance(month - 1)
        monthly_rate = self.mortgage_rate / 12
        return balance * monthly_rate
    
    def calculate_mortgage_principal(self, month):
        """Calculate the mortgage principal payment for a specific month"""
        payment = self.calculate_mortgage_payment()
        interest = self.calculate_mortgage_interest(month)
        return payment - interest
    
    def run_analysis(self):
        """Run the rent vs buy analysis"""
        months = self.time_horizon_years * 12
        
        # Initialize results arrays
        buy_costs = np.zeros(months + 1)
        rent_costs = np.zeros(months + 1)
        buy_equity = np.zeros(months + 1)
        rent_investments = np.zeros(months + 1)
        property_values = np.zeros(months + 1)
        mortgage_balances = np.zeros(months + 1)
        
        # Initial values
        property_values[0] = self.property_value
        mortgage_balances[0] = self.property_value * (1 - self.down_payment_percent)
        buy_costs[0] = self.property_value * self.down_payment_percent + self.buying_closing_costs
        rent_investments[0] = buy_costs[0]  # Assume renter invests the equivalent of down payment + closing costs
        
        # Monthly mortgage payment
        monthly_mortgage = self.calculate_mortgage_payment()
        
        # Monthly analysis
        for month in range(1, months + 1):
            # Property value appreciation
            property_values[month] = property_values[0] * (1 + self.property_appreciation_rate / 12) ** month
            
            # Mortgage balance
            mortgage_balances[month] = self.calculate_remaining_mortgage_balance(month)
            
            # Buy scenario monthly costs
            monthly_property_tax = (property_values[month] * self.property_tax_rate) / 12
            monthly_maintenance = (property_values[month] * self.maintenance_percent) / 12
            monthly_insurance = (property_values[month] * self.insurance_rate) / 12
            monthly_strata = self.strata_fee_monthly
            
            mortgage_interest = self.calculate_mortgage_interest(month)
            mortgage_principal = self.calculate_mortgage_principal(month)
            
            # Tax benefit from mortgage interest and property tax (if applicable)
            tax_benefit = (mortgage_interest + monthly_property_tax) * self.marginal_tax_rate / 12
            
            buy_monthly_cost = monthly_mortgage + monthly_property_tax + monthly_maintenance + monthly_insurance + monthly_strata - tax_benefit
            buy_costs[month] = buy_costs[month - 1] + buy_monthly_cost
            
            # Rent scenario monthly costs
            current_monthly_rent = self.monthly_rent * (1 + self.rent_increase_rate / 12) ** month
            monthly_renter_insurance = self.renter_insurance / 12
            
            rent_monthly_cost = current_monthly_rent + monthly_renter_insurance
            rent_costs[month] = rent_costs[month - 1] + rent_monthly_cost
            
            # Investment growth for renter (down payment equivalent + monthly savings)
            monthly_investment = buy_monthly_cost - rent_monthly_cost
            if monthly_investment > 0:
                # Renter invests the difference if buying costs more
                rent_investments[month] = rent_investments[month - 1] * (1 + self.investment_return_rate / 12) + monthly_investment
            else:
                # If renting costs more, investments grow but no new contributions
                rent_investments[month] = rent_investments[month - 1] * (1 + self.investment_return_rate / 12)
            
            # Home equity
            buy_equity[month] = property_values[month] - mortgage_balances[month]
        
        # Final calculations
        selling_costs = property_values[months] * self.selling_closing_costs_percent
        net_proceeds_from_sale = property_values[months] - mortgage_balances[months] - selling_costs
        
        # Final net worth in each scenario
        buy_final_net_worth = net_proceeds_from_sale
        rent_final_net_worth = rent_investments[months]
        
        # Results
        results = {
            "time_years": np.arange(months + 1) / 12,
            "buy_costs": buy_costs,
            "rent_costs": rent_costs,
            "property_values": property_values,
            "mortgage_balances": mortgage_balances,
            "buy_equity": buy_equity,
            "rent_investments": rent_investments,
            "buy_final_net_worth": buy_final_net_worth,
            "rent_final_net_worth": rent_final_net_worth,
            "buy_vs_rent_difference": buy_final_net_worth - rent_final_net_worth,
            "break_even_year": None  # Will be calculated in plot_results
        }
        
        return results
    
    def money_formatter(self, x, pos):
        """Format numbers as currency"""
        return f'${x:,.0f}'
    
    def plot_results(self, results):
        """Plot the analysis results"""
        formatter = FuncFormatter(self.money_formatter)
        
        # Create figure and subplots
        fig = plt.figure(figsize=(12, 8))
        
        # 1. Cumulative costs
        ax1 = fig.add_subplot(4, 1, 1)
        ax1.plot(results["time_years"], results["buy_costs"], label="Buy (Cumulative Costs)")
        ax1.plot(results["time_years"], results["rent_costs"], label="Rent (Cumulative Costs)")
        ax1.set_title("Cumulative Housing Costs Over Time")
        ax1.set_xlabel("Years")
        ax1.set_ylabel("Cumulative Costs (CAD)")
        ax1.yaxis.set_major_formatter(formatter)
        ax1.legend()
        ax1.grid(True)
        
        # 2. Property Value and Mortgage Balance
        ax2 = fig.add_subplot(4, 1, 2)
        ax2.plot(results["time_years"], results["property_values"], label="Property Value")
        ax2.plot(results["time_years"], results["mortgage_balances"], label="Mortgage Balance")
        ax2.set_title("Property Value and Mortgage Balance")
        ax2.set_xlabel("Years")
        ax2.set_ylabel("Value (CAD)")
        ax2.yaxis.set_major_formatter(formatter)
        ax2.legend()
        ax2.grid(True)
        
        # 3. Net Worth Comparison
        ax3 = fig.add_subplot(4, 1, 3)
        # Calculate net worth over time consistent with final summary
        # Buyer's net worth: Home Equity - Potential Selling Costs
        buy_net_worth_over_time = results["buy_equity"] - (self.selling_closing_costs_percent * results["property_values"])
        # Renter's net worth: Total value of their investments
        rent_net_worth_over_time = results["rent_investments"]
        
        # Find break-even point - more precisely detect when lines cross
        # Calculate the difference between buy and rent net worth
        net_worth_diff = buy_net_worth_over_time - rent_net_worth_over_time
        
        # Find where the difference changes from negative to positive (crossover point)
        # This indicates the buyer's net worth has just exceeded the renter's
        crossover_indices = np.where((net_worth_diff[:-1] <= 0) & (net_worth_diff[1:] > 0))[0]
        
        if len(crossover_indices) > 0:
            # We found a crossover point - this is where buy becomes better than rent
            crossover_month = crossover_indices[0]
            
            # Calculate a more precise break-even point using linear interpolation
            # between the month before and the month of crossover
            if crossover_month >= 0:
                # Get values on both sides of the crossover
                before_diff = net_worth_diff[crossover_month]
                after_diff = net_worth_diff[crossover_month + 1]
                
                # Calculate fraction of month where lines would exactly cross
                if after_diff - before_diff != 0:  # Avoid division by zero
                    fraction = -before_diff / (after_diff - before_diff)
                else:
                    fraction = 0
                    
                # Calculate the precise crossover month
                precise_break_even_month = crossover_month + fraction
                break_even_year = precise_break_even_month / 12
                
                # Update results dictionary with the break-even year
                results["break_even_year"] = break_even_year  # Update results dict for print_summary
                
                # Add vertical line to plot
                ax3.axvline(x=break_even_year, color='green', linestyle='--', alpha=0.7, 
                           label=f'Break-even: {break_even_year:.1f} years')
            else:
                # Break-even occurs immediately
                results["break_even_year"] = 0 
        elif net_worth_diff[0] > 0:
            # Buying is better from the start
            results["break_even_year"] = 0
            ax3.axvline(x=0, color='green', linestyle='--', alpha=0.7,
                       label=f'Break-even at start')
        else:
            # No break-even found in the analysis period
            results["break_even_year"] = None
        
        ax3.plot(results["time_years"], buy_net_worth_over_time, label="Buy (Net Worth)")
        ax3.plot(results["time_years"], rent_net_worth_over_time, label="Rent (Net Worth)")
        ax3.set_title("Net Worth Comparison")
        ax3.set_xlabel("Years")
        ax3.set_ylabel("Net Worth (CAD)")
        ax3.yaxis.set_major_formatter(formatter)
        ax3.legend()
        ax3.grid(True)
        
        # 4. Buy vs Rent Final Comparison
        ax4 = fig.add_subplot(4, 1, 4)
        labels = ['Buy', 'Rent']
        final_values = [results["buy_final_net_worth"], results["rent_final_net_worth"]]
        
        bars = ax4.bar(labels, final_values)
        ax4.set_title(f"Final Net Worth After {self.time_horizon_years} Years")
        ax4.set_ylabel("Net Worth (CAD)")
        ax4.yaxis.set_major_formatter(formatter)
        ax4.grid(True, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax4.annotate(f'${height:,.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        difference = results["buy_vs_rent_difference"]
        winner = "Buying" if difference > 0 else "Renting"
        diff_text = f"{winner} wins by ${abs(difference):,.0f}"
        ax4.text(0.5, 0.9, diff_text, horizontalalignment='center', transform=ax4.transAxes, fontsize=12)
        
        # Adjust layout and add super title
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Get CMHC details for title
        cmhc_info = ""
        cmhc_premium = self.calculate_cmhc_premium()
        if cmhc_premium > 0:
            cmhc_rate = self.get_cmhc_premium_rate()
            cmhc_info = f", CMHC: {cmhc_rate*100:.1f}%"
        
        plt.suptitle(f"Rent vs Buy Analysis - Vancouver, BC ({current_date})\n"
                    f"Property: ${self.property_value:,.0f}, Down Payment: {self.down_payment_percent*100:.0f}%{cmhc_info}, "
                    f"Mortgage Rate: {self.mortgage_rate*100:.2f}%, Term: {self.mortgage_term_years}y\n"
                    f"Initial Rent: ${self.monthly_rent:.0f}/mo, Renter Insurance: ${self.renter_insurance/12:.2f}/mo, Investment Return: {self.investment_return_rate*100:.1f}%", 
                    fontsize=14)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.subplots_adjust(hspace=0.3)
        
        return fig
    
    def print_summary(self, results):
        """Print a summary of the analysis results"""
        print("\n" + "="*80)
        print(f"RENT VS BUY ANALYSIS SUMMARY - VANCOUVER, BC ({datetime.now().strftime('%Y-%m-%d')})")
        print("="*80)
        
        print("\nINPUT PARAMETERS:")
        print(f"Property Value: ${self.property_value:,.0f}")
        print(f"Down Payment: {self.down_payment_percent*100:.0f}% (${self.property_value * self.down_payment_percent:,.0f})")
        
        # CMHC premium information
        cmhc_rate = self.get_cmhc_premium_rate()
        cmhc_premium = self.calculate_cmhc_premium()
        if cmhc_premium > 0:
            print(f"CMHC Insurance: {cmhc_rate*100:.2f}% premium (${cmhc_premium:,.0f})")
            if self.mortgage_term_years > 25 and (self.is_first_time_buyer or self.is_new_construction):
                print(f"  - Includes 0.20% premium for 30-year amortization")
                print(f"  - Eligible as {'first-time buyer' if self.is_first_time_buyer else 'new construction'}")
        else:
            print("CMHC Insurance: Not required (20%+ down payment or property > $1.5M)")
            
        print(f"Mortgage Rate: {self.mortgage_rate*100:.2f}%")
        print(f"Mortgage Term: {self.mortgage_term_years} years")
        print(f"Property Appreciation Rate: {self.property_appreciation_rate*100:.1f}% annually")
        print(f"Property Tax Rate: {self.property_tax_rate*100:.3f}% annually")
        print(f"Maintenance: {self.maintenance_percent*100:.1f}% of property value annually")
        print(f"Property Insurance: {self.insurance_rate*100:.2f}% of property value annually")
        print(f"Strata/Condo Fees: ${self.strata_fee_monthly:,.2f} per month")
        print(f"Buying Closing Costs: ${self.buying_closing_costs:,.0f} ({self.buying_closing_costs/self.property_value*100:.1f}% of property)")
        print(f"Selling Closing Costs: {self.selling_closing_costs_percent*100:.1f}% of final property value")
        print(f"Initial Monthly Rent: ${self.monthly_rent:.0f}")
        print(f"Rent Increase Rate: {self.rent_increase_rate*100:.1f}% annually")
        print(f"Renter Insurance: ${self.renter_insurance/12:.2f}/month (${self.renter_insurance:.2f}/year)")
        print(f"Investment Return Rate: {self.investment_return_rate*100:.1f}% annually")
        print(f"Analysis Period: {self.time_horizon_years} years")
        print(f"Inflation Rate: {self.inflation_rate*100:.1f}% annually")
        print(f"Marginal Tax Rate: {self.marginal_tax_rate*100:.0f}%")
          # Calculate average monthly costs
        avg_buy_monthly = (results["buy_costs"][-1] - results["buy_costs"][0]) / (self.time_horizon_years * 12)
        avg_rent_monthly = (results["rent_costs"][-1] - results["rent_costs"][0]) / (self.time_horizon_years * 12)
        
        print("\nRESULTS:")
        
        # Show mortgage details including CMHC premium
        mortgage_payment = self.calculate_mortgage_payment()
        principal = self.property_value * (1 - self.down_payment_percent)
        cmhc_premium = self.calculate_cmhc_premium()
        total_borrowed = principal + cmhc_premium
        
        print("\nMortgage Details:")
        print(f"Monthly Payment: ${mortgage_payment:,.2f}")
        print(f"Principal Amount: ${principal:,.0f}")
        if cmhc_premium > 0:
            cmhc_rate = self.get_cmhc_premium_rate()
            print(f"CMHC Premium: ${cmhc_premium:,.0f} ({cmhc_rate*100:.2f}% of loan)")
            print(f"Total Borrowed (Principal + CMHC): ${total_borrowed:,.0f}")
            if self.mortgage_term_years > 25 and (self.is_first_time_buyer or self.is_new_construction):
                print(f"  - Includes 0.20% premium increase for 30-year amortization")
                print(f"  - Eligible due to {'first-time buyer status' if self.is_first_time_buyer else 'new construction'}")
        
        print("\nMonthly Costs:")
        print(f"Average Monthly Cost - Buying: ${avg_buy_monthly:,.0f}")
        print(f"Average Monthly Cost - Renting: ${avg_rent_monthly:,.0f}")
        print(f"Monthly Difference: ${abs(avg_buy_monthly - avg_rent_monthly):,.0f} ({'buying' if avg_buy_monthly > avg_rent_monthly else 'renting'} costs more)")
        
        print(f"\nFinal Property Value: ${results['property_values'][-1]:,.0f}")
        print(f"Final Mortgage Balance: ${results['mortgage_balances'][-1]:,.0f}")
        print(f"Final Home Equity: ${results['buy_equity'][-1]:,.0f}")
        print(f"Final Investment Portfolio (Renting): ${results['rent_investments'][-1]:,.0f}")
        
        print(f"\nFinal Net Worth - Buying: ${results['buy_final_net_worth']:,.0f}")
        print(f"Final Net Worth - Renting: ${results['rent_final_net_worth']:,.0f}")
        
        difference = results["buy_vs_rent_difference"]
        print(f"Buy vs Rent Difference: ${abs(difference):,.0f} ({'buying' if difference > 0 else 'renting'} wins)")
        
        if results["break_even_year"] is not None:
            if results["break_even_year"] > 0:
                years = int(results["break_even_year"])
                months = round((results["break_even_year"] - years) * 12)
                # Handle case where months rounds up to 12
                if months == 12:
                    years += 1
                    months = 0
                print(f"Break-even Point: {years} years and {months} months")
            else:
                print("Break-even Point: Immediate (buying better from start)")
        else:
            print("Break-even Point: None within the analysis period")
        
        print("\nNOTE: This analysis is based on the given assumptions and market conditions.")
        print("      Actual results may vary based on changes in interest rates, housing market,")
        print("      rental market, and personal financial circumstances.")
        print("="*80)

    def calculate_optimal_down_payment(self, min_percent=0.05, max_percent=0.50, step=0.05):
        """Calculate the optimal down payment percentage
        
        Args:
            min_percent: Minimum down payment to test (default: 5%)
            max_percent: Maximum down payment to test (default: 50%)
            step: Step size between percentages to test (default: 5%)
            
        Returns:
            dict: Results containing optimal down payment and analysis for each tested percentage
        """
        percentages = np.arange(min_percent, max_percent + step, step)
        results_by_percent = {}
        best_percent = None
        best_difference = float('-inf')
        
        # Store original down payment percentage to restore later
        original_down_payment = self.down_payment_percent
        
        for percent in percentages:
            # Set the current down payment percentage
            self.down_payment_percent = percent
            
            # Run the analysis with this down payment
            results = self.run_analysis()
            
            # Calculate the financial advantage of buying vs renting
            buy_vs_rent_diff = results["buy_final_net_worth"] - results["rent_final_net_worth"]
            
            # Calculate break-even point for this down payment
            # Create a temporary figure to calculate the break-even point
            temp_fig = self.plot_results(results)
            plt.close(temp_fig)  # Close the figure as we don't need to display it
            
            # Store results for this percentage
            results_by_percent[percent] = {
                "buy_final_net_worth": results["buy_final_net_worth"],
                "rent_final_net_worth": results["rent_final_net_worth"],
                "difference": buy_vs_rent_diff,
                "break_even_year": results["break_even_year"],
                "monthly_mortgage": self.calculate_mortgage_payment()
            }
            
            # Check if this is the best so far
            if buy_vs_rent_diff > best_difference:
                best_difference = buy_vs_rent_diff
                best_percent = percent
        
        # Restore original down payment
        self.down_payment_percent = original_down_payment
        
        # Return the results
        return {
            "optimal_percent": best_percent,
            "optimal_difference": best_difference,
            "results_by_percent": results_by_percent
        }
    
    def plot_down_payment_comparison(self, optimal_results):
        """Plot the comparison of different down payment percentages
        
        Args:
            optimal_results: Results from calculate_optimal_down_payment
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        # Extract data
        percentages = sorted(list(optimal_results["results_by_percent"].keys()))
        differences = [optimal_results["results_by_percent"][p]["difference"] for p in percentages]
        buy_net_worths = [optimal_results["results_by_percent"][p]["buy_final_net_worth"] for p in percentages]
        rent_net_worths = [optimal_results["results_by_percent"][p]["rent_final_net_worth"] for p in percentages]
        
        # Highlight optimal percentage
        optimal_percent = optimal_results["optimal_percent"]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot financial advantage
        ax1.plot(percentages, differences, 'o-', color='blue')
        ax1.axvline(x=optimal_percent, color='green', linestyle='--', 
                   label=f'Optimal: {optimal_percent*100:.0f}%')
        ax1.set_title('Financial Advantage of Buying vs Renting by Down Payment')
        ax1.set_xlabel('Down Payment Percentage')
        ax1.set_ylabel('Buy vs Rent Advantage (CAD)')
        ax1.grid(True)
        ax1.legend()
        ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
        ax1.yaxis.set_major_formatter(FuncFormatter(self.money_formatter))
        
        # Plot net worths
        ax2.plot(percentages, buy_net_worths, 'o-', color='blue', label='Buy Net Worth')
        ax2.plot(percentages, rent_net_worths, 'o-', color='red', label='Rent Net Worth')
        ax2.axvline(x=optimal_percent, color='green', linestyle='--',
                   label=f'Optimal: {optimal_percent*100:.0f}%')
        ax2.set_title('Final Net Worth by Down Payment')
        ax2.set_xlabel('Down Payment Percentage')
        ax2.set_ylabel('Final Net Worth (CAD)')
        ax2.grid(True)
        ax2.legend()
        ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
        ax2.yaxis.set_major_formatter(FuncFormatter(self.money_formatter))
        
        plt.tight_layout()
        
        return fig
    
    def print_optimal_down_payment_summary(self, optimal_results):
        """Print a summary of the optimal down payment analysis
        
        Args:
            optimal_results: Results from calculate_optimal_down_payment
        """
        print("\n" + "="*80)
        print(f"OPTIMAL DOWN PAYMENT ANALYSIS - VANCOUVER, BC ({datetime.now().strftime('%Y-%m-%d')})")
        print("="*80)
        
        # Extract optimal percentage
        optimal_percent = optimal_results["optimal_percent"]
        optimal_diff = optimal_results["optimal_difference"]
        
        print(f"\nOptimal Down Payment: {optimal_percent*100:.0f}%")
        print(f"Financial Advantage: ${optimal_diff:,.0f}")
        
        # Table header
        print("\nDOWN PAYMENT COMPARISON:")
        print(f"{'Down Payment':<15} {'Monthly Pmt':<15} {'Break-even':<15} {'Buyer Net Worth':<20} {'Renter Net Worth':<20} {'Advantage':<15}")
        print("-" * 100)
        
        # Sort percentages
        sorted_percentages = sorted(optimal_results["results_by_percent"].keys())
        
        # Print results for each percentage
        for percent in sorted_percentages:
            result = optimal_results["results_by_percent"][percent]
            buy_worth = result["buy_final_net_worth"]
            rent_worth = result["rent_final_net_worth"]
            diff = result["difference"]
            monthly_payment = result["monthly_mortgage"]
            
            # Format the break-even time
            break_even = result["break_even_year"]
            if break_even is None:
                break_even_str = "Never"
            elif break_even == 0:
                break_even_str = "Immediate"
            else:
                years = int(break_even)
                months = round((break_even - years) * 12)
                if months == 12:
                    years += 1
                    months = 0
                break_even_str = f"{years}y {months}m"
            
            # Highlight the optimal percentage
            highlight = "* " if percent == optimal_percent else "  "
            
            print(f"{highlight}{percent*100:<13.0f}% ${monthly_payment:<13,.0f} {break_even_str:<15} ${buy_worth:<18,.0f} ${rent_worth:<18,.0f} ${diff:<13,.0f}")
        
        print("\nNOTE: The optimal down payment maximizes your financial advantage (buyer net worth - renter net worth).")
        print("      This analysis considers your specific financial parameters including mortgage rate,")
        print("      property appreciation rate, investment returns, and tax implications.")
        print("      A higher down payment reduces mortgage interest but may reduce potential investment returns.")
        print("="*80)

def main():
    # Create and configure the analysis
    analysis = RentVsBuy()
    
    # =====================================================================
    # USER CONFIGURATION SECTION
    # =====================================================================
    # Modify these parameters to match your specific situation
    
    # Property parameters
    # analysis.property_value = 1200000          # Property price in CAD
    # analysis.property_appreciation_rate = 0.04  # 4% annual appreciation
      # Buying parameters
    # analysis.down_payment_percent = 0.20       # 20% down payment
    # analysis.mortgage_rate = 0.04              # 4% mortgage rate
    # analysis.mortgage_term_years = 30          # 30 year amortization
    # analysis.is_first_time_buyer = True        # First-time home buyer status
    # analysis.is_new_construction = False       # New construction property
    
    # Renting parameters
    # analysis.monthly_rent = 2800               # Monthly rent in CAD
    # analysis.rent_increase_rate = 0.03         # 3% annual increase
    
    # Investment parameters
    # analysis.investment_return_rate = 0.07     # 7% annual return
    
    # Analysis parameters
    # analysis.time_horizon_years = 10           # Compare over 10 years
    
    # =====================================================================
    # ANALYSIS OPTIONS
    # =====================================================================
    
    # Find the optimal down payment percentage (may take a moment to run)
    calculate_optimal = True                     # Set to False to skip
    
    if calculate_optimal:
        # Calculate the optimal down payment
        print("\nCalculating optimal down payment percentage...")
        optimal_results = analysis.calculate_optimal_down_payment(
            min_percent=0.05,  # Start at 5% down payment
            max_percent=0.50,  # Up to 50% down payment
            step=0.05          # Test every 5% increment
        )
        
        # Plot the comparison of different down payments
        opt_fig = analysis.plot_down_payment_comparison(optimal_results)
        
        # Print optimal down payment summary
        analysis.print_optimal_down_payment_summary(optimal_results)
        
        # Set the optimal down payment for the main analysis
        analysis.down_payment_percent = optimal_results['optimal_percent']
        print(f"\nSetting down payment to optimal value: {analysis.down_payment_percent*100:.0f}%")
    
    # Run the standard analysis
    results = analysis.run_analysis()
    
    # Plot results (this calculates the break-even point)
    fig = analysis.plot_results(results)
    
    # Print summary (after break-even has been calculated)
    analysis.print_summary(results)
    
    # Show the plot(s)
    plt.show()
    
    print("\nYou can modify the input parameters in the main() function to customize the analysis.")
    print("Set calculate_optimal=True to find the optimal down payment percentage.")
    print("Consider factors like property location, size, expected length of stay, etc.")

if __name__ == "__main__":
    main()

