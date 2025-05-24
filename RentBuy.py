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
        self.property_appreciation_rate = 0.04  # 4% annual appreciation
        
        # Buying parameters
        self.down_payment_percent = 0.20  # 20% down payment
        self.mortgage_rate = 0.045  # 4.5% mortgage rate
        self.mortgage_term_years = 25  # 25 year amortization
        self.property_tax_rate = 0.00311827  # Vancouver property tax rate (0.311827%)
        self.maintenance_percent = 0.01  # 1% of property value annually
        self.insurance_rate = 0.0035  # 0.35% of property value annually
        self.buying_closing_costs = 0.015 * self.property_value  # 1.5% of property value
        self.selling_closing_costs_percent = 0.08  # 8% (includes realtor fees, legal, etc.)
        
        # Renting parameters
        self.monthly_rent = 2600  # Average 1-bedroom rent in CAD
        self.rent_increase_rate = 0.04  # 4% annual increase (BC allows 2% + inflation)
        self.renter_insurance = 53.65 * 12  # $53.65/month for renter's insurance
        
        # Investment parameters
        self.investment_return_rate = 0.06  # 6% annual return on investments
        
        # Analysis parameters
        self.time_horizon_years = 15  # Compare over 15 years
        self.inflation_rate = 0.025  # 2.5% annual inflation
        self.marginal_tax_rate = 0.40  # 40% marginal tax rate
        
    def calculate_mortgage_payment(self):
        """Calculate the monthly mortgage payment"""
        principal = self.property_value * (1 - self.down_payment_percent)
        monthly_rate = self.mortgage_rate / 12
        num_payments = self.mortgage_term_years * 12
        
        # Monthly payment formula
        if monthly_rate == 0:
            return principal / num_payments
        else:
            return principal * (monthly_rate * (1 + monthly_rate) ** num_payments) / ((1 + monthly_rate) ** num_payments - 1)
    
    def calculate_remaining_mortgage_balance(self, months):
        """Calculate the remaining mortgage balance after a number of months"""
        principal = self.property_value * (1 - self.down_payment_percent)
        monthly_rate = self.mortgage_rate / 12
        num_payments = self.mortgage_term_years * 12
        payment = self.calculate_mortgage_payment()
        
        if monthly_rate == 0:
            return max(0, principal - (payment * months))
        else:
            return max(0, principal * ((1 + monthly_rate) ** num_payments - (1 + monthly_rate) ** months) / 
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
            
            mortgage_interest = self.calculate_mortgage_interest(month)
            mortgage_principal = self.calculate_mortgage_principal(month)
            
            # Tax benefit from mortgage interest and property tax (if applicable)
            tax_benefit = (mortgage_interest + monthly_property_tax) * self.marginal_tax_rate / 12
            
            buy_monthly_cost = monthly_mortgage + monthly_property_tax + monthly_maintenance + monthly_insurance - tax_benefit
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
        fig = plt.figure(figsize=(15, 20))
        
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
        plt.suptitle(f"Rent vs Buy Analysis - Vancouver, BC ({current_date})\n"
                    f"Property: ${self.property_value:,.0f}, Down Payment: {self.down_payment_percent*100:.0f}%, "
                    f"Mortgage Rate: {self.mortgage_rate*100:.2f}%\n"
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
        print(f"Mortgage Rate: {self.mortgage_rate*100:.2f}%")
        print(f"Mortgage Term: {self.mortgage_term_years} years")
        print(f"Property Appreciation Rate: {self.property_appreciation_rate*100:.1f}% annually")
        print(f"Property Tax Rate: {self.property_tax_rate*100:.3f}% annually")
        print(f"Maintenance: {self.maintenance_percent*100:.1f}% of property value annually")
        print(f"Property Insurance: {self.insurance_rate*100:.2f}% of property value annually")
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

def main():
    # Create and configure the analysis
    analysis = RentVsBuy()
    
    # User can modify parameters here
    # For example:
    # analysis.property_value = 1200000
    # analysis.monthly_rent = 2800
    
    # Run the analysis
    results = analysis.run_analysis()
    
    # Plot results (this calculates the break-even point)
    fig = analysis.plot_results(results)
    
    # Print summary (after break-even has been calculated)
    analysis.print_summary(results)
    
    # Show the plot
    plt.show()
    
    print("\nYou can modify the input parameters in the main() function to customize the analysis.")
    print("Consider factors like property location, size, expected length of stay, etc.")

if __name__ == "__main__":
    main()

