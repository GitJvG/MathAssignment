import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom, norm, t, sem, ttest_1samp, ttest_ind

def descriptive_stat(var, var_name="Variable"): 
    mean = np.mean(var)
    median = np.median(var)
    range = np.max(var) - np.min(var)
    variance = np.var(var)
    std = np.std(var, ddof=1)
    cv = (np.std(var, ddof=1)) / mean * 100
    Standard_error = (sem(var))
    summary_df = pd.DataFrame({'Metric': ['Mean', 'Median', 'Range', 'Variance', 'Std Deviation', 'Coefficient of variation', 'Std Error'],
        var_name : [mean, median, range, variance, std, cv, Standard_error]})      
                         
    return mean, median, range, variance, std, cv, Standard_error, summary_df

df = pd.read_csv("dataset_1672627.csv")
print(df.head())
print(df.info())
print(df.nunique())

df = df.drop(columns=['barrel_id'])
df = df.dropna()
df = pd.get_dummies(df, columns=['quality_check']).astype('Float64')

print(df.info())
sample_size = len(df)

rev_mean, rev_median, rev_range, rev_variance, rev_std, rev_cv, rev_sE, summary_rev = descriptive_stat(df['revenue'], 'revenue')
cost_mean, cost_median, cost_range, cost_variance, cost_std, cost_cv, cost_sE, summary_cost = descriptive_stat(df['costs'], 'costs')
merged_summary = pd.merge(summary_rev, summary_cost)
print(merged_summary)

# Plot histograms for Revenue and Costs per Barrel
plt.figure(figsize=(12, 6))

# Histogram for Revenue
plt.subplot(1, 2, 1)
plt.hist(df['revenue'], bins=16, color='skyblue', edgecolor='black')
plt.title('Revenue per Barrel')
plt.xlabel('Revenue (€)')
plt.ylabel('Frequency')

# Histogram for Costs
plt.subplot(1, 2, 2)
plt.hist(df['costs'], bins=16, color='lightgreen', edgecolor='black')
plt.title('Costs per Barrel')
plt.xlabel('Costs (€)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# 𝑒𝑥𝑝𝑒𝑐𝑡𝑒𝑑 𝑝𝑟𝑜𝑓𝑖𝑡 (€)=𝑋̅𝑟𝑒𝑣/𝑏 ∗𝑏−0.025∗𝑏2 −𝑋̅𝑐𝑜𝑠𝑡/𝑏 ∗𝑏−5125 

# Define the profit function
def expected_profit(b, rev, cost):
    return (rev * b) - (0.025 * b**2) - (cost * b) - 5125

b_values = np.arange(0, 1001)
profit_values = expected_profit(b_values, rev_mean, cost_mean)

max_profit = np.max(profit_values) # Max profit
b_at_max_profit = b_values[np.argmax(profit_values)]  # Number of barrels at max profit

# LaTeX version of the formula
formula = r"$\left(\frac{X_{\text{rev}}}{b} \times b - 0.025 \times b^2 - \frac{X_{\text{cost}}}{b} \times b - 5125\right)$"

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(b_values, profit_values, label=r"Expected Profit (€) " + formula, color='black')
plt.title("Expected Profit as a function of Barrels Produced")
plt.xlabel("Number of Barrels Produced")
plt.ylabel("Expected Profit (€)")
plt.grid(True)
plt.legend()

plt.scatter(b_at_max_profit, max_profit, color='red', zorder=5)
plt.text(b_at_max_profit, max_profit - 1900, f'{b_at_max_profit} barrels\n€{max_profit:.2f}', 
         color='black', ha='center', fontsize=10)

plt.show()

del_mean, del_median, del_range, del_variance, del_std, del_cv, del_sE, del_summary = descriptive_stat(df['delivery_time'], 'delivery_time')

print(del_summary)
print(df['delivery_time'].value_counts(normalize=True) * 100)
N_expired = 15000
growth_rate = 1.65

# Calculate days till expirement and required additives
def expiry(additives=0):
    return(np.log(N_expired) / growth_rate) + (additives / 1.21)

days_till_exp = expiry()
print(f"days_till_exp without additives: {days_till_exp}")

def required_additives(days_till_exp, total_shelf_life):
    days_to_extend =  (total_shelf_life - days_till_exp) 
    return days_to_extend * 1.21

# Needed shelf_life
target_shelf_life = del_mean + 7 # Delivery time + in-store shelf life
print(f"Average delivery days: {del_mean}")
print(f"Needed shelf days: {target_shelf_life}")

print(f"Required additives to reach target: {required_additives(days_till_exp, target_shelf_life)}")
print(f"Required additives to reach target with a day shorter delivery time: {required_additives(days_till_exp, (target_shelf_life-1))}")

print(f"Days to expiry given 1.5 additives: {expiry(1.5)}")
print(f"In-store shelf-life given 1.5 additives {expiry(1.5) - del_mean}")

additives_range = np.arange(0, (10*1.21))
expiry_values = expiry(additives_range)

plt.figure(figsize=(10, 6))
plt.plot(additives_range, expiry_values, label="Days to expiry")
plt.title("Expected days to Expiry")
plt.xlabel("Units of additives added")
plt.ylabel("Expected days to expiry")
plt.grid(True)
plt.legend()

plt.show

# Calculating the probability of failing the quality check
fail_count = df['quality_check_fail'].sum()
fail_probability = fail_count / sample_size
print(f"Probability of failing quality check: {fail_probability}")

p_value = 1 - binom.cdf(fail_count - 1, sample_size, 0.1)
print(f"Probability of observing {fail_count} or more failures given p = {0.1}: {p_value}")

# Calculating the number of barrels that failed the quality check and have a short shelf life
fail_and_short_shelf_life = len(df[(df['quality_check_fail'] == 1) & (df['shelf_life'] < 8)])

# Conditional probability P(Short Shelf Life | Fails Quality Check)
conditional_probability = fail_and_short_shelf_life / fail_count
print(conditional_probability)

# Conditional probability P(Short Shelf Life | Passes Quality Check)
success_count = df['quality_check_pass'].sum()
success_and_short_shelf_life = len(df[(df['quality_check_pass'] == 1) & (df['shelf_life'] < 8)])
print(success_and_short_shelf_life/success_count)

sl_mean, sl_median, sl_range, sl_variance, sl_std, sl_cv, sl_sE, summary_sl = descriptive_stat(df['shelf_life'], 'shelf_life')
print(summary_sl)

plt.figure(figsize=(10, 6))
plt.hist(df['shelf_life'], bins=16, color='skyblue', edgecolor='black', density=True)

# Overlay normal distribution curve
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, sl_mean, sl_std)
plt.plot(x, p, 'k', linewidth=2)
plt.title('Shelf Life Distribution with Normal Curve')
plt.xlabel('Shelf Life')
plt.ylabel('Density')
plt.show()

# Calculate the warranty days required to cover the first 95% of shelf lives
Z = norm.ppf(0.95)
warranty = sl_mean + Z * sl_std 
print(f"Warranty should cover shelf lives till {warranty} days")

# Since the warranty days should probably be a whole number, the % of barrels covered by 10 days and 11 daqys is calculated.
shelf_lives = [10, 11]
for sl in shelf_lives:
    Z = (sl - sl_mean) / sl_std
    percentile = norm.cdf(Z)  # CDF to find the percentile
    print(f"Shelf life of {sl} days covers approximately {percentile * 100:.2f}% of the population.")

population_mean = 2.0

t_statistic, p_value = ttest_1samp(df['delivery_time'], popmean=population_mean, alternative = 'greater')

print(f"Calculated t-statistic: {t_statistic:.2f}")
print(f"P-value: {p_value:.4f}")

critical_t_value = t.ppf(0.95, sample_size-1)
print(f"Critical t-value {critical_t_value}")

#We reject the null hypothesis because p < 0.05 and critical_t_value < t-statistic

print(df['quality_check_pass'].value_counts())
liters_passed = df[df['quality_check_pass'] == 1]['liters']
liters_failed = df[df['quality_check_fail'] == 1]['liters']

t_stat, p_value = ttest_ind(liters_passed, liters_failed, alternative='two-sided')

critical_t_value = t.ppf(0.95, sample_size-2)

print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Critical t-value {critical_t_value}")

# We do not reject null hypothesis because p > 0.05 and critical_t_value > t-statistic

df = sample_size - 1

# Get the t-value for 95% confidence
alpha = 0.05
t_value = t.ppf(1 - alpha/2, df)

# Calculate the margin of error
margin_of_error = t_value * (sl_std / np.sqrt(sample_size))
print(f"Margin of Error: {margin_of_error:.2f} days")

# Calculate the confidence interval
l_bound = sl_mean - margin_of_error
u_bound = sl_mean + margin_of_error

print(f"The true shelf life is likely to fall between {l_bound:.2f} days and {u_bound:.2f} days.")

percentage_variance = margin_of_error / sl_mean * 100
print(f"Percentage of MoE: {percentage_variance}")