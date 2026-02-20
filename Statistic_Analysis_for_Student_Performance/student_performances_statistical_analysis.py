import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import (
    skew, kurtosis, norm)
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load Dataset
df = pd.read_csv("student_performance.csv")

column = "total_score"
data = df[column].dropna()

# DESCRIPTIVE STATISTICS
mean_value = data.mean()
median_value = data.median()
mode_value = data.mode().iloc[0]
variance_value = data.var()
std_dev_value = data.std()
range_value = data.max() - data.min()

print("\nDescriptive Statistical Analysis")
print("----------------------")
print("Mean:", mean_value)
print("Median:", median_value)
print("Mode:", mode_value)
print("Variance:", variance_value)
print("Standard Deviation:", std_dev_value)
print("Range:", range_value)

# HISTOGRAM
plt.figure()
plt.hist(data, bins=10)
plt.title("Histogram of " + column)
plt.xlabel(column)
plt.ylabel("Frequency")
plt.show()

# BOX PLOT (for Oulier Detection)
plt.figure()
plt.boxplot(data)
plt.title("Boxplot of " + column)
plt.ylabel(column)
plt.show()

# DENSITY DISTRIBUTION
plt.figure()
density = stats.gaussian_kde(data)
x_vals = np.linspace(data.min(), data.max(), 200)
plt.plot(x_vals, density(x_vals))
plt.title("Density Distribution of " + column)
plt.xlabel(column)
plt.ylabel("Density")
plt.show()

# SKEWNESS VISUALIZATION
skewness_value = skew(data)
print("\nSkewness:", skewness_value)
if skewness_value > 0:
    print("Interpretation: Data is Positively Skewed (Right-skewed)")
elif skewness_value < 0:
    print("Interpretation: Data is Negatively Skewed (Left-skewed)")
else:
    print("Interpretation: Data is Symmetrical")

plt.figure()
plt.hist(data, bins=10, density=True)

x = np.linspace(data.min(), data.max(), 200)
plt.plot(x, norm.pdf(x, data.mean(), data.std()))

plt.title("Skewness Visualization - " + column)
plt.xlabel(column)
plt.ylabel("Density")
plt.show()

# KURTOSIS (STANDARDIZED)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.values.reshape(-1, 1)).flatten()

kurt_value = kurtosis(scaled_data)
print("\nKurtosis (Standardized Data):", kurt_value)

if kurt_value > 0:
    print("Interpretation: Leptokurtic Distribution (Heavy tails)")
elif kurt_value < 0:
    print("Interpretation: Platykurtic Distribution (Light tails)")
else:
    print("Interpretation: Mesokurtic Distribution (Normal)")

plt.figure()
plt.hist(scaled_data, bins=10, density=True)

x = np.linspace(min(scaled_data), max(scaled_data), 200)
plt.plot(x, norm.pdf(x, 0, 1))

plt.title("Kurtosis Visualization (Standardized)")
plt.xlabel("Standardized Scores")
plt.ylabel("Density")
plt.show()

print("\nInferential Statistical Analysis")
print("-" * 50)
# Independent T-Test
# Compare total_score between High vs Low attendance groups
# Create two groups based on median attendance
median_attendance = df['attendance_percentage'].median()

high_attendance = df[df['attendance_percentage'] >= median_attendance]['total_score']
low_attendance = df[df['attendance_percentage'] < median_attendance]['total_score']

t_stat, p_value = stats.ttest_ind(high_attendance, low_attendance)

print("Independent T-Test Results:")
print("T-statistic:", t_stat)
print("P-value:", p_value)

if p_value < 0.05:
    print("T-test shows significant difference between attendance groups.")
else:
    print("T-test shows no significant difference.")
print("-" * 50)

# ANOVA
# Compare total_score across different grades
anova_model = smf.ols('total_score ~ C(grade)', data=df).fit()
anova_table = sm.stats.anova_lm(anova_model, typ=2)

print("ANOVA Results:")
print(anova_table)
print("-" * 50)

plt.figure()
df.boxplot(column='total_score', by='grade')
plt.title("Total Score by Grade")
plt.suptitle("")  # Removes automatic title
plt.xlabel("Grade")
plt.ylabel("Total Score")
plt.show()


# Pearson Correlation Test
# Relationship between study hours and total score
corr_coef, corr_p = stats.pearsonr(df['weekly_self_study_hours'], df['total_score'])

print("Pearson Correlation Test:")
print("Correlation Coefficient:", corr_coef)
print("P-value:", corr_p)
print("-" * 50)

print("\nPredictive Statistical Analysis")
print("-" * 50)
# Linear Regression
# Predict total_score using study hours + attendance + participation
# -----------------------------

X = df[['weekly_self_study_hours', 'attendance_percentage', 'class_participation']]
y = df['total_score']

X = sm.add_constant(X)  # Add intercept
model = sm.OLS(y, X).fit()

print("Linear Regression Summary:")
print(model.summary())