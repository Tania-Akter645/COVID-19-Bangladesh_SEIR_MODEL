#Author=Tania Akter Rahima


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import load_and_clean_data

# Plot style
sns.set(style="darkgrid")

# Load cleaned data
df = load_and_clean_data('data/COVID-19-Bangladesh.csv')

# 1. Daily Confirmed Cases Trend
plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['daily_confirmed'], label='daily confirmed', color='blue')
plt.title('daily confirmed COVID-19 Cases in Bangladesh')
plt.xlabel('date')
plt.ylabel('Number of cases')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('report/figures/trend_plot.png')  # Save figure
plt.show()

# 2. Daily Deaths Trend
plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['daily_deaths'], label='daily Deaths', color='red')
plt.title('daily deaths due to COVID-19 in Bangladesh')
plt.xlabel('date')
plt.ylabel('Number of deaths')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('report/figures/daily_deaths.png')  # Save figure
plt.show()

#3. Total_Quarantine Trend
plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['total_quarantine'], label='total_quarantine', color='green')
plt.title('daily quarantine due to COVID-19 in Bangladesh')
plt.xlabel('date')
plt.ylabel('Number of daily quarantine')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('report/figures/daily_quarantine.png')  # Save figure to the report file
plt.show()

# 4. Correlation Heatmap
plt.figure(figsize=(6, 5))
corr = df[['daily_confirmed', 'daily_deaths', 'daily_recovered', 'daily_quarantine']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('report/figures/correlation_heatmap.png')  # Save figure to the report file
plt.show()
# SEIR Model function (for simulation) using differentiation equation
def seir_model(y, t, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I / 165_000_000  # Bangladesh approx population
    dEdt = beta * S * I / 165_000_000 - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

## y is the current state: S,E,I,R
# Where:
# S = Susceptible
# E = Exposed
# I = Infectious
# R = Recovered
# t is the time (automatically passed by odeint during integration)
# N = Total population
# beta = Transmission rate
#sigma = Rate at which exposed individuals become infectious (E → I).This represents the inverse of the incubation period.
#gamma = Recovery rate (I → R).This represents the inverse of the infectious period.
