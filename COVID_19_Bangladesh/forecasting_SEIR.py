# Author Tania Akter Rahima
# forecasting SEIR model
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from datetime import timedelta
from exploratory_analysis import seir_model



# I Load the dataset which was collected from kaggle
df = pd.read_csv("data/COVID-19-Bangladesh.csv")
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df = df.sort_values("date")

# Initial values from last row of data

# I0 = Initial number of infectious people.
# R0 = Initial number of recovered people.
# D0 = Initial number of deaths.
# S0 = Initial number of susceptible people.

I0 = df['new_confirmed'].iloc[-1]
R0 = df['total_recovered'].iloc[-1]
D0 = df['total_deaths'].iloc[-1]
S0 = 165_000_000 - I0 - R0 - D0  # Bangladesh population approx
N = S0 + I0 + R0 + D0

# SEIR parameters
beta = 0.6 #Transmission rate
sigma = 1/5.2 # Rate
gamma = 1/14 # Recovery rate

# Timeframe for forecasting (next 30 days)
forecast_days = 30
t = np.linspace(0, forecast_days, forecast_days)

# Run SEIR model
# y is the current state: S,E,I,R
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

def seir_model(y, t, N, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]
ret = odeint(seir_model, [S0, 0, I0, R0], t, args=(N, beta, sigma, gamma))
S, E, I, R = ret.T

# Forecasted dates
last_date = df['date'].iloc[-1]
forecast_dates = [last_date + timedelta(days=int(x)) for x in t]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(forecast_dates, I, label="Forecasted Infected", color='r')
plt.plot(forecast_dates, R, label="Forecasted Recovered", color='g')
plt.plot(forecast_dates, [D0]*len(forecast_dates), label="Current Deaths (constant)", color='k', linestyle='--')
plt.title("30-Day Forecast of COVID-19 in Bangladesh (SEIR Model)")
plt.xlabel("Date")
plt.ylabel("Population")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Save the figure to the report file
plt.savefig("report/figures/forecasting_seir.png")
plt.show()
