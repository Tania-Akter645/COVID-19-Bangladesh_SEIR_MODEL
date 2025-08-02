# Author: Tania Akter Rahima
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from preprocessing import load_and_clean_data

# Load data
df = load_and_clean_data('data/COVID-19-Bangladesh.csv')

# Initial population
N = 165000000  # Bangladesh approx population
I0 = df['daily_confirmed'].iloc[0]
E0 = I0 * 1.5  # estimate: exposed are more than infected
R0 = 0
S0 = N - I0 - E0 - R0

# Time points (days)
t = np.arange(0, 160, 1)

# SEIR Model
def seir_model(y, t, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

# Parameters
beta = 0.4     # transmission rate
sigma = 1/5.2  # incubation period
gamma = 1/10   # recovery rate

# Initial condition vector
y0 = [S0, E0, I0, R0]

# Integrate ODEs
result = odeint(seir_model, y0, t, args=(beta, sigma, gamma))
S, E, I, R = result.T

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Days')
plt.ylabel('Population')
plt.title('SEIR Model Simulation for COVID-19 in Bangladesh')
plt.legend()
plt.tight_layout()

# Save plot
import os
os.makedirs('report/figures', exist_ok=True)
plt.savefig('report/figures/seir_simulation.png')
plt.show()
