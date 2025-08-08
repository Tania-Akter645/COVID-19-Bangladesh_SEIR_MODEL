# Author Tania Akter Rahima
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from preprocessing import load_and_clean_data
from exploratory_analysis import seir_model

# Load cleaned data
df = load_and_clean_data('data/COVID-19-Bangladesh.csv')

# Initial number of people
# I0 = Initial number of infectious people.
# R0 = Initial number of recovered people.
# E0 = Initial number of exposed (infected but not yet infectious) individuals.
# S0 = Initial number of susceptible people.
N = 165_000_000  # approx population
I0 = df['new_confirmed'].iloc[0]
E0 = 0
R0 = 0
S0 = N - I0 - E0 - R0

# Best-fit parameters from curve fitting
beta = 0.6  #Transmission rate
sigma = 1/5.2 # Rate
gamma = 1/10  # Recovery rate

# Time points (60 days for prediction)
t = np.linspace(0, 60, 61)

# Integrate SEIR equations
y0 = S0, E0, I0, R0
ret = odeint(seir_model, y0, t, args=(beta, sigma, gamma))
S, E, I, R = ret.T  #ret.T transposes the result so we can unpack each column (S, E, I, R) into separate arrays.


# Plot the result
plt.figure(figsize=(10, 6))
plt.plot(t, I, 'r-', label='Predicted Infected')
plt.plot(t, E, 'y--', label='Exposed')
plt.plot(t, R, 'g--', label='Recovered')
plt.plot(t, S, 'b--', label='Susceptible')
plt.xlabel('Days')
plt.ylabel('Population')
plt.title('COVID-19 SEIR Model Simulation (Next 60 Days)')
plt.legend()
plt.tight_layout()

import os
os.makedirs('report/figures', exist_ok=True)
plt.savefig('report/figures/seir_simulation.png') # save figure
plt.show()
