# Author: Tania Akter Rahima
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from preprocessing import load_and_clean_data

# Load data
df = load_and_clean_data('data/COVID-19-Bangladesh.csv')

# Infection data to fit
infected_data = df['daily_confirmed'].values[:100]  # প্রথম 100 দিনের data

# Total population
N = 165000000
I0 = infected_data[0]
E0 = I0 * 2
R0 = 0
S0 = N - I0 - E0 - R0
y0 = [S0, E0, I0, R0]

# Time points
t = np.arange(0, len(infected_data), 1)

# SEIR Model
def seir_model(y, t, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

# Solve the SEIR equations
def solve_seir(params):
    beta, sigma, gamma = params
    ret = odeint(seir_model, y0, t, args=(beta, sigma, gamma))
    return ret[:, 2]  # Return only Infected

# Loss function (RMSE)
def loss(params):
    I_pred = solve_seir(params)
    return np.sqrt(np.mean((infected_data - I_pred)**2))

# Initial guess
params_initial = [0.5, 1/5.2, 1/10]

# Minimize loss
result = minimize(loss, params_initial, bounds=[(0.1, 1), (0.1, 1), (0.05, 0.5)])
beta_fit, sigma_fit, gamma_fit = result.x

print(f"Fitted Parameters:\nBeta = {beta_fit:.4f}, Sigma = {sigma_fit:.4f}, Gamma = {gamma_fit:.4f}")

# Simulate with best-fit
I_fitted = solve_seir([beta_fit, sigma_fit, gamma_fit])

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t, infected_data, 'o', label='Actual Infected')
plt.plot(t, I_fitted, '-', label='Fitted SEIR Infected')
plt.xlabel('Days')
plt.ylabel('Infected Cases')
plt.title('SEIR Model Curve Fitting to COVID-19 Data')
plt.legend()
plt.tight_layout()

# Save figure
import os
os.makedirs('report/figures', exist_ok=True)
plt.savefig('report/figures/seir_curve_fitting.png')
plt.show()
