# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 18:51:19 2024

@author: mikek
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
alpha = 1.0  # Example value for alpha
beta = 1.0   # Example value for beta
gamma = 5 / 3  # Polytropic index
p0 = 1.0      # Initial dimensionless pressure
h = 0.01      # Step size for Runge-Kutta

# Define the system of ODEs
def polytrope_system(y, r, alpha, beta, gamma):
    p, m = y  # y[0] = p, y[1] = m
    if p <= 0:  # Stop when pressure is non-positive
        return [0, 0]
    dpdr = -alpha * (p**(1/gamma)) * m / r**2 if r > 0 else 0  # Avoid division by zero
    dmdr = beta * r**2 * (p**(1/gamma))
    return [dpdr, dmdr]

# Runge-Kutta 4th-order method
def rk4_step(y, r, h, alpha, beta, gamma):
    k1 = h * np.array(polytrope_system(y, r, alpha, beta, gamma))
    k2 = h * np.array(polytrope_system(y + 0.5 * k1, r + 0.5 * h, alpha, beta, gamma))
    k3 = h * np.array(polytrope_system(y + 0.5 * k2, r + 0.5 * h, alpha, beta, gamma))
    k4 = h * np.array(polytrope_system(y + k3, r + h, alpha, beta, gamma))
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

# Initialize variables
r_values = [1e-5]  # Start from a small radius to avoid division by zero
y_values = [[p0, 0]]  # Initial conditions: p(0)=p0, m(0)=0

# Integration loop
while True:
    r_current = r_values[-1]
    y_current = y_values[-1]

    # Stop if pressure becomes non-positive
    if y_current[0] <= 0:
        break

    # Perform a Runge-Kutta step
    y_next = rk4_step(y_current, r_current, h, alpha, beta, gamma)

    # Append the results
    r_values.append(r_current + h)
    y_values.append(y_next)

# Extract pressure and mass
r_values = np.array(r_values)
y_values = np.array(y_values)
pressure = y_values[:, 0]
mass = y_values[:, 1]

# Determine the star's radius and mass
R_star = r_values[pressure > 0][-1]
M_star = mass[pressure > 0][-1]

# Plot results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(r_values[pressure > 0], pressure[pressure > 0], label='Pressure')
plt.xlabel('Radius (r)')
plt.ylabel('Pressure (dimensionless)')
plt.title('Pressure Profile')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(r_values[pressure > 0], mass[pressure > 0], label='Mass')
plt.xlabel('Radius (r)')
plt.ylabel('Mass (dimensionless)')
plt.title('Mass Profile')
plt.legend()

plt.tight_layout()
plt.show()

# Print results
print(f"Star's Radius (R): {R_star:.3f}")
print(f"Star's Mass (M): {M_star:.3f}")
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import numpy as np

def polytrope(y, r, alpha, beta, gamma):
    """
    Computes the derivatives dp/dr and dm/dr.
    """
    p, m = y  # y[0] = p, y[1] = m
    if p <= 0:  # Stop integration if pressure is non-positive
        return np.array([0, 0])  # Ensure this is a NumPy array
    dpdr = -alpha * (p**(1 / gamma)) * m / r**2 if r > 0 else 0  # Avoid division by zero
    dmdr = beta * r**2 * (p**(1 / gamma))
    return np.array([dpdr, dmdr])  # Return as NumPy array


# Runge-Kutta 4th Order Method
def rk4_step(func, y, r, h, alpha, beta, gamma):
    """
    Performs a single RK4 step.
    """
    k1 = h * func(y, r, alpha, beta, gamma)
    k2 = h * func(y + 0.5 * k1, r + 0.5 * h, alpha, beta, gamma)
    k3 = h * func(y + 0.5 * k2, r + 0.5 * h, alpha, beta, gamma)
    k4 = h * func(y + k3, r + h, alpha, beta, gamma)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Integration function
def integrate_polytrope_rk4(alpha, beta, gamma, p0, r_max, dr):
    """
    Integrates the polytrope equations using RK4.
    """
    # Initial conditions
    r = 1e-5  # Start just above r=0 to avoid singularity
    y = np.array([p0, 0])  # Initial values: p(0) = p0, m(0) = 0
    radii = [r]
    pressures = [y[0]]
    masses = [y[1]]

    # RK4 loop
    while r < r_max and y[0] > 0:  # Stop when pressure drops to zero
        y = rk4_step(polytrope, y, r, dr, alpha, beta, gamma)
        r += dr
        radii.append(r)
        pressures.append(y[0])
        masses.append(y[1])

    # Return the radius where pressure becomes zero and the corresponding mass
    R = radii[-1] if y[0] <= 0 else r
    M = masses[-1]
    return R, M, np.array(radii), np.array(pressures), np.array(masses)

# Constants for relativistic and non-relativistic cases
alpha_rel = 1.473  # km
beta_rel = 52.46  # /km^3
gamma_rel = 4 / 3
p0_rel = 1e-15  # Central pressure

alpha_nonrel = 0.05  # km
beta_nonrel = 0.005924  # /km^3
gamma_nonrel = 5 / 3
p0_nonrel = 1e-16  # Central pressure

# Relativistic Case
R_rel, M_rel, radii_rel, pressures_rel, masses_rel = integrate_polytrope_rk4(
    alpha_rel, beta_rel, gamma_rel, p0_rel, 20000, dr=0.1
)
print(f"Relativistic Case (kF >> me):")
print(f"Central Pressure: {p0_rel:.1e}, Radius: {R_rel:.2f} km, Mass: {M_rel:.4f} Msun")

# Non-Relativistic Case
R_nonrel, M_nonrel, radii_nonrel, pressures_nonrel, masses_nonrel = integrate_polytrope_rk4(
    alpha_nonrel, beta_nonrel, gamma_nonrel, p0_nonrel, 20000, dr=0.1
)
print(f"Non-Relativistic Case (kF << me):")
print(f"Central Pressure: {p0_nonrel:.1e}, Radius: {R_nonrel:.2f} km, Mass: {M_nonrel:.4f} Msun")
