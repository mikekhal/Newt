# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 17:48:37 2024

@author: mi423
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

h_bar = 1.0545718e-34  # Reduced Planck constant (J·s)
m_e = 9.10938356e-31  # Electron mass (kg)
c = 3.0e8  # Speed of light (m/s)

def trapezoidal(f, a, b, n):
    g = 0
    x = np.linspace(a,b,n+1)
    y = f(x)

    h = abs(b-a)/float(n)
    g += h*0.5*((y[0]+y[-1]))
    for i in range(1,n):

        g += h*y[i]
    return g
def integrand_elec(u):
    return (u**2 + 1)**0.5 * u**2 * m_e**4 * c**5 * (np.pi)**-2 * h_bar **-3

def integrand_pressure(u):
    return (u**2 + 1)**-0.5 * u**4 * m_e**4 * c**5 * (np.pi)**-2 * h_bar **-3/3


def energy_density(k_F):
    x = k_F / (m_e * c)
    integral = trapezoidal(integrand_elec, 0, x, 500)
    return integral

def pressure(k_F):
    x = k_F / (m_e * c)
    integral = trapezoidal(integrand_pressure, 0, x, 500)
    return integral
def idealmodel(k_F):
    return 3e-9 * energy_density(k_F)**(4/3)
# Compute energy density and pressure for a range of k_F values
k_F = np.linspace(0, 2 * m_e * c, 100)  # Example Fermi momentum (m^-1)

e_rho = np.array(energy_density(k_F))
p = np.array(pressure(k_F))

# Plot pressure vs energy density
plt.plot(e_rho, p)
plt.plot(e_rho,idealmodel(k_F))

plt.show()