# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 16:42:48 2024

@author: mikek
"""
#key references:
#https://physics.stackexchange.com/questions/760425/tov-equation-and-the-equation-of-state-solving-numerically
#https://colab.research.google.com/drive/1yMD2j3Y6TcsykCI59YWiW9WAMW-SPf12#scrollTo=9zySYOTJqhgM
import numpy as np
import matplotlib.pyplot as plt

# Define EOS functions
def EOS_p2erc(p, K=100., Gamma=2.):
    """
    Equation of state (EOS)
    Given pressure, return energy density, rest-mass density and sound speed
    """
    ene = (p / K)**(1. / Gamma) + p / (Gamma - 1.)
    rho = (p / K)**(1. / Gamma)
    cs2 = K * Gamma * (Gamma - 1) / (Gamma - 1 + K * Gamma * rho**(Gamma - 1)) * rho**(Gamma - 1)
    return ene, rho, cs2

def EOS_r2pe(rho, K=100., Gamma=2.):
    """
    Equation of state (EOS)
    Given rest-mass density, return energy density and pressure
    """
    p = K * rho**Gamma
    e = rho + p / (Gamma - 1.)
    return p, e

# Conversion factors
LCGS = 1.476701332464468e+05  # Length scale (in cm)
M_sun = 1.989e30  # Solar mass (in kg)
km_to_m = 1e3  # Conversion factor from km to meters

# TOV Equations
def TOV(t, y):
    """
    Tolmann-Oppenheimer-Volkhoff equations
    d/dt y(t) = R.H.S. 
    """
    r = t
    m = y[0]  # Mass of a sphere of radius r
    p = y[1]  # Pressure
    ene, _, _ = EOS_p2erc(p) 
    dy = np.empty_like(y)
    dy[0] = 4 * np.pi * ene * r**2                               
    dy[1] = -(ene + p) * (m + 4 * np.pi * r**3 * p) / (r * (r - 2 * m))
    return dy

# Event function to stop integration when pressure drops to zero
def found_radius(t, y, pfloor=0.):
    """
    Event function: Stop the ODE integration when pressure goes below pfloor
    """
    return (y[1] - pfloor) <= 0.

def solve_ode_euler(t, y0, dydt_fun, stop_event=None, verbose=False):
    """
    Euler algorithm 
    Returns the last time and state vector.
    """
    N = len(t)
    dt = np.diff(t)[0]  # assume a uniformly spaced t array
    y = np.array(y0)
    for i in range(N):
        yprev = np.copy(y)  # store previous for returning pre-event data
        y += dt * dydt_fun(t[i], y)
        if verbose:
            print(t[i], y)
        if stop_event:
            if bool(stop_event(t[i], y)):
                print("Event reached.")
                return t[i - 1], yprev  # Return the last valid time and state
    if stop_event:
        print("No event reached.")
    return t[-1], y  # Return final time and state if no event

# Initial conditions
def set_initial_conditions(rho, rmin):
    """
    Utility routine to set initial data, given rho0
    """
    p, e = EOS_r2pe(rho)  # EOS to get pressure and energy density
    m = 4. / 3. * np.pi * rho * rmin**3  # Mass from rest-mass density
    return [m, p]


#%%

# In# Increase resolution and extend the range for central densities
rmin, rmax = 1e-6, 5000.  # Further extend the radius range to capture spiral
N = 500000  # Even higher resolution for radius integration
rspan = np.linspace(rmin, rmax, N)

# Extend and refine central density range
rhospan = np.linspace(0.6e-4, 5e-2, 2000)  # Much wider range and finer steps

R = []
M = []

for rho0 in rhospan:
    # Solve TOV for rho0
    sol0 = set_initial_conditions(rho0, rmin)
    t, sol = solve_ode_euler(rspan, sol0, TOV, stop_event=found_radius)
    R.append(t)  # Add radius when pressure drops to zero
    M.append(sol[0])  # Add total mass

M = np.array(M)
R = np.array(R)

# Convert radius to km for plotting
R_km = R * LCGS * 1e-5

# Plot the extended mass-radius curve, including spiral
fig = plt.figure(figsize=(10, 6))
ax = plt.subplot(111)
plt.title('Mass-Radius Diagram Showing Spiral for Neutron Stars')
plt.plot(R_km, M, 'b-', label='Mass-Radius Curve', alpha=0.7)

# Highlight the maximum mass point
immax = np.argmax(M)
Mmax = M[immax]
Rmax = R_km[immax]
plt.plot(Rmax, Mmax, 'ro', label='Maximum Mass Point')


ax.legend()
plt.xlabel('$R$ $(km)$')
plt.ylabel('$M$ $(M_\odot)$')
plt.grid(True)
plt.show()

#%%
def solve_ode_euler(t, y0, dydt_fun, stop_event=None, verbose=False):
    """
    Euler method to solve the ODEs. Collect results at each step.
    """
    N = len(t)
    dt = np.diff(t)[0]  # Assume uniformly spaced time steps
    y = np.zeros((N, len(y0)))  # Store solution at each timestep
    y[0] = y0  # Set initial conditions
    R = []  # Radius array (in km)
    M = []  # Mass array (in solar masses)
    p = []   # Pressure array (in dyn/cm²)
    
    for i in range(1, N):
        y[i] = y[i-1] + dt * dydt_fun(t[i-1], y[i-1])  # Euler update
        
        # Store results
        R.append(t[i] * LCGS * 1e-5)  # Radius in km
        M.append(y[i][0] / M_sun)  # Mass in solar masses
        p.append(y[i][1])  # Pressure in dyn/cm²

        # Check for event (pressure goes below threshold)
        if stop_event and stop_event(t[i], y[i]):
            print("Event reached: pressure dropped to zero")
            break

    return np.array(R), np.array(M), np.array(p)  # Return all three arrays


#%%
# Parameters
rmin, rmax = 1e-6, 20.  # Small radius to avoid singularity at r=0
N = 1000  # Increase resolution
rspan = np.linspace(rmin, rmax, N)

# Build and solve models for different central densities
rho0 = 1.28e-3  # Central rest-mass density (example)
m0, p0 = set_initial_conditions(rho0, rmin)

# Solve the TOV equations
R, M, p = solve_ode_euler(rspan, [m0, p0], TOV, stop_event=found_radius, verbose=True)

# Plotting the results
km = LCGS * 1e-5  # Conversion factor for radius to km

# Plot Pressure vs Radius
plt.figure(figsize=(10, 6))
plt.plot(R, p, label="Pressure Profile")
plt.xlabel("Radius (km)")
plt.ylabel("Pressure (dyn/cm²)")
plt.title("Pressure vs Radius")
plt.grid(True)
plt.legend()
plt.show()

# Plot Mass vs Radius
plt.figure(figsize=(10, 6))
plt.plot(R, M, label="Mass Profile")
plt.xlabel("Radius (km)")
plt.ylabel("Mass (M⊙)")
plt.title("Mass vs Radius")
plt.grid(True)
plt.legend()
plt.show()



# Print final values
print(f"Final radius: {R[-1]} km")
print(f"Final mass: {M[-1]} M⊙")
print(f"Final pressure: {p[-1]} dyn/cm²")
