# --------------------------------------------------------------------
import math
import numpy as np
from sympy.physics.wigner import wigner_6j
import pandas as pd
# --------------------------------------------------------------------
# PHYSICAL CONSTANTS
eCharge = 1.60217663e-19 # e, electric charge, units of Coulombs
a0 = 5.29177210544e-11   # a_0, Bohr Radius, units of meters
pi = np.pi
hbar = 1.054571817e-34 # units of J*s
h_planck = 6.62607015e-34 # units of J*s
c = 299792458.0 # units of m/s
epsilon0 = 8.854e-12 # units of F/m
lande_g = 0.665  # For 5p_3/2 F=3 , Unitless
mu_B = 9.2740100657e-24 # Units of J/T
# --------------------------------------------------------------------
# USER PARAMETERS
intensity = 1  # Intensity of laser beam in W/m^2  Example: 1e6.  intensity=1 is for unit intensity (later edit in the actual beam value)
E_squared = 2 * intensity / c / epsilon0
pol = +1        # Convention here is that right-handed = +1 and left-handed = -1 (for the electric field amplitude cross product term)
theta = pi / 2  # This is the angle between the electric field \vec{E} and the quantization axis, taken to be \hat{z}
I_qn = 1.5      # Nuclear spin.  For Rubidium, I = 3/2.
# --------------------------------------------------------------------
# 0th row is F'=2, 1st row is F'=3, 2nd row is F'=4
# Order is: 5s_1/2, 4 5 6 7 8d_5/2, 4 5 6 7 8d_3/2
energies_hf = np.array( [
    [0.085493, 19355.204809, 25703.520802, 28689.390367, 30281.620216, 31222.453129, 
     19355.648275, 25700.536192, 28687.126657, 30280.112821, 31221.440087],
    [-1.0, 19355.203073, 25703.520038, 28689.390017, 30281.620010, 31222.453006,
     19355.650825, 25700.537667, 28687.127459, 30280.113283, 31221.440377],
    [-1.0, 19355.200929, 25703.519080, 28689.389580, 30281.619753, 31222.452852,
     -1.0, -1.0, -1.0, -1.0, -1.0] ] ) * 1.98645e-23 # units of Joules

F_qn = 3        # Total hyperfine angular momentum quantum number (F=J+I)
J_qn = F_qn - I_qn
energy_initial = 12816.551462 * 1.98645e-23 # E_nJF , Look up this value (using NIST Atomic Spectra Database) and report it in units of cm^-1 - converted to Joules here
RME_elec = np.array( [-6.003, -10.889, 1.80, 1.658, -1.118, -0.855, 3.630, -0.600, -0.553, 0.373, 0.285] ) * (eCharge*a0) # 5p_3/2 analysis
# [5s_1/2, 4 5 6 7 8d_5/2, 4 5 6 7 8d_3/2]  for 5p_3/2 analysis
# --------------------------------------------------------------------
lambda_1 = 1529.1245
lambda_2 = 1529.6006
omega_1 = c / (lambda_1*10**-9) * 2 * pi
omega_2 = c / (lambda_2*10**-9) * 2 * pi
# --------------------------------------------------------------------
# Vector Term
def alpha_vector(F, w):
    vector_temp_val = 0.0
    for i, row in enumerate(energies_hf): # 0, 1, 2
        for j, energy_hf in enumerate(row):  # 0, 1, 2, 3, 4, 5
            if energy_hf == -1.0: continue
            if i == 0: 
                Fprime = 2
                if j == 0: Jprime = 0.5
                else: Jprime = 2.5
            elif i == 1:
                Fprime = 3
                Jprime = 2.5
            elif i == 2:
                Fprime = 4
                Jprime = 2.5
            w6j = float(wigner_6j(1, 1, 1, F, F, Fprime))
            omega_FprimeF = 1/hbar*(energy_hf - energy_initial)
            RME_hf = (-1)**(Jprime + I_qn + F + 1) * math.sqrt((2*Fprime + 1)*(2*F + 1))*float(wigner_6j(Jprime, Fprime, I_qn, F, J_qn, 1)) * RME_elec[j]
            vector_temp_val += ( (-1)**(F+Fprime+1) * math.sqrt(6.0*F*(2*F+1)/(F+1)) * w6j / hbar * omega_FprimeF/(omega_FprimeF**2 - w**2) * RME_hf**2 )            
    return vector_temp_val

a_V1 = alpha_vector(3,omega_1)
a_V2 = alpha_vector(3,omega_2)
B_eff1 = -1 * a_V1 * E_squared / mu_B / lande_g / F_qn # Units of Tesla
B_eff2 = -1 * a_V2 * E_squared / mu_B / lande_g / F_qn # Units of Tesla
I_gradient1 = 1 / 10 / B_eff1 / 1e6 # Units of W/cm^3
I_gradient2 = 1 / 10 / B_eff2 / 1e6 # Units of W/cm^3
B_eff1 *= 10000 # Units of Gauss
B_eff2 *= 10000 # UNits of Gauss
B_total = B_eff1 + B_eff2
I_gradient_total = I_gradient1 + I_gradient2

print("Effective magnetic field below is intensity-independent.")
print(f"Effective Magnetic Field of Laser 1: {B_eff1} Gauss")
print(f"Effective Magnetic Field of Laser 2: {B_eff2} Gauss")
print(f"Intensity Gradient Required of Laser 1: {I_gradient1} W/cm^3")
print(f"Intensity Gradient Required of Laser 2: {I_gradient2} W/cm^3")
print(f"Effective Total Magnetic Field: {B_total} Gauss")
print(f"Total Necessary Intensity Gradient: {I_gradient_total} W/cm^3")
