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
epsilon0 = 8.8541878188e-12 # units of F/m
# --------------------------------------------------------------------
# USER PARAMETERS
intensity = 1  # Intensity of laser beam in W/m^2  Example: 1e6.  intensity=1 is for unit intensity (later edit in the actual beam value)
E_squared = 2 * intensity / c / epsilon0
pol = +1        # Convention here is that right-handed = +1 and left-handed = -1 (for the electric field amplitude cross product term)
theta = pi / 2  # This is the angle between the electric field \vec{E} and the quantization axis, taken to be \hat{z}
I_qn = 1.5      # Nuclear spin.  For Rubidium, I = 3/2.
# --------------------------------------------------------------------
# Calculate the shift of the 5S_1/2 F = 2 (ground) state
F_qn = 2        # Total hyperfine angular momentum quantum number (F=J+I)
J_qn = 0.5
w6j_vector = np.array( [ float(wigner_6j(1, 1, 1, F_qn, F_qn, 1)), float(wigner_6j(1, 1, 1, F_qn, F_qn, 2)), float(wigner_6j(1, 1, 1, F_qn, F_qn, 3)) ] )
w6j_tensor = np.array( [ float(wigner_6j(1, 1, 2, F_qn, F_qn, 1)), float(wigner_6j(1, 1, 2, F_qn, F_qn, 2)), float(wigner_6j(1, 1, 2, F_qn, F_qn, 3)) ] )
energy_initial = 0.0
# [5p_1/2, 6p_1/2, 7p_1/2, 8p_1/2, 5p_3/2, 6p_3/2, 7p_3/2, 8p_3/2]  for 5s_1/2 analysis
RME_elec = np.array( [4.231, 0.3236, 0.115, 0.060, 5.978, 0.5230, 0.202, 0.111] ) * (eCharge*a0) # 5s_1/2 analysis

# 0th row is F'=1, 1st row is F'=2, 2nd row is F'=3
# Order is:  5 6 7 8p_1/2, 5 6 7 8p_3/2
energies_hf = np.array( [ [ 12578.847444823, 23714.989980198, 27834.932008573, 29834.853168063, 12816.451840308, 23792.502999323, 27870.023368498, 29853.703896951 ], 
                        [ 12578.874744823, 23714.998823598, 27834.936006573, 29834.855310879, 12816.457075268, 23792.504715323, 27870.024150098, 29853.704315340 ],
                        [ np.nan, np.nan, np.nan, np.nan, 12816.465969808, 23792.507619323, 27870.025464998, 29853.705020895 ] ] ) * h_planck * c * 100 # units of Joules

# --------------------------------------------------------------------
# PLOTTING PARAMETERS - Plot from lambda = [min, max] (nm).
minLambda = 1529.1243  #400
maxLambda = 1529.4753  #1500
Npt = 17550 # Number of plotting points over interval
# --------------------------------------------------------------------
# Total AC Stark Energy Shift
def energy_shift(a_0, a_1, a_2, m_F):
    scalar_term = -a_0 * E_squared
    vector_term = -a_1 * pol * E_squared * m_F / F_qn
    tensor_term = -a_2 / 2.0 * (3 * E_squared * np.cos(theta)**2 - E_squared) * ((3 * m_F**2 - F_qn*(F_qn+1))/(F_qn*(2*F_qn - 1)))
    # return ( (scalar_term + vector_term + tensor_term) * 6.241509e18 )  # in units of eV
    return ( (scalar_term + vector_term + tensor_term) / h_planck / 1e6 )  # in units of MHz

# Scalar Polarizability
def scalar(F, w):
    scalar_temp_val = 0.0
    for i, row in enumerate(energies_hf): # 0, 1, 2
        for j, energy_hf in enumerate(row):  # 0, 1, 2, 3, 4, 5, 6, 7
            if np.isnan(energy_hf): continue
            if i == 0: 
                Fprime = 1
                if (j < 4): Jprime = 0.5
                elif (j >= 4): Jprime = 1.5
            elif i == 1: 
                Fprime = 2
                if (j < 4): Jprime = 0.5
                elif (j >= 4): Jprime = 1.5
            elif i == 2:
                Fprime = 3
                if (j < 4): Jprime = 0.5
                elif (j >= 4): Jprime = 1.5
            omega_FprimeF = 1/hbar*(energy_hf - energy_initial)
            RME_hf = (-1)**(Jprime + I_qn + F + 1) * math.sqrt((2*Fprime + 1)*(2*F + 1))*float(wigner_6j(Jprime, Fprime, I_qn, F, J_qn, 1)) * RME_elec[j]
            scalar_temp_val += ((2/3/hbar)*omega_FprimeF/(omega_FprimeF**2 - w**2) * RME_hf**2 )
    return scalar_temp_val

# Vector Polarizability
def vector(F, w):
    vector_temp_val = 0.0
    for i, row in enumerate(energies_hf): # 0, 1, 2
        for j, energy_hf in enumerate(row):  # 0, 1, 2, 3, 4, 5, 6, 7
            if np.isnan(energy_hf): continue
            if i == 0: 
                Fprime = 1
                if (j < 4): Jprime = 0.5
                elif (j >= 4): Jprime = 1.5
            elif i == 1:
                Fprime = 2
                if (j < 4): Jprime = 0.5
                elif (j >= 4): Jprime = 1.5
            elif i == 2:
                Fprime = 3
                if (j < 4): Jprime = 0.5
                elif (j >= 4): Jprime = 1.5
            #w6j = float(wigner_6j(1, 1, 1, F, F, Fprime))
            w6j = w6j_vector[(Fprime - 1)]
            omega_FprimeF = 1/hbar*(energy_hf - energy_initial)
            RME_hf = (-1)**(Jprime + I_qn + F + 1) * math.sqrt((2*Fprime + 1)*(2*F + 1))*float(wigner_6j(Jprime, Fprime, I_qn, F, J_qn, 1)) * RME_elec[j]
            vector_temp_val += ( (-1)**(F+Fprime+1) * math.sqrt(6.0*F*(2*F+1)/(F+1)) * w6j / hbar * omega_FprimeF/(omega_FprimeF**2 - w**2) * RME_hf**2 )
    return vector_temp_val

# Tensor Polarizability
def tensor(F, w):
    tensor_temp_val = 0.0
    for i, row in enumerate(energies_hf): # 0, 1, 2
        for j, energy_hf in enumerate(row):  # 0, 1, 2, 3, 4, 5, 6, 7
            if np.isnan(energy_hf): continue
            if i == 0: 
                Fprime = 1
                if (j < 4): Jprime = 0.5
                elif (j >= 4): Jprime = 1.5
            elif i == 1:
                Fprime = 2
                if (j < 4): Jprime = 0.5
                elif (j >= 4): Jprime = 1.5
            elif i == 2:
                Fprime = 3
                if (j < 4): Jprime = 0.5
                elif (j >= 4): Jprime = 1.5
            #w6j = float(wigner_6j(1, 1, 2, F, F, Fprime))
            w6j = w6j_tensor[(Fprime - 1)]
            omega_FprimeF = 1/hbar*(energy_hf - energy_initial)
            RME_hf = (-1)**(Jprime + I_qn + F + 1) * math.sqrt((2*Fprime + 1)*(2*F + 1))*float(wigner_6j(Jprime, Fprime, I_qn, F, J_qn, 1)) * RME_elec[j]
            tensor_temp_val += ( (-1)**(F+Fprime) * math.sqrt(40.0*F*(2*F+1)*(2*F-1)/3/(F+1)/(2*F+3)) * w6j / hbar * omega_FprimeF/(omega_FprimeF**2 - w**2) * RME_hf**2 )
    return tensor_temp_val
# --------------------------------------------------------------------
# SAVING TO CSV FILE
length = Npt
col_mF = []
col_wavelength = []
col_alpha0 = []
col_alpha1 = []
col_alpha2 = []
col_scalarTerm = []
col_vectorTerm = []
col_tensorTerm = []
col_energyShift = []

counter = 0
wavelengths = np.linspace(minLambda, maxLambda, Npt)
for real_lambda in wavelengths:
    if (counter % 1000 == 0): print(counter)
    counter += 1
    w = 2 * pi * c / (real_lambda*1e-9)
    alpha_0 = scalar(F_qn, w)
    alpha_1 = vector(F_qn, w)
    alpha_2 = tensor(F_qn, w)
    for m_F in [-2,-1,0,1,2]:
        #if (real_lambda % 1000 == 0): print(real_lambda)
        col_mF.append(m_F)
        col_wavelength.append(real_lambda)
        col_alpha0.append(alpha_0)
        col_alpha1.append(alpha_1)
        col_alpha2.append(alpha_2)
        col_energyShift.append(energy_shift(alpha_0, alpha_1, alpha_2, m_F))
        col_scalarTerm.append(energy_shift(alpha_0, 0, 0, m_F))
        col_vectorTerm.append(energy_shift(0, alpha_1, 0, m_F))
        col_tensorTerm.append(energy_shift(0, 0, alpha_2, m_F))

df = pd.DataFrame( {"Wavelength (nm)": col_wavelength, "m_F": col_mF, "Scalar Polarizability": col_alpha0, "Vector Polarizability": col_alpha1, "Tensor Polarizability": col_alpha2,
                   "Scalar Term (MHz/I)": col_scalarTerm, "Vector Term (MHz/I)": col_vectorTerm, "Tensor Term (MHz/I)": col_tensorTerm, "Energy Shift (MHz/I)": col_energyShift} )

df.to_csv("5s_12_Shift.csv", index=False)
print("CSV file created successfully.")