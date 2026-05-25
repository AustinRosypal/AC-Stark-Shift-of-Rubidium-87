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
c = 299792458.0
epsilon0 = 8.854e-12 # (F/m)
# --------------------------------------------------------------------
# USER PARAMETERS
analyze_5p32 = True
analyze_5s12 = False
intensity = 10e5  # Intensity of laser beam in W/m^2
pol = +1  # Convention here is that right-handed = +1 and left-handed = -1 (for the electric field amplitude cross product term)
theta = pi / 2  # This is the angle between the electric field \vec{E} and the quantization axis, taken to be \hat{z}
I_qn = 1.5        # Nuclear spin.  For Rubidium, I = 3/2.
lambda_of_interest = 630.05  #456  1528

# 0th row is F'=2, 1st row is F'=3, 2nd row is F'=4
# Order is 5s_1/2, 4 5 6 7 8d_5/2
energies_hf = np.array( [
    [0.085493, 19355.204809, 25703.520802, 28689.390367, 30281.620216, 31222.453129],
    [0.0, 19355.203073, 25703.520038, 28689.390017, 30281.620010, 31222.453006],
    [0.0, 19355.200929, 25703.519080, 28689.389580, 30281.619753, 31222.452852]
] )
energies_hf *= 1.98645e-23

if (analyze_5p32):
    F_qn = 3        # Total hyperfine angular momentum quantum number (F=J+I)
    Fprime = 4   # For matrix elements
    w = 1.231950011e15     # Angular frequency of experimental laser light (currently corresponds to 1529nm light, bridging 5p_3/2 with 4d_5/2)
    nu = w / 2 / pi   # Since w = 2*pi*nu and it's good to express frequency as nu
    energy_initial = 12816.551462 * 1.98645e-23 #2.54594e-19 # E_nJF , Look up this value (using NIST Atomic Spectra Database) and report it in the correct units of cm^-1
    #energy_final = np.array( [0, 19355.203, 25703.498, 28689.390, 30281.620, 31222.453] ) # 5p_3/2 analysis.  Units of cm^-1
    RME_elec = np.array( [-6.003, -10.889, 1.80, 1.658, -1.118, -0.855] ) # 5p_3/2 analysis
    # [5s_1/2, 4d_5/2, 5d_5/2, 6d_5/2, 7d_5/2, 8d_5/2]  for 5p_3/2 analysis
    #energy_final *= 1.98645e-23
    RME_elec *= (eCharge*a0)
elif (analyze_5s12):
    F_qn = 2        # Total hyperfine angular momentum quantum number (F=J+I)
    Fprime = 3   # For matrix elements
    w = 2.414192e15
    nu = w / 2 / pi   # Since w = 2*pi*nu and it's good to express frequency as nu
    energy_initial = 0.
    #energy_final = np.array( [12816.545, 23792.591] ) # 5s_1/2 analysis
    RME_elec = np.array( [6.003, 0.5230] ) # 5s_1/2 analysis
    # [5p_3/2, 6p_3/2] for 5s_1/2 analysis
    #energy_final *= 1.98645e-23
    RME_elec *= (eCharge*a0)

# PLOTTING PARAMETERS - Plot from lambda = [min, max] (nm).
minLambda = 400  #400
maxLambda = 2000 #1500
Npt = 16000 # Number of plotting points over interval
# --------------------------------------------------------------------
# COMPUTATIONAL DEFINITIONS (automated, no user input here)

alpha_0 = 0.0  # scalar polarizability
alpha_1 = 0.0  # vector polarizability
alpha_2 = 0.0  # tensor polarizability
scalar_term = 0.0
vector_term = 0.0
tensor_term = 0.0
delta_E = 0.0  # Energy shift due to the AC Stark shift
J_qn = F_qn - I_qn
Jprime = Fprime - I_qn
w6j = 0.0
RME_hf = 0.0
omega_FprimeF = 0.0 # Difference between energies
E_squared = 2 * intensity / c / epsilon0
# --------------------------------------------------------------------
# Total AC Stark Energy Shift
def energy_shift(a_0, a_1, a_2, m_F):
    scalar_term = -a_0 * E_squared
    vector_term = -a_1 * pol * E_squared * m_F / F_qn
    tensor_term = -a_2 / 2.0 * (3 * E_squared * np.cos(theta)**2 - E_squared) * ((3 * m_F**2 - F_qn*(F_qn+1))/(F_qn*(2*F_qn - 1)))
    return ( (scalar_term + vector_term + tensor_term) * 6.241509e18 )  # in units of eV

# Scalar Part
def scalar(F, Fprime, w):
    scalar_temp_val = 0.0
    for i, row in enumerate(energies_hf): # 0, 1, 2
        for j, column in enumerate(row):  # 0, 1, 2, 3, 4, 5
            if energies_hf[i][j] == 0: continue
            if i == 0: 
                Fprime = 2 # 5s_1/2 contribution
                if j == 0: Jprime = 0.5
                else: Jprime = 2.5
            elif i == 1: 
                Fprime = 3
                Jprime = 2.5
            elif i == 2:
                Fprime = 4
                Jprime = 2.5
            omega_FprimeF = 1/hbar*(energies_hf[i][j] - energy_initial)
            RME_hf = (-1)**(Jprime + I_qn + F + 1) * math.sqrt((2*Fprime + 1)*(2*F + 1))*float(wigner_6j(Jprime, Fprime, I_qn, F, J_qn, 1)) * RME_elec[j]
            scalar_temp_val += ((2/3/hbar)*omega_FprimeF/(omega_FprimeF**2 - w**2) * RME_hf**2 )
    return scalar_temp_val

# Vector Part
def vector(F, Fprime, w):
    vector_temp_val = 0.0
    w6j = float(wigner_6j(1, 1, 1, F, F, Fprime))
    for i, row in enumerate(energies_hf): # 0, 1, 2
        for j, column in enumerate(row):  # 0, 1, 2, 3, 4, 5
            if energies_hf[i][j] == 0: continue
            if i == 0: 
                Fprime = 2 # 5s_1/2 contribution
                if j == 0: Jprime = 0.5
                else: Jprime = 2.5
            elif i == 1:
                Fprime = 3
                Jprime = 2.5
            elif i == 2:
                Fprime = 4
                Jprime = 2.5
            omega_FprimeF = 1/hbar*(energies_hf[i][j] - energy_initial)
            RME_hf = (-1)**(Jprime + I_qn + F + 1) * math.sqrt((2*Fprime + 1)*(2*F + 1))*float(wigner_6j(Jprime, Fprime, I_qn, F, J_qn, 1)) * RME_elec[j]
            vector_temp_val += ( (-1)**(F+Fprime+1) * math.sqrt(6.0*F*(2*F+1)/(F+1)) * w6j / hbar * omega_FprimeF/(omega_FprimeF**2 - w**2) * RME_hf**2 )
    return vector_temp_val

# Tensor Part
def tensor(F, Fprime, w):
    tensor_temp_val = 0.0
    w6j = float(wigner_6j(1, 1, 2, F, F, Fprime))
    for i, row in enumerate(energies_hf): # 0, 1, 2
        for j, column in enumerate(row):  # 0, 1, 2, 3, 4, 5
            if energies_hf[i][j] == 0: continue
            if i == 0: 
                Fprime = 2 # 5s_1/2 contribution
                if j == 0: Jprime = 0.5
                else: Jprime = 2.5
            elif i == 1:
                Fprime = 3
                Jprime = 2.5
            elif i == 2:
                Fprime = 4
                Jprime = 2.5
            omega_FprimeF = 1/hbar*(energies_hf[i][j] - energy_initial)
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

for lambda_ in np.arange(0, Npt) + minLambda:
    for m_F in [-3,-2,-1,0,1,2,3]:
        real_lambda = (maxLambda - minLambda)/(Npt)*(lambda_ - minLambda) + minLambda
        if (real_lambda % 100 == 0): print(real_lambda)
        nu = c / (real_lambda*10**-9)
        w = 2 * pi * nu
        col_mF.append(m_F)
        col_wavelength.append(real_lambda)
        alpha_0 = scalar(F_qn,Fprime,w)
        alpha_1 = vector(F_qn, Fprime, w)
        alpha_2 = tensor(F_qn, Fprime, w)
        col_alpha0.append(alpha_0)
        col_alpha1.append(alpha_1)
        col_alpha2.append(alpha_2)
        col_energyShift.append(energy_shift(alpha_0, alpha_1, alpha_2, m_F))
        col_scalarTerm.append(energy_shift(alpha_0, 0, 0, m_F))
        col_vectorTerm.append(energy_shift(0, alpha_1, 0, m_F))
        col_tensorTerm.append(energy_shift(0, 0, alpha_2, m_F))

df = pd.DataFrame( {"Wavelength (nm)": col_wavelength, "m_F": col_mF, "Scalar Polarizability": col_alpha0, "Vector Polarizability": col_alpha1, "Tensor Polarizability": col_alpha2,
                   "Scalar Term": col_scalarTerm, "Vector Term": col_vectorTerm, "Tensor Term": col_tensorTerm, "Energy Shift": col_energyShift} )
df.to_csv("pMOT_Data_May20.csv", index=False)
print("CSV file created successfully.")