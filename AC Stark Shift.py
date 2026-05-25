# Here, I will calculate the energy shift between states 5p3/2 and 4d5/2 due to the AC Stark Shift
# The unperturbed frequency (energy) difference for 5d_5/2 is: 775.9787 nm = 386.3410916 THz

# When I log in, I have to do: `Ctr+Shift+P`, "select Python Interpreter" --> Choose C:\Users\ajros\OneDrive\Documents\Python Projects\venv\Scripts\python.exe.  Then the code compiles!

# --------------------------------------------------------------------
import matplotlib.pyplot as plt
import math
import numpy as np
from sympy.physics.wigner import wigner_6j
from matplotlib.animation import FuncAnimation, PillowWriter
import os
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

save_dir = "May15Plots"
os.makedirs(save_dir, exist_ok=True)

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
maxLambda = 1800 #1500
Npt = 28000 # Number of plotting points over interval
# --------------------------------------------------------------------
# COMPUTATIONAL DEFINITIONS (automated, no user input here)
scalar_val_arr = [0]*len(energies_hf[1]) # [0]*len(energy_final)
vector_val_arr = [0]*len(energies_hf[1])
tensor_val_arr = [0]*len(energies_hf[1])
delta_E_arr = [0]*len(energies_hf[1])
contrib_arr = [0]*len(energies_hf[1])
scalar_val_arr_ind = [0]*len(energies_hf[1])
vector_val_arr_ind = [0]*len(energies_hf[1])
tensor_val_arr_ind = [0]*len(energies_hf[1])
delta_E_arr_ind = [0]*len(energies_hf[1])
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
    # print(f"Scalar Term: {scalar_term} \nVector Term: {vector_term} \nTensor Term: {tensor_term}")
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
            scalar_val_arr[i] = scalar_temp_val
            scalar_val_arr_ind[i] = ((2/3/hbar)*omega_FprimeF/(omega_FprimeF**2 - w**2) * RME_hf**2 )
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
            vector_val_arr[i] = vector_temp_val
            vector_val_arr_ind[i] = ( (-1)**(F+Fprime+1) * math.sqrt(6.0*F*(2*F+1)/(F+1)) * w6j / hbar * omega_FprimeF/(omega_FprimeF**2 - w**2) * RME_hf**2 )
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
            tensor_val_arr[i] = tensor_temp_val
            tensor_val_arr_ind[i] = ( (-1)**(F+Fprime) * math.sqrt(40.0*F*(2*F+1)*(2*F-1)/3/(F+1)/(2*F+3)) * w6j / hbar * omega_FprimeF/(omega_FprimeF**2 - w**2) * RME_hf**2 )
    return tensor_temp_val
# --------------------------------------------------------------------
# PLOTTING
length = Npt
arr_eShift = [0]*length
arr_wavelength = [0]*length
arr_alpha0 = [0]*length
arr_alpha1 = [0]*length
arr_alpha2 = [0]*length
arr_scalarTerm = [0]*length
arr_vectorTerm = [0]*length
arr_tensorTerm = [0]*length

for m_F in [-3,-2,-1,0,1,2,3]:
    for lambda_ in np.arange(0, Npt) + minLambda:
        real_lambda = (maxLambda - minLambda)/(Npt)*(lambda_-minLambda) + minLambda
        if (real_lambda % 100 == 0): print(real_lambda)
        nu = c / (real_lambda*10**-9)
        w = 2 * pi * nu
        alpha_0 = scalar(F_qn,Fprime,w)
        alpha_1 = vector(F_qn,Fprime,w)
        alpha_2 = tensor(F_qn,Fprime,w)
        delta_E = energy_shift(alpha_0, alpha_1, alpha_2, m_F)
        arr_scalarTerm[lambda_-minLambda] = energy_shift(alpha_0, 0, 0, m_F)
        arr_vectorTerm[lambda_-minLambda] = energy_shift(0, alpha_1, 0, m_F)
        arr_tensorTerm[lambda_-minLambda] = energy_shift(0, 0, alpha_2, m_F)
        arr_wavelength[lambda_-minLambda] = real_lambda
        arr_alpha0[lambda_-minLambda] = alpha_0
        arr_alpha1[lambda_-minLambda] = alpha_1
        arr_alpha2[lambda_-minLambda] = alpha_2
        arr_eShift[lambda_-minLambda] = delta_E
        #print(f"The energy shift for the given configuration of omega = {w:.3e} is: {delta_E}")

        if (real_lambda == lambda_of_interest):
            for i, value in enumerate(scalar_val_arr):
                contrib_arr[i] = i + 1
                delta_E_arr[i] = energy_shift(scalar_val_arr[i], vector_val_arr[i], tensor_val_arr[i], m_F)
                delta_E_arr_ind[i] = energy_shift(scalar_val_arr_ind[i], vector_val_arr_ind[i], tensor_val_arr_ind[i], m_F)
            fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
            fig2.suptitle(rf'Higher Order Corrections to Energy Shift: $\lambda = {lambda_of_interest}$ nm', fontsize=16, fontweight='bold')
            axes2[0].scatter(contrib_arr, delta_E_arr, s=12, color='green')
            axes2[0].plot(contrib_arr, delta_E_arr, marker='x')
            axes2[0].set_title('Summed Higher Order Corrections to Energy Shift')
            axes2[0].set_xlabel('Number of Terms Included')
            axes2[0].set_ylabel('Energy Shift (eV)')
            axes2[0].set_yscale('symlog')
            axes2[1].scatter(contrib_arr, delta_E_arr_ind, s=12, color='green')
            axes2[1].plot(contrib_arr, delta_E_arr_ind, marker='x')
            axes2[1].set_title(r'Individual Higher Order Corrections to Energy Shift')
            axes2[1].set_xlabel('Individual Transition Shifts')
            axes2[1].set_ylabel('Energy Shift (eV)')
            axes2[1].set_yscale('symlog')
            plt.tight_layout()
            filename = "TestLambdaOfInterest630nm.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()
            #plt.show()

    fig3, axes3 = plt.subplots(2, 2, figsize=(10, 8))
    fig3.suptitle('Terms of the Energy Shift Rb-87 $5p_{3/2}$ $F=3$ $m_F=3$', fontsize=16, fontweight='bold')

    axes3[0,0].scatter(arr_wavelength, arr_eShift, s=12, color='green', label='Energy Shift')
    axes3[0,0].axhline(y=0.0, color='blue', linestyle='--')
    axes3[0,0].plot(arr_wavelength, arr_eShift, linestyle=':', color='red')         # Optional: Connect with a dashed line
    axes3[0,0].set_xlabel('Wavelength (nm)')
    axes3[0,0].set_ylabel('Total Energy Shift (eV)')
    axes3[0,0].set_title('Total Energy Shift (eV)')
    axes3[0,1].scatter(arr_wavelength, arr_scalarTerm, s=12, color='green', label='Energy Shift')
    axes3[0,1].axhline(y=0.0, color='blue', linestyle='--')
    axes3[0,1].plot(arr_wavelength, arr_scalarTerm, linestyle=':', color='red')         # Optional: Connect with a dashed line
    axes3[0,1].set_xlabel('Wavelength (nm)')
    axes3[0,1].set_ylabel('Scalar Term Energy Shift (eV)')
    axes3[0,1].set_title('Scalar Term Contribution to Energy Shift')
    axes3[1,0].scatter(arr_wavelength, arr_vectorTerm, s=12, color='green', label='Energy Shift')
    axes3[1,0].axhline(y=0.0, color='blue', linestyle='--')
    axes3[1,0].plot(arr_wavelength, arr_vectorTerm, linestyle=':', color='red')         # Optional: Connect with a dashed line
    axes3[1,0].set_xlabel('Wavelength (nm)')
    axes3[1,0].set_ylabel('Vector Term Energy Shift (eV)')
    axes3[1,0].set_title('Vector Term Contribution to Energy Shift')
    axes3[1,1].scatter(arr_wavelength, arr_tensorTerm, s=12, color='green', label='Energy Shift')
    axes3[1,1].axhline(y=0.0, color='blue', linestyle='--')
    axes3[1,1].plot(arr_wavelength, arr_tensorTerm, linestyle=':', color='red')         # Optional: Connect with a dashed line
    axes3[1,1].set_xlabel('Wavelength (nm)')
    axes3[1,1].set_ylabel('Tensor Term Energy Shift (eV)')
    axes3[1,1].set_title('Tensor Term Contribution to Energy Shift')

    plt.tight_layout()
    filename = "EnergyShiftingof5p32Terms.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    #plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(r'pMOT: Rb-87 $5p_{3/2}$ $F=3$ $m_F=3$', fontsize=16, fontweight='bold')

    axes[0,0].scatter(arr_wavelength, arr_eShift, s=12, color='green', label='Energy Shift') # Plot individual points
    axes[0,0].axhline(y=0.0, color='blue', linestyle='--')
    axes[0,0].plot(arr_wavelength, arr_eShift, linestyle=':', color='red')         # Optional: Connect with a dashed line
    axes[0,0].set_xlabel('Wavelength (nm)')
    axes[0,0].set_ylabel(r'Energy Shift $\Delta E$ (eV)')
    axes[0,0].set_title('Total Energy Shift')
    axes[0,0].legend()

    axes[0,1].scatter(arr_wavelength, arr_alpha0, s=12, color='green', label=r'$\alpha^{(0))}$') # Plot individual points
    axes[0,1].axhline(y=0.0, color='blue', linestyle='--')
    axes[0,1].plot(arr_wavelength, arr_alpha0, linestyle=':', color='red')         # Optional: Connect with a dashed line
    axes[0,1].set_xlabel('Wavelength (nm)')
    axes[0,1].set_ylabel(r'Scalar Polarizability $\alpha^{(0)}$ ($m^2C/V$)')
    axes[0,1].set_title(r'Scalar Polarizability $\alpha^{(0)}$')
    axes[0,1].legend()

    axes[1,0].scatter(arr_wavelength, arr_alpha1, s=12, color='green', label=r'$\alpha^{(1)}$') # Plot individual points
    axes[1,0].axhline(y=0.0, color='blue', linestyle='--')
    axes[1,0].plot(arr_wavelength, arr_alpha1, linestyle=':', color='red')         # Optional: Connect with a dashed line
    axes[1,0].set_xlabel('Wavelength (nm)')
    axes[1,0].set_ylabel(r'Vector Polarizability $\alpha^{(1)}$ ($m^2C/V$)')
    axes[1,0].set_title(r'Vector Polarizability $\alpha^{(1)}$')
    axes[1,0].legend()

    axes[1,1].scatter(arr_wavelength, arr_alpha2, s=12, color='green', label=r'$\alpha^{(2)}$') # Plot individual points
    axes[1,1].axhline(y=0.0, color='blue', linestyle='--')
    axes[1,1].plot(arr_wavelength, arr_alpha2, linestyle=':', color='red')         # Optional: Connect with a dashed line
    axes[1,1].set_xlabel('Wavelength (nm)')
    axes[1,1].set_ylabel(r'Tensor Polarizability $\alpha^{(2)}$ ($m^2C/V$)')
    axes[1,1].set_title(r'Tensor Polarizability $\alpha^{(2)}$')
    axes[1,1].legend()

    plt.tight_layout()
    filename = "EnergyShiftingof5p32Pols.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    #plt.show()

# print(f"The energy shift for the given configuration of F = {F_qn}, m_F = {m_F}, omega = {w:.3e} is: {delta_E}")
# print(f"Scalar Pol: {alpha_0} \nVector Pol: {alpha_1} \nTensor Pol: {alpha_2}")




# ------------------------------------------------------------------#
# Energy Shift for all m_F Values for a Single Wavelength Value
# ------------------------------------------------------------------#
# # Example list of m_F values
# mF_values = np.array([-3, -2, -1, 0, 1, 2, 3])
# wavelength_nm = 1000.0
# nu = c / (wavelength_nm*10**-9)
# w = 2 * pi * nu
# energy_shifts = []
# for mF in mF_values:
#     alpha_0 = scalar(F_qn,Fprime,w)
#     alpha_1 = vector(F_qn,Fprime,w)
#     alpha_2 = tensor(F_qn,Fprime,w)
#     delta_E = energy_shift(alpha_0, alpha_1, alpha_2, mF)
#     print(rf"The energy shift for m_F={mF} is: {delta_E}")
#     energy_shifts.append(delta_E)

# energy_shifts = np.array(energy_shifts)
# # Create x-axis labels for each state
# state_labels = [rf"$|5p_{{3/2}}; m_F={mF}\rangle$" for mF in mF_values]
# # Choose colors for each point
# colors = ['red', 'orange', 'gold', 'green', 'blue', 'purple', 'black']
# # Plot
# plt.figure(figsize=(10, 6))
# # Scatter points
# plt.scatter(
#     range(len(mF_values)),
#     energy_shifts,
#     s=180,              # point size
#     c=colors,
#     edgecolors='gray',
#     linewidths=1.2,
#     zorder=3
# )
# # Optional thin line connecting the points
# plt.plot(range(len(mF_values)), energy_shifts, linewidth=1.0, alpha=0.6, zorder=2)
# # Label each point with its m_F value
# for i, (mF, shift) in enumerate(zip(mF_values, energy_shifts)):
#     plt.text(i, shift, f"  {mF}", fontsize=11, va='bottom')
# # Axes and ticks
# plt.xticks(range(len(mF_values)), state_labels, rotation=25)
# plt.xlabel(r"$|5p_{3/2};\, m_F\rangle$", fontsize=13)
# plt.ylabel("AC Stark Energy Shift", fontsize=13)
# plt.title(f"AC Stark Energy Shift vs. $m_F$ at {wavelength_nm:.1f} nm", fontsize=14)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()




# ------------------------------#
# ANIMATION CODE
# ------------------------------#
mF_values = np.array([-3, -2, -1, 0, 1, 2, 3])
wavelengths = np.arange(400, 1801, 1)   # 400 nm to 1800 nm in 1 nm steps
x_positions = np.arange(len(mF_values))
state_labels = [rf"$|5p_{{3/2}};\, m_F={mF}\rangle$" for mF in mF_values]
colors = ['red', 'orange', 'gold', 'green', 'blue', 'purple', 'black']

shifts = np.zeros((len(wavelengths), len(mF_values)))

def get_bound(yvals, min_bound):
    max_abs = np.max(np.abs(yvals))
    # If everything is zero or tiny, keep at least the minimum bound
    if max_abs <= min_bound:
        return min_bound
    # Find the next power of ten above max_abs
    exponent = np.ceil(np.log10(max_abs))
    bound = 10.0**exponent
    # Never go below the chosen minimum bound
    return max(bound, min_bound)

for i, wl in enumerate(wavelengths):
    for j, mF in enumerate(mF_values):
        tempWavelength = wavelengths[i]
        nu = c / (tempWavelength*10**-9)
        w = 2 * pi * nu
        mF = mF_values[j]
        #print(rf"Wavelength: {tempWavelength} and mF: {mF}")
        alpha_0 = scalar(F_qn,Fprime,w)
        alpha_1 = vector(F_qn,Fprime,w)
        alpha_2 = tensor(F_qn,Fprime,w)
        delta_E = energy_shift(alpha_0, alpha_1, alpha_2, mF)
        shifts[i,j] = delta_E

ymin = -1e-9 #np.min(shifts)
ymax = 1e-9 #np.max(shifts)
padding = 0.08 * (ymax - ymin) if ymax > ymin else 1.0
ymin_plot = ymin - padding
ymax_plot = ymax + padding

fig, ax = plt.subplots(figsize=(10, 6))
initial_y = shifts[0]
line, = ax.plot(x_positions, initial_y, linewidth=1.5, alpha=0.7)

scatter = ax.scatter(
    x_positions,
    initial_y,
    s=180,
    c=colors,
    edgecolors='gray',
    linewidths=1.2,
    zorder=3
)

text_labels = []
for x, y, mF in zip(x_positions, initial_y, mF_values):
    txt = ax.text(x, y, f"  {mF}", fontsize=11, va='bottom')
    text_labels.append(txt)
ax.set_xticks(x_positions)
ax.set_xticklabels(state_labels, rotation=25)
ax.set_xlabel(r"$|5p_{3/2};\, m_F\rangle$", fontsize=13)
ax.set_ylabel("AC Stark Energy Shift", fontsize=13)
ax.set_ylim(ymin_plot, ymax_plot)
ax.grid(True, alpha=0.3)

title = ax.set_title(
    f"AC Stark Energy Shift vs. $m_F$ at {wavelengths[0]} nm",
    fontsize=14
)

plt.tight_layout()

# Animation update function
def update(frame):
    wl = wavelengths[frame]
    yvals = shifts[frame]
    line.set_data(x_positions, yvals)
    scatter.set_offsets(np.column_stack((x_positions, yvals)))

    for txt, x, y, mF in zip(text_labels, x_positions, yvals, mF_values):
        txt.set_position((x, y))
        txt.set_text(f"  {mF}")
    title.set_text(f"AC Stark Energy Shift vs. $m_F$ at {wl} nm")

    bound = get_bound(yvals, 1e-9)
    ax.set_ylim(-bound, bound)

    return [line, scatter, title, *text_labels]

anim = FuncAnimation(
    fig,
    update,
    frames=len(wavelengths),
    interval=40,
    blit=False
)

anim.save("ac_stark_shift_scan_May15.gif", writer=PillowWriter(fps=20))











# ------------------------------------------------------------------------------------------------------------------------

# # Here, I will calculate the energy shift between 5p3/2 and a single other level 

# import matplotlib.pyplot as plt
# import math
# import numpy as np
# from sympy.physics.wigner import wigner_6j

# F_qn = 3        # Total hyperfine angular momentum quantum number (F=J+I)
# m_F = 3         # Zeeman sublevels of F, projection of F along the quantization axis, if you will.
# Fprime = 4   # For matrix elements
# m_Fprime = m_F + 1  # For matrix elements
# intensity = 10e7  # intensity of laser beam in W/m^2
# w = 3.893408545e14      # Frequency of experimental laser light (currently corresponds to 770nm light)
# energy_initial = 2.54594e-19 # E_nJF , Look up this value (using NIST Atomic Spectra Database) and report it in the correct units
# energy_final = 6.40050225e-19
# pol = +1  # Convention here is that right-handed = +1 and left-handed = -1 (for the electric field amplitude cross product term)
# theta = np.pi / 2  # This is the angle between the electric field \vec{E} and the quantization axis, taken to be \hat{z}

# alpha_0 = 0.0  # scalar polarizability
# alpha_1 = 0.0  # vector polarizability
# alpha_2 = 0.0  # tensor polarizability
# scalar_term = 0.0
# vector_term = 0.0
# tensor_term = 0.0
# delta_E = 0.0  # Energy shift due to the AC Stark shift
# I_qn = 1.5        # Nuclear spin.  For Rubidium, I = 3/2.
# J_qn = F_qn - I_qn
# Jprime = Fprime - I_qn
# w6j = 0.0
# omega_FprimeF = 0.0 # Difference between energies
# RME_d = 0.0
# hbar = 1.054571817e-34
# c = 299792458.0
# epsilon0 = 8.854e-12 # (F/m)
# pi = np.pi
# eCharge = 1.60217663e-19 # e, electric charge, units of Coulombs
# a0 = 5.29177210544e-11   # a_0, Bohr Radius, units of meters
# E_squared = 2 * intensity / c / epsilon0

# # Total AC Stark Energy Shift
# # Note this is inaccurate.  The projections along z must be incorporated, as well as the magnitude of E_plus, etc.
# def energy_shift(a_0, a_1, a_2):
#     scalar_term = -a_0 * E_squared
#     vector_term = -a_1 * pol * E_squared * m_F / F_qn
#     tensor_term = -a_2 / 2.0 * (3 * E_squared * np.cos(theta)**2 - E_squared) * ((3 * m_F**2 - F_qn*(F_qn+1))/(F_qn*(2*F_qn - 1)))
#     print(f"Scalar Term: {scalar_term} \nVector Term: {vector_term} \nTensor Term: {tensor_term}")
#     return (scalar_term + vector_term + tensor_term)

# # The rank-2 polarizability tensor \alpha can be decomposed into irreducible scalar, vector, and tensor components.
# # Scalar Part - this is still incorrect because I don't know what the terms in this thing are.
# def scalar(F, Fprime, w):
#     omega_FprimeF = 1/hbar*(energy_final - energy_initial)
#     RME_d = 1.80 * eCharge * a0
#     return ( (2/3/hbar)*omega_FprimeF/(omega_FprimeF**2 - w**2)*RME_d**2 )

# # Vector Part
# def vector(F, Fprime, w):
#     w6j = float(wigner_6j(1, 1, 1, F, F, Fprime))
#     return ( (alpha_0*3/2) * (-1)**(F+Fprime+1) * math.sqrt(6.0*F*(2*F+1)/(F+1)) * w6j )

# # Tensor Part
# def tensor(F, Fprime, w):
#     w6j = float(wigner_6j(1, 1, 2, F, F, Fprime))
#     return ( (alpha_0*3/2) * (-1)**(F+Fprime) * math.sqrt(40.0*F*(2*F+1)*(2*F-1)/3/(F+1)/(2*F+3)) * w6j )

# alpha_0 = scalar(F_qn,Fprime,w)
# alpha_1 = vector(F_qn,Fprime,w)
# alpha_2 = tensor(F_qn,Fprime,w)
# delta_E = energy_shift(alpha_0, alpha_1, alpha_2)

# print(f"The energy shift for the given configuration of F = {F_qn}, m_F = {m_F}, omega = {w:.3e}, connecting to F' = {Fprime}, m_F' = {m_Fprime} is: {delta_E}")
# print(f"Scalar Pol: {alpha_0} \nVector Pol: {alpha_1} \nTensor Pol: {alpha_2}")