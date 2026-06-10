import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
import pandas as pd
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter

h_planck = 6.62607015e-34 # units of J*s
c = 299792458.0 # units of m/s
epsilon0 = 8.8541878188e-12 # units of F/m

csv_file = "Differential_Polarizabilities.csv"
df = pd.read_csv(csv_file)
print(df.columns)
conv = 1 / c / h_planck / epsilon0 / 1e6 * 1e4

frequency = (c / df.iloc[:, 0] * 1e9 / 1e12).to_numpy()  # Units of THz
alpha0      = (df.iloc[:, 1] * conv).to_numpy() # Units of MHz/I = MHz/(mW/um^2)
alpha1      = (df.iloc[:, 2] * conv).to_numpy() # Units of MHz/I = MHz/(mW/um^2)
alpha2      = (df.iloc[:, 3] * conv).to_numpy() # Units of MHz/I = MHz/(mW/um^2)


a0_pos = np.where(alpha0 > 0, alpha0, np.nan)
a0_neg = np.where(alpha0 < 0, np.abs(alpha0), np.nan)
a1_pos = np.where(alpha1 > 0, alpha1, np.nan)
a1_neg = np.where(alpha1 < 0, np.abs(alpha1), np.nan)
a2_pos = np.where(alpha2 > 0, alpha2, np.nan)
a2_neg = np.where(alpha2 < 0, np.abs(alpha2), np.nan)


fig3, axes3 = plt.subplots(figsize=(10,8))
#axes3.scatter(wavelength, alpha0, s=12, color='green',label=r'Scalar $\alpha^{(0)}$')
axes3.plot(frequency, a0_pos, linestyle='-', color='blue',label=r'$+$ Scalar $\alpha^{(0)}$')
axes3.plot(frequency, a0_neg, linestyle='--', color='blue',label=r'$-$ Scalar $\alpha^{(0)}$')
#axes3.scatter(wavelength, alpha1, s=12, color='orange',label=r'Vector $\alpha^{(1)}$')
axes3.plot(frequency, a1_pos, linestyle='-', color='red',label=r'$+$ Vector $\alpha^{(1)}$')
axes3.plot(frequency, a1_neg, linestyle='--', color='red',label=r'$-$ Vector $\alpha^{(1)}$')
#axes3.scatter(wavelength, alpha2, s=12, color='purple',label=r'Tensor $\alpha^{(2)}$')
axes3.plot(frequency, a2_pos, linestyle='-', color='green',label=r'$+$ Tensor $\alpha^{(2)}$')
axes3.plot(frequency, a2_neg, linestyle='--', color='green',label=r'$-$ Tensor $\alpha^{(2)}$')
axes3.axhline(y=0.0,color='blue',linestyle='--')
axes3.set_xlabel(r"Optical Frequency $\nu$ (THz)")
axes3.set_ylabel(r"Differential Polarizabilities (MHz/(mW/(100$\mu$m)$^2$))")
axes3.set_title("Scalar, Vector, and Tensor Differential Polarizabilities")
axes3.set_yscale("log")
axes3.legend()
axes3.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
axes3.ticklabel_format(style="plain", axis="x")
current_ticks = len(axes3.get_yticks())
axes3.yaxis.set_major_locator(ticker.LogLocator(base=10.0,numticks=15))
plt.tight_layout()
plt.grid(True, which='both',ls="--",alpha=0.5)
plt.show()
