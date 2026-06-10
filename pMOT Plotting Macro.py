import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
import pandas as pd
from matplotlib.ticker import ScalarFormatter

h_planck = 6.62607015e-34 # units of J*s
c = 299792458.0 # units of m/s
epsilon0 = 8.8541878188e-12 # units of F/m

#csv_file = "DataROI_perIntensity.csv"
#csv_file = "June9_Updated.csv"
csv_file = "June10_5p_32_Data.csv"
df = pd.read_csv(csv_file)
print(df.columns)
df_mF3 = df[df.iloc[:, 1] == 3].copy()
df_mF3 = df_mF3.sort_values(by=df.columns[0])
conv = 1 / c / h_planck / epsilon0 / 1e6 * 1e3

frequency   = (c / df_mF3.iloc[:, 0] * 1e9 / 1e12).to_numpy()  # Units of THz
alpha0      = (df_mF3.iloc[:, 2] * conv).to_numpy() # Units of MHz/I = MHz/(W/cm^2)
alpha1      = (df_mF3.iloc[:, 3] * conv).to_numpy() # Units of MHz/I = MHz/(W/cm^2)
alpha2      = (df_mF3.iloc[:, 4] * conv).to_numpy() # Units of MHz/I = MHz/(W/cm^2)
scalarTerm  = (df_mF3.iloc[:, 5]).to_numpy()
vectorTerm  = (df_mF3.iloc[:, 6]).to_numpy()
tensorTerm  = (df_mF3.iloc[:, 7]).to_numpy()
energyShift = (df_mF3.iloc[:, 8]).to_numpy()


a0_pos = np.where(alpha0 > 0, alpha0, np.nan)
a0_neg = np.where(alpha0 < 0, np.abs(alpha0), np.nan)
a1_pos = np.where(alpha1 > 0, alpha1, np.nan)
a1_neg = np.where(alpha1 < 0, np.abs(alpha1), np.nan)
a2_pos = np.where(alpha2 > 0, alpha2, np.nan)
a2_neg = np.where(alpha2 < 0, np.abs(alpha2), np.nan)

fig1, axes1 = plt.subplots(2, 2, figsize=(10, 8))
fig1.suptitle('Terms of the Energy Shift Rb-87 $5p_{3/2}$ $F=3$ $m_F=3$', fontsize=16, fontweight='bold')

axes1[0,0].scatter(frequency, energyShift, s=12, color='green', label='Energy Shift')
axes1[0,0].axhline(y=0.0, color='blue', linestyle='--')
axes1[0,0].plot(frequency, energyShift, linestyle=':', color='red')         # Optional: Connect with a dashed line
axes1[0,0].set_xlabel('Wavelength (nm)')
axes1[0,0].set_ylabel('Total Energy Shift (eV)')
axes1[0,0].set_title('Total Energy Shift (eV)')
axes1[0,1].scatter(frequency, scalarTerm, s=12, color='green', label='Energy Shift')
axes1[0,1].axhline(y=0.0, color='blue', linestyle='--')
axes1[0,1].plot(frequency, scalarTerm, linestyle=':', color='red')         # Optional: Connect with a dashed line
axes1[0,1].set_xlabel('Wavelength (nm)')
axes1[0,1].set_ylabel('Scalar Term Energy Shift (eV)')
axes1[0,1].set_title('Scalar Term Contribution to Energy Shift')
axes1[1,0].scatter(frequency, vectorTerm, s=12, color='green', label='Energy Shift')
axes1[1,0].axhline(y=0.0, color='blue', linestyle='--')
axes1[1,0].plot(frequency, vectorTerm, linestyle=':', color='red')         # Optional: Connect with a dashed line
axes1[1,0].set_xlabel('Wavelength (nm)')
axes1[1,0].set_ylabel('Vector Term Energy Shift (eV)')
axes1[1,0].set_title('Vector Term Contribution to Energy Shift')
axes1[1,1].scatter(frequency, tensorTerm, s=12, color='green', label='Energy Shift')
axes1[1,1].axhline(y=0.0, color='blue', linestyle='--')
axes1[1,1].plot(frequency, tensorTerm, linestyle=':', color='red')         # Optional: Connect with a dashed line
axes1[1,1].set_xlabel('Wavelength (nm)')
axes1[1,1].set_ylabel('Tensor Term Energy Shift (eV)')
axes1[1,1].set_title('Tensor Term Contribution to Energy Shift')

plt.tight_layout()
#plt.savefig("plot1test.svg",bbox_inches="tight")
#plt.show()

fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))
fig2.suptitle(r'pMOT: Rb-87 $5p_{3/2}$ $F=3$ $m_F=3$', fontsize=16, fontweight='bold')

axes2[0,0].scatter(frequency, energyShift, s=12, color='green', label='Energy Shift') # Plot individual points
axes2[0,0].axhline(y=0.0, color='blue', linestyle='--')
axes2[0,0].plot(frequency, energyShift, linestyle=':', color='red')         # Optional: Connect with a dashed line
axes2[0,0].set_xlabel('Wavelength (nm)')
axes2[0,0].set_ylabel(r'Energy Shift $\Delta E$ (eV)')
axes2[0,0].set_title('Total Energy Shift')
axes2[0,0].legend()

axes2[0,1].scatter(frequency, alpha0, s=12, color='green', label=r'$\alpha^{(0))}$') # Plot individual points
axes2[0,1].axhline(y=0.0, color='blue', linestyle='--')
axes2[0,1].plot(frequency, alpha0, linestyle=':', color='red')         # Optional: Connect with a dashed line
axes2[0,1].set_xlabel('Wavelength (nm)')
axes2[0,1].set_ylabel(r'Scalar Polarizability $\alpha^{(0)}$ (MHz/(W/m^2))')
axes2[0,1].set_title(r'Scalar Polarizability $\alpha^{(0)}$')
axes2[0,1].legend()

axes2[1,0].scatter(frequency, alpha1, s=12, color='green', label=r'$\alpha^{(1)}$') # Plot individual points
axes2[1,0].axhline(y=0.0, color='blue', linestyle='--')
axes2[1,0].plot(frequency, alpha1, linestyle=':', color='red')         # Optional: Connect with a dashed line
axes2[1,0].set_xlabel('Wavelength (nm)')
axes2[1,0].set_ylabel(r'Vector Polarizability $\alpha^{(1)}$ (MHz/(W/m^2))')
axes2[1,0].set_title(r'Vector Polarizability $\alpha^{(1)}$')
axes2[1,0].legend()

axes2[1,1].scatter(frequency, alpha2, s=12, color='green', label=r'$\alpha^{(2)}$') # Plot individual points
axes2[1,1].axhline(y=0.0, color='blue', linestyle='--')
axes2[1,1].plot(frequency, alpha2, linestyle=':', color='red')         # Optional: Connect with a dashed line
axes2[1,1].set_xlabel('Wavelength (nm)')
axes2[1,1].set_ylabel(r'Tensor Polarizability $\alpha^{(2)}$ (MHz/(W/m^2))')
axes2[1,1].set_title(r'Tensor Polarizability $\alpha^{(2)}$')
axes2[1,1].legend()

plt.tight_layout()
#plt.savefig("plot2test.svg",bbox_inches="tight")
#plt.show()


fig3, axes3 = plt.subplots(figsize=(10,8))
#axes3.scatter(wavelength, alpha0, s=12, color='green',label=r'Scalar $\alpha^{(0)}$')
axes3.plot(frequency, a0_pos, linestyle='-', color='green',label=r'$+$ Scalar $\alpha^{(0)}$')
axes3.plot(frequency, a0_neg, linestyle='--', color='green',label=r'$-$ Scalar $\alpha^{(0)}$')
#axes3.scatter(wavelength, alpha1, s=12, color='orange',label=r'Vector $\alpha^{(1)}$')
axes3.plot(frequency, a1_pos, linestyle='-', color='orange',label=r'$+$ Vector $\alpha^{(1)}$')
axes3.plot(frequency, a1_neg, linestyle='--', color='orange',label=r'$-$ Vector $\alpha^{(1)}$')
#axes3.scatter(wavelength, alpha2, s=12, color='purple',label=r'Tensor $\alpha^{(2)}$')
axes3.plot(frequency, a2_pos, linestyle='-', color='purple',label=r'$+$ Tensor $\alpha^{(2)}$')
axes3.plot(frequency, a2_neg, linestyle='--', color='purple',label=r'$-$ Tensor $\alpha^{(2)}$')
axes3.axhline(y=0.0,color='blue',linestyle='--')
axes3.set_xlabel(r"Optical Frequency $\nu$ (THz)")
axes3.set_ylabel("Polarizabilities (MHz/(W/cm^2))")
axes3.set_title("Scalar, Vector, and Tensor Polarizabilities - 5P_3/2 State")
axes3.set_yscale("log")
axes3.legend()
axes3.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
axes3.ticklabel_format(style="plain", axis="x")
plt.tight_layout()
plt.show()

# ------------------------------#
# ANIMATION CODE
# ------------------------------#

lambda_min = 1490
lambda_max = 1550
output_gif = "testACStarkShift2.gif"
mF_values = np.array([-3, -2, -1, 0, 1, 2, 3])
colors = ["black", "orange", "gold", "green", "blue", "purple", "red"]
min_bound = 1e-9
fps = 100 # 20    Frames per second

def get_decade_bound(yvals, min_bound=1e-9):
    max_abs = np.max(np.abs(yvals))
    if max_abs <= min_bound:
        return min_bound
    exponent = np.ceil(np.log10(max_abs))
    bound = 10.0**exponent
    return max(bound, min_bound)

df = pd.read_csv(csv_file)

# Keep only relevant columns
df = df[["Wavelength (nm)", "m_F", "Energy Shift (MHz/I)"]].copy()

# Make sure values are numeric
df["Wavelength (nm)"] = pd.to_numeric(df["Wavelength (nm)"], errors="coerce")
df["m_F"] = pd.to_numeric(df["m_F"], errors="coerce")
df["Energy Shift (MHz/I)"] = pd.to_numeric(df["Energy Shift (MHz/I)"], errors="coerce")

# Drop bad rows if any exist
df = df.dropna(subset=["Wavelength (nm)", "m_F", "Energy Shift (MHz/I)"])

# Convert mF to integer if appropriate
df["m_F"] = df["m_F"].astype(int)

# Filter wavelength range
df = df[
    (df["Wavelength (nm)"] >= lambda_min) &
    (df["Wavelength (nm)"] <= lambda_max)
].copy()

# Sort for clean animation
df = df.sort_values(["Wavelength (nm)", "m_F"])

# Pivot data into animation-friendly format
# Rows: wavelength, Columns: mF, Values: energy shift
pivot = df.pivot_table(
    index="Wavelength (nm)",
    columns="m_F",
    values="Energy Shift (MHz/I)",
    aggfunc="mean" )

# Ensure columns are ordered as mF = -3,...,+3
pivot = pivot.reindex(columns=mF_values)

# Drop wavelengths where not all mF values are available
pivot = pivot.dropna(subset=mF_values)

# Extract wavelength array and shift matrix
wavelengths = pivot.index.to_numpy()
shifts = pivot[mF_values].to_numpy()

print(f"Loaded {len(wavelengths)} wavelength frames.")
print(f"Wavelength range: {wavelengths[0]} nm to {wavelengths[-1]} nm")
print("First frame shifts:", shifts[0])
print("Last frame shifts:", shifts[-1])


# ============================================================
# Create animation
# ============================================================
x_positions = np.arange(len(mF_values))
state_labels = [
    rf"$|5p_{{3/2}};\,m_F={mF}\rangle$"
    for mF in mF_values
]
fig, ax = plt.subplots(figsize=(10, 6))

# Initial frame
y0 = shifts[0]
wl0 = wavelengths[0]

line, = ax.plot(
    x_positions,
    y0,
    linewidth=1.5,
    alpha=0.7,
    zorder=2
)

scatter = ax.scatter(
    x_positions,
    y0,
    s=180,
    c=colors,
    edgecolors="gray",
    linewidths=1.2,
    zorder=3
)

# Optional labels above each point
text_labels = []
for x, y, mF in zip(x_positions, y0, mF_values):
    txt = ax.text(
        x,
        y,
        rf"$m_F={mF}$",
        fontsize=10,
        ha="center",
        va="bottom"
    )
    text_labels.append(txt)

# Axes formatting
ax.set_xticks(x_positions)
ax.set_xticklabels(state_labels, rotation=25)
ax.set_xlabel(r"$|5p_{3/2};\,m_F\rangle$", fontsize=13)
ax.set_ylabel("AC Stark Energy Shift", fontsize=13)

initial_bound = get_decade_bound(y0, min_bound=min_bound)
ax.set_ylim(-initial_bound, initial_bound)

ax.grid(True, alpha=0.3)

# Scientific notation on y-axis
formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-2, 2))
ax.yaxis.set_major_formatter(formatter)

title = ax.set_title(
    rf"AC Stark Energy Shift vs. $m_F$ at $\lambda={wl0:.1f}$ nm "
    rf"$(y=\pm {initial_bound:.0e})$",
    fontsize=14
)

plt.tight_layout()

# Animation update function
def update(frame):
    wl = wavelengths[frame]
    yvals = shifts[frame, :]

    # Update line
    line.set_xdata(x_positions)
    line.set_ydata(yvals)

    # Update scatter
    scatter.set_offsets(np.column_stack([x_positions, yvals]))

    # Stepwise decade y-axis scaling
    bound = get_decade_bound(yvals, min_bound=min_bound)
    ax.set_ylim(-bound, bound)

    # Update point labels
    for txt, x, y, mF in zip(text_labels, x_positions, yvals, mF_values):
        txt.set_position((x, y))
        txt.set_text(rf"$m_F={mF}$")

    # Update title
    title.set_text(
        rf"AC Stark Energy Shift vs. $m_F$ at $\lambda={wl:.1f}$ nm "
        rf"$(y=\pm {bound:.0e})$"
    )

    return [line, scatter, title, *text_labels]

anim = FuncAnimation(
    fig,
    update,
    frames=len(wavelengths),
    interval=1000 / fps,
    blit=False
)
#anim.save(output_gif, writer=PillowWriter(fps=fps))
