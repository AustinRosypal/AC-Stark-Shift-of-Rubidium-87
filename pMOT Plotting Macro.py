import matplotlib.pyplot as plt
import numpy as np
from sympy.physics.wigner import wigner_6j
from matplotlib.animation import FuncAnimation, PillowWriter
import pandas as pd
from matplotlib.ticker import ScalarFormatter

csv_file = "pMOT_Data_May20.csv"
df = pd.read_csv(csv_file)
print(df.columns)
counter = -1
length = len(df)
wavelength = [0]*length
mF = [0]*length
alpha0 = [0]*length
alpha1 = [0]*length
alpha2 = [0]*length
scalarTerm = [0]*length
vectorTerm = [0]*length
tensorTerm = [0]*length
energyShift = [0]*length

for index, row in df.iterrows():
    counter += 1
    wavelength[counter] = row.iloc[0]
    mF[counter] = row.iloc[1]
    alpha0[counter] = row.iloc[2]
    alpha1[counter] = row.iloc[3]
    alpha2[counter] = row.iloc[4]
    scalarTerm[counter] = row.iloc[5]
    vectorTerm[counter] = row.iloc[6]
    tensorTerm[counter] = row.iloc[7]
    energyShift[counter] = row.iloc[8]
    if (counter % 10000 == 0): print(counter) # print(wavelength, mF, alpha0, alpha1, alpha2, scalarTerm, vectorTerm, tensorTerm, energyShift)

fig1, axes1 = plt.subplots(2, 2, figsize=(10, 8))
fig1.suptitle('Terms of the Energy Shift Rb-87 $5p_{3/2}$ $F=3$ $m_F=3$', fontsize=16, fontweight='bold')

axes1[0,0].scatter(wavelength, energyShift, s=12, color='green', label='Energy Shift')
axes1[0,0].axhline(y=0.0, color='blue', linestyle='--')
axes1[0,0].plot(wavelength, energyShift, linestyle=':', color='red')         # Optional: Connect with a dashed line
axes1[0,0].set_xlabel('Wavelength (nm)')
axes1[0,0].set_ylabel('Total Energy Shift (eV)')
axes1[0,0].set_title('Total Energy Shift (eV)')
axes1[0,1].scatter(wavelength, scalarTerm, s=12, color='green', label='Energy Shift')
axes1[0,1].axhline(y=0.0, color='blue', linestyle='--')
axes1[0,1].plot(wavelength, scalarTerm, linestyle=':', color='red')         # Optional: Connect with a dashed line
axes1[0,1].set_xlabel('Wavelength (nm)')
axes1[0,1].set_ylabel('Scalar Term Energy Shift (eV)')
axes1[0,1].set_title('Scalar Term Contribution to Energy Shift')
axes1[1,0].scatter(wavelength, vectorTerm, s=12, color='green', label='Energy Shift')
axes1[1,0].axhline(y=0.0, color='blue', linestyle='--')
axes1[1,0].plot(wavelength, vectorTerm, linestyle=':', color='red')         # Optional: Connect with a dashed line
axes1[1,0].set_xlabel('Wavelength (nm)')
axes1[1,0].set_ylabel('Vector Term Energy Shift (eV)')
axes1[1,0].set_title('Vector Term Contribution to Energy Shift')
axes1[1,1].scatter(wavelength, tensorTerm, s=12, color='green', label='Energy Shift')
axes1[1,1].axhline(y=0.0, color='blue', linestyle='--')
axes1[1,1].plot(wavelength, tensorTerm, linestyle=':', color='red')         # Optional: Connect with a dashed line
axes1[1,1].set_xlabel('Wavelength (nm)')
axes1[1,1].set_ylabel('Tensor Term Energy Shift (eV)')
axes1[1,1].set_title('Tensor Term Contribution to Energy Shift')

plt.tight_layout()
plt.savefig("plot1test.svg",bbox_inches="tight")
plt.show()

fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))
fig2.suptitle(r'pMOT: Rb-87 $5p_{3/2}$ $F=3$ $m_F=3$', fontsize=16, fontweight='bold')

axes2[0,0].scatter(wavelength, energyShift, s=12, color='green', label='Energy Shift') # Plot individual points
axes2[0,0].axhline(y=0.0, color='blue', linestyle='--')
axes2[0,0].plot(wavelength, energyShift, linestyle=':', color='red')         # Optional: Connect with a dashed line
axes2[0,0].set_xlabel('Wavelength (nm)')
axes2[0,0].set_ylabel(r'Energy Shift $\Delta E$ (eV)')
axes2[0,0].set_title('Total Energy Shift')
axes2[0,0].legend()

axes2[0,1].scatter(wavelength, alpha0, s=12, color='green', label=r'$\alpha^{(0))}$') # Plot individual points
axes2[0,1].axhline(y=0.0, color='blue', linestyle='--')
axes2[0,1].plot(wavelength, alpha0, linestyle=':', color='red')         # Optional: Connect with a dashed line
axes2[0,1].set_xlabel('Wavelength (nm)')
axes2[0,1].set_ylabel(r'Scalar Polarizability $\alpha^{(0)}$ ($m^2C/V$)')
axes2[0,1].set_title(r'Scalar Polarizability $\alpha^{(0)}$')
axes2[0,1].legend()

axes2[1,0].scatter(wavelength, alpha1, s=12, color='green', label=r'$\alpha^{(1)}$') # Plot individual points
axes2[1,0].axhline(y=0.0, color='blue', linestyle='--')
axes2[1,0].plot(wavelength, alpha1, linestyle=':', color='red')         # Optional: Connect with a dashed line
axes2[1,0].set_xlabel('Wavelength (nm)')
axes2[1,0].set_ylabel(r'Vector Polarizability $\alpha^{(1)}$ ($m^2C/V$)')
axes2[1,0].set_title(r'Vector Polarizability $\alpha^{(1)}$')
axes2[1,0].legend()

axes2[1,1].scatter(wavelength, alpha2, s=12, color='green', label=r'$\alpha^{(2)}$') # Plot individual points
axes2[1,1].axhline(y=0.0, color='blue', linestyle='--')
axes2[1,1].plot(wavelength, alpha2, linestyle=':', color='red')         # Optional: Connect with a dashed line
axes2[1,1].set_xlabel('Wavelength (nm)')
axes2[1,1].set_ylabel(r'Tensor Polarizability $\alpha^{(2)}$ ($m^2C/V$)')
axes2[1,1].set_title(r'Tensor Polarizability $\alpha^{(2)}$')
axes2[1,1].legend()

plt.tight_layout()
plt.savefig("plot2test.svg",bbox_inches="tight")
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
df = df[["Wavelength (nm)", "m_F", "Energy Shift"]].copy()

# Make sure values are numeric
df["Wavelength (nm)"] = pd.to_numeric(df["Wavelength (nm)"], errors="coerce")
df["m_F"] = pd.to_numeric(df["m_F"], errors="coerce")
df["Energy Shift"] = pd.to_numeric(df["Energy Shift"], errors="coerce")

# Drop bad rows if any exist
df = df.dropna(subset=["Wavelength (nm)", "m_F", "Energy Shift"])

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
    values="Energy Shift",
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