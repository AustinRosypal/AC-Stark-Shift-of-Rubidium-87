import matplotlib.pyplot as plt
import numpy as np
from sympy.physics.wigner import wigner_6j
from matplotlib.animation import FuncAnimation, PillowWriter
import pandas as pd
from matplotlib.ticker import ScalarFormatter

csv_file = "pMOT_Data_May20.csv"
df = pd.read_csv(csv_file)
print(df.columns)

# --------------------------------------#
# ANIMATION CODE for total energy shift #
# --------------------------------------#

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
anim.save(output_gif, writer=PillowWriter(fps=fps))