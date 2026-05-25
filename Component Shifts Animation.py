# --------------------------------------#
# ANIMATION CODE for S, V, T Components #
# --------------------------------------#

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.ticker import ScalarFormatter
from matplotlib.animation import FFMpegWriter

import matplotlib.animation as animation
print(animation.writers.list())

# ============================================================
# User inputs
# ============================================================

csv_file = "pMOT_Data_May20.csv"

wavelength_col = "Wavelength (nm)"
mF_col = "m_F"
scalar_col = "Scalar Term"
vector_col = "Vector Term"
tensor_col = "Tensor Term"

lambda_min = 1500
lambda_max = 1550

output_gif = "ac_stark_scalar_vector_tensor_testing.gif"

mF_values = np.array([-3, -2, -1, 0, 1, 2, 3])

fps = 100
min_bound = 1e-9

# ============================================================
# Helper: decade-based y-axis scaling
# ============================================================

def get_decade_bound(yvals, min_bound=1e-9):
    max_abs = np.max(np.abs(yvals))
    if max_abs <= min_bound:
        return min_bound
    exponent = np.ceil(np.log10(max_abs))
    bound = 10.0**exponent
    return max(bound, min_bound)

# ============================================================
# Read and clean CSV
# ============================================================

raw = pd.read_csv(csv_file)

df = pd.DataFrame({
    wavelength_col: pd.to_numeric(raw[wavelength_col], errors="coerce"),
    mF_col: pd.to_numeric(raw[mF_col], errors="coerce"),
    scalar_col: pd.to_numeric(raw[scalar_col], errors="coerce"),
    vector_col: pd.to_numeric(raw[vector_col], errors="coerce"),
    tensor_col: pd.to_numeric(raw[tensor_col], errors="coerce"),
})

df = df.dropna(subset=[wavelength_col, mF_col, scalar_col, vector_col, tensor_col]).copy()
df[mF_col] = df[mF_col].astype(int)

# If wavelengths are supposed to be integer nm values, this helps avoid accidental duplicates
df[wavelength_col] = df[wavelength_col].round(1)  # df[wavelength_col] = df[wavelength_col].round(0).astype(int)

# Filter wavelength range
df = df[
    (df[wavelength_col] >= lambda_min) &
    (df[wavelength_col] <= lambda_max)
].copy()

# Sort
df = df.sort_values([wavelength_col, mF_col])

# ============================================================
# Build pivot tables
# ============================================================

pivot_scalar = df.pivot_table(
    index=wavelength_col,
    columns=mF_col,
    values=scalar_col,
    aggfunc="mean"
)

pivot_vector = df.pivot_table(
    index=wavelength_col,
    columns=mF_col,
    values=vector_col,
    aggfunc="mean"
)

pivot_tensor = df.pivot_table(
    index=wavelength_col,
    columns=mF_col,
    values=tensor_col,
    aggfunc="mean"
)

# Force the same mF column ordering
pivot_scalar = pivot_scalar.reindex(columns=mF_values)
pivot_vector = pivot_vector.reindex(columns=mF_values)
pivot_tensor = pivot_tensor.reindex(columns=mF_values)

# Keep only wavelengths for which all three data sets exist for all mF values
common_index = (
    pivot_scalar.dropna(subset=mF_values).index
    .intersection(pivot_vector.dropna(subset=mF_values).index)
    .intersection(pivot_tensor.dropna(subset=mF_values).index)
)

pivot_scalar = pivot_scalar.loc[common_index]
pivot_vector = pivot_vector.loc[common_index]
pivot_tensor = pivot_tensor.loc[common_index]

# Optional downsampling if needed for speed:
# pivot_scalar = pivot_scalar.iloc[::5]
# pivot_vector = pivot_vector.iloc[::5]
# pivot_tensor = pivot_tensor.iloc[::5]

wavelengths = pivot_scalar.index.to_numpy()
scalar_shifts = pivot_scalar[mF_values].to_numpy()
vector_shifts = pivot_vector[mF_values].to_numpy()
tensor_shifts = pivot_tensor[mF_values].to_numpy()

print(f"Number of wavelength frames: {len(wavelengths)}")
print(f"Shift array shapes:")
print("  scalar:", scalar_shifts.shape)
print("  vector:", vector_shifts.shape)
print("  tensor:", tensor_shifts.shape)

# ============================================================
# Plot setup
# ============================================================

fig, ax = plt.subplots(figsize=(11, 6))
ax.set_yscale("linear")

x_positions = np.arange(len(mF_values))

# Small horizontal offsets so the markers don't lie exactly on top of each other
dx = 0.13
x_scalar = x_positions - dx
x_vector = x_positions
x_tensor = x_positions + dx

state_labels = [rf"$|5p_{{3/2}};\,m_F={mF}\rangle$" for mF in mF_values]

# Initial frame
y_scalar0 = scalar_shifts[0]
y_vector0 = vector_shifts[0]
y_tensor0 = tensor_shifts[0]

# You can use dataset-specific colors
scalar_color = "blue"
vector_color = "red"
tensor_color = "green"

# Plot each dataset as markers + thin connecting line
line_scalar, = ax.plot(
    x_scalar, y_scalar0,
    marker="o", linestyle="-", linewidth=1.2, markersize=8,
    color=scalar_color, label="Scalar Shift"
)

line_vector, = ax.plot(
    x_vector, y_vector0,
    marker="s", linestyle="-", linewidth=1.2, markersize=8,
    color=vector_color, label="Vector Shift"
)

line_tensor, = ax.plot(
    x_tensor, y_tensor0,
    marker="^", linestyle="-", linewidth=1.2, markersize=9,
    color=tensor_color, label="Tensor Shift"
)

ax.set_xticks(x_positions)
ax.set_xticklabels(state_labels, rotation=25)
ax.set_xlabel(r"$|5p_{3/2};\,m_F\rangle$", fontsize=13)
ax.set_ylabel("AC Stark Energy Shift", fontsize=13)

# Initial y-limits based on all three datasets
all_y0 = np.concatenate([y_scalar0, y_vector0, y_tensor0])
initial_bound = get_decade_bound(all_y0, min_bound=min_bound)
ax.set_ylim(-initial_bound, initial_bound)

ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-2, 2))
ax.yaxis.set_major_formatter(formatter)

title = ax.set_title(
    rf"AC Stark Shift Components vs. $m_F$ at $\lambda={wavelengths[0]:.0f}$ nm "
    rf"$(y=\pm {initial_bound:.0e})$",
    fontsize=14
)

plt.tight_layout()

# ============================================================
# Animation update
# ============================================================

def update(frame):
    if frame % 100 == 0:
        print(f"Rendering frame {frame}/{len(wavelengths)}")

    wl = wavelengths[frame]

    y_scalar = scalar_shifts[frame]
    y_vector = vector_shifts[frame]
    y_tensor = tensor_shifts[frame]

    line_scalar.set_data(x_scalar, y_scalar)
    line_vector.set_data(x_vector, y_vector)
    line_tensor.set_data(x_tensor, y_tensor)

    # Scale y-axis based on all 3 datasets together
    all_y = np.concatenate([y_scalar, y_vector, y_tensor])
    bound = get_decade_bound(all_y, min_bound=min_bound)
    ax.set_ylim(-bound, bound)

    title.set_text(
        rf"AC Stark Shift Components vs. $m_F$ at $\lambda={wl:.0f}$ nm "
        rf"$(y=\pm {bound:.0e})$"
    )

    return [line_scalar, line_vector, line_tensor, title]

anim = FuncAnimation(
    fig,
    update,
    frames=len(wavelengths),
    interval=1000 / fps,
    blit=False
)

output_mp4 = "ac_stark_components_vid_testing.mp4"

writer = FFMpegWriter(fps=100, bitrate=1800)
anim.save(output_mp4, writer=writer)

#anim.save(output_gif, writer=PillowWriter(fps=fps))

plt.show()