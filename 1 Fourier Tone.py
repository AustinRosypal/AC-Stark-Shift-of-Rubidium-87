# import pandas as pd

# def find_good_row_pairs(
#     csv_path,
#     s_col,
#     v_col,
#     t_col,
#     mF_col,
#     wavelength_col,
#     ratio_scalar_threshold,
#     ratio_tensor_threshold,
# ):

#     df = pd.read_csv(csv_path)
#     mF_value = 3

#     required_cols = [s_col, v_col, t_col, mF_col, wavelength_col]
#     for col in required_cols:
#         if col not in df.columns:
#             raise ValueError(f"Missing required column: {col}")

#     df = df[df[mF_col] == mF_value].copy()
#     if len(df) < 2:
#         raise ValueError(f"Fewer than two rows found with {mF_col} = {mF_value}.")

#     df = df.reset_index(drop=False).rename(columns={"index": "original_index"})

#     a = df[s_col].to_numpy(dtype=float)
#     b = df[v_col].to_numpy(dtype=float)
#     c = df[t_col].to_numpy(dtype=float)
#     wavelength = df[wavelength_col].to_numpy(dtype=float)

#     n = len(df)
#     print(f"Length of CSV File: {n}")

#     if n < 2:
#         raise ValueError("CSV must contain at least two rows.")

#     for i in range (n):
#         if (b[i]/a[i] >= ratio_scalar_threshold and b[i]/c[i] >= ratio_tensor_threshold):
#             v_s_ratio = b[i]/a[i]
#             v_t_ratio = b[i]/c[i]
#             print(f"Wavelength: {wavelength[i]}")
#             print(f"V/S = {v_s_ratio}")
#             print(f"V/T = {v_t_ratio}")
#             print("------------------------")

# if __name__ == "__main__":

#     results = find_good_row_pairs(
#         #csv_path="DataROI_perIntensity.csv",
#         csv_path="June10_5p_32_Data.csv",
#         wavelength_col="Wavelength (nm)",
#         mF_col="m_F",
#         #s_col="Scalar Term (MHz/I)",
#         #v_col="Vector Term (MHz/I)",
#         #t_col="Tensor Term (MHz/I)",
#         s_col="Scalar Polarizability",
#         v_col="Vector Polarizability",#"Vector Polarizability",
#         t_col="Tensor Polarizability", #"Tensor Polarizability",
#         ratio_scalar_threshold=2.0,
#         ratio_tensor_threshold=2.0,
#     )





# For Differential Polarizability CSV

import pandas as pd
import numpy as np

def find_good_row_pairs(
    csv_path,
    wavelength_col,
    s_col,
    v_col,
    t_col,
    ratio_scalar_threshold,
    ratio_tensor_threshold,
):

    df = pd.read_csv(csv_path)

    required_cols = [s_col, v_col, t_col, wavelength_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.reset_index(drop=False).rename(columns={"index": "original_index"})

    a = df[s_col].to_numpy(dtype=float)
    b = df[v_col].to_numpy(dtype=float)
    c = df[t_col].to_numpy(dtype=float)
    wavelength = df[wavelength_col].to_numpy(dtype=float)

    n = len(df)
    print(f"Length of CSV File: {n}")

    if n < 2:
        raise ValueError("CSV must contain at least two rows.")

    numFound = 0
    for i in range (n):
        if (np.abs(b[i]/a[i]) >= ratio_scalar_threshold and np.abs(b[i]/c[i]) >= ratio_tensor_threshold):
            numFound += 1
            v_s_ratio = b[i]/a[i]
            v_t_ratio = b[i]/c[i]
            print(f"Wavelength: {wavelength[i]} nm")
            print(f"Frequency: {299792458 / (wavelength[i]*1e-9) / 1e12} THz")
            print(f"V/S = {v_s_ratio}")
            print(f"V/T = {v_t_ratio}")
            print("------------------------")
    print(f"Total Candidates: {numFound}")

if __name__ == "__main__":

    results = find_good_row_pairs(
        csv_path="Differential_Polarizabilities.csv",
        wavelength_col="Wavelength (nm)",
        s_col="Differential Scalar Polarizability",
        v_col="Differential Vector Polarizability",#"Vector Polarizability",
        t_col="Differential Tensor Polarizability", #"Tensor Polarizability",
        ratio_scalar_threshold=100.0,
        ratio_tensor_threshold=1.0,
    )
