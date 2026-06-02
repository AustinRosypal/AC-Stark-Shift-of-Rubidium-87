import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

def find_good_row_pairs(
    csv_path,
    output_path, #"pMOT_Data_May20.csv",
    s_col,
    v_col,
    t_col,
    mF_col,
    wavelength_col,
    alpha,
    beta,
    gamma,
    ratio_scalar_threshold,
    ratio_tensor_threshold,
    max_ac_radius,
    max_abs_a_sum,
    max_abs_c_sum,
    require_positive_b_sum,
    keep_top,
):
    """
    Find pairs of rows where:
        a1 + a2 (scalar) is close to 0
        c1 + c2 (tensor) is close to 0
        b1 + b2 (vector) is large

    Acceptance conditions:
        abs(b1 + b2) > ratio_threshold * abs(a1 + a2)
        abs(b1 + b2) > ratio_threshold * abs(c1 + c2)

    By default, also require:
        b1 + b2 > 0

    Optional geometric constraint:
        sqrt((a1+a2)^2 + (c1+c2)^2) <= max_ac_radius

    Optional component-wise constraints:
        abs(a1+a2) <= max_abs_a_sum
        abs(c1+c2) <= max_abs_c_sum

    Optional residual check:
        If alpha, beta, gamma are supplied, the code computes
            residual_i = alpha*a_i + beta*b_i + gamma*c_i - E_i
        for diagnostic purposes.
    """

    df = pd.read_csv(csv_path)
    mF_value = 3

    required_cols = [s_col, v_col, t_col, mF_col, wavelength_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df[df[mF_col] == mF_value].copy()
    if len(df) < 2:
        raise ValueError(f"Fewer than two rows found with {mF_col} = {mF_value}.")

    df = df.reset_index(drop=False).rename(columns={"index": "original_index"})

    a = df[s_col].to_numpy(dtype=float)
    b = df[v_col].to_numpy(dtype=float)
    c = df[t_col].to_numpy(dtype=float)
    wavelength = df[wavelength_col].to_numpy(dtype=float)

    n = len(df)
    print(n)

    if n < 2:
        raise ValueError("CSV must contain at least two rows.")

    # We search in the (a, c) plane because we want:
    #     a_j ≈ -a_i
    #     c_j ≈ -c_i
    points = np.column_stack([a, c])
    tree = cKDTree(points)

    # If no search radius is provided, choose a scale-aware default.
    # You should tune this for your data.
    if max_ac_radius is None:
        a_scale = np.nanstd(a)
        c_scale = np.nanstd(c)
        data_scale = np.sqrt(a_scale**2 + c_scale**2)

        if data_scale == 0 or not np.isfinite(data_scale):
            data_scale = 1.0

        max_ac_radius = 5*data_scale #0.05 * data_scale

    residual = None
    if alpha is not None and beta is not None and gamma is not None:
        residual = alpha * a + beta * b + gamma * c

    results = []

    for i in range(n):
        if (i%1000==0): print(i)
        target = np.array([-a[i], -c[i]])

        candidate_indices = tree.query_ball_point(
            target,
            r=max_ac_radius
        )

        for j in candidate_indices:
            # Avoid self-pairing and duplicate pairs.
            if j <= i:
                continue

            a_sum = a[i] + a[j]
            b_sum = b[i] + b[j]
            c_sum = c[i] + c[j]

            abs_a_sum = abs(a_sum)
            abs_b_sum = abs(b_sum)
            abs_c_sum = abs(c_sum)

            if require_positive_b_sum and b_sum <= 0:
                continue

            if max_abs_a_sum is not None and abs_a_sum > max_abs_a_sum:
                continue

            if max_abs_c_sum is not None and abs_c_sum > max_abs_c_sum:
                continue

            ac_error = np.sqrt(a_sum**2 + c_sum**2)

            # Acceptance Criteria:
            #     |b_sum| > R |a_sum|
            #     |b_sum| > R |c_sum|
            #     \Delta \lambda <= 0.3904 nm
            condition_a = abs_b_sum > ratio_scalar_threshold * abs_a_sum
            condition_c = abs_b_sum > ratio_tensor_threshold * abs_c_sum
            condition_EOM_i = wavelength[i] >= 1529.314 - 0.3904 and wavelength[i] <= 1529.314 + 0.3904
            condition_EOM_j = wavelength[j] >= 1529.314 - 0.3904 and wavelength[j] <= 1529.314 + 0.3904

            if not (condition_a and condition_c and condition_EOM_i and condition_EOM_j): #condition_proximity):
                continue

            if abs_a_sum == 0:
                b_over_a_cancellation = np.inf
            else:
                b_over_a_cancellation = abs_b_sum / abs_a_sum

            if abs_c_sum == 0:
                b_over_c_cancellation = np.inf
            else:
                b_over_c_cancellation = abs_b_sum / abs_c_sum

            record = {
                "Tone1 Wavelength (nm)": wavelength[i],
                "Tone2 Wavelength (nm)": wavelength[j],
                "Combined_ST_Magnitude": ac_error,

                # "alphaS_1": a[i],
                # "alphaV_1": b[i],
                # "alphaT_1": c[i],
                # "alphaS_2": a[j],
                # "alphaV_2": b[j],
                # "alphaT_2": c[j],
                # "alphaS_sum": a_sum,
                # "alphaV_sum": b_sum,
                # "alphaT_sum": c_sum,
                # "abs_alphaS_sum": abs_a_sum,
                # "abs_alphaV_sum": abs_b_sum,
                # "abs_alphaT_sum": abs_c_sum,
                # "alphaV to alphaS Ratio": b_over_a_cancellation,
                # "alphaV to alphaT Ratio": b_over_c_cancellation,

                "S_1": a[i],
                "V_1": b[i],
                "T_1": c[i],
                "S_2": a[j],
                "V_2": b[j],
                "T_2": c[j],
                "S_sum": a_sum,
                "V_sum": b_sum,
                "T_sum": c_sum,
                "abs_S_sum": abs_a_sum,
                "abs_V_sum": abs_b_sum,
                "abs_T_sum": abs_c_sum,
                "V to S Ratio": b_over_a_cancellation,
                "V to T Ratio": b_over_c_cancellation,
            }

            if residual is not None:
                record["residual1"] = residual[i]
                record["residual2"] = residual[j]
                record["abs_residual1"] = abs(residual[i])
                record["abs_residual2"] = abs(residual[j])

            results.append(record)

    results = pd.DataFrame(results)

    if len(results) == 0:
        print("No valid row pairs found.")
        return results

    # If requiring positive b_sum, sort by largest positive b_sum.
    # Otherwise sort by largest |b_sum|.
    if require_positive_b_sum:
        results = results.sort_values(
            by=["alphaV_sum", "Combined_ST_Magnitude"],
            ascending=[False, True]
        )
    else:
        results = results.sort_values(
            by=["abs_alphaV_sum", "Combined_ST_Magnitude"],
            ascending=[False, True]
        )

    if keep_top is not None:
        results = results.head(keep_top)

    results.to_csv(output_path, index=False)

    print(f"Found {len(results)} valid row pairs.")
    print(f"Saved results to: {output_path}")

    return results


if __name__ == "__main__":

    results = find_good_row_pairs(
        csv_path="May30_tenthousandth_perIntensity.csv",
        output_path="June2Tones_terms.csv",

        wavelength_col="Wavelength (nm)",
        mF_col="m_F",
        s_col="Scalar Term (MHz/I)",
        v_col="Vector Term (MHz/I)",
        t_col="Tensor Term (MHz/I)",
        #s_col="Scalar Polarizability",
        #v_col="Vector Polarizability",
        #t_col="Tensor Polarizability",
       
        # Optional residual diagnostics for:
        # alpha*a + beta*b + gamma*c = E
        alpha=None,
        beta=None,
        gamma=None,

        # Require:
        # |b1 + b2| > 10*|a1 + a2|
        # |b1 + b2| > 10*|c1 + c2|
        ratio_scalar_threshold=100.0,
        ratio_tensor_threshold=1.0,

        # Search radius in the (a, c) plane.
        # Tune this if you get too many or too few results.
        max_ac_radius=None,

        # Optional hard cutoffs.
        max_abs_a_sum=None,
        max_abs_c_sum=None,

        # True means b1 + b2 must be positive.
        # False allows either positive or negative b sums.
        require_positive_b_sum=True,

        # None saves all results.
        # Example: keep_top=1000 keeps only the best 1000.
        keep_top=None,
    )

    print(results.head(20))
