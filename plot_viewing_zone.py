from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base = Path(".")

VIEW_DISTANCE_MM = 100
STEP_MM = 30
POSITION_MAP = {"33": -90, "22": -60, "11": -30, "0": 0, "1": 30, "2": 60, "3": 90}
BANDS_NM = {
    "B": (430, 500),
    "G": (500, 580),
    "R": (600, 680),
}

def read_spectrum_csv(path: Path):
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    metadata = {}
    rows = []
    in_spectrum = False
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if not in_spectrum:
            if len(parts) >= 2 and parts[0] == "波長(nm)":
                in_spectrum = True
                continue
            if len(parts) >= 2 and parts[0]:
                metadata[parts[0]] = parts[1]
        else:
            if len(parts) >= 2 and parts[0].isdigit():
                rows.append((int(parts[0]), float(parts[1])))
    df = pd.DataFrame(rows, columns=["wavelength_nm", "spectral_power"])
    return metadata, df

def get_integration_time_ms(metadata):
    for key in ["積分時間(ms)", "積分時間", "Integration Time(ms)", "Integration Time"]:
        if key in metadata:
            value = str(metadata[key]).replace("ms", "").strip()
            return float(value)
    raise KeyError(f"Cannot find integration time in metadata keys: {list(metadata.keys())}")

def integrate_band(df, lo, hi, value_col="spectral_power"):
    sub = df[(df["wavelength_nm"] >= lo) & (df["wavelength_nm"] <= hi)]
    return np.trapezoid(sub[value_col], sub["wavelength_nm"])

def parse_position_from_name(stem: str, prefix: str):
    m = re.match(rf"{prefix}(\d+)-", stem)
    if not m:
        raise ValueError(f"Cannot parse scan index from filename: {stem}")
    idx = m.group(1)
    if idx not in POSITION_MAP:
        raise ValueError(f"Unknown scan index {idx} in {stem}")
    return idx, POSITION_MAP[idx]

def summarize_group(prefix: str):
    records = []
    spectra = {}
    for path in sorted(base.glob(f"{prefix}*.csv")):
        if "!" in path.name:
            continue
        stem = path.stem
        idx, x_mm = parse_position_from_name(stem, prefix)
        metadata, df = read_spectrum_csv(path)

        integration_time_ms = get_integration_time_ms(metadata)
        df["spectral_power_itnorm"] = df["spectral_power"] / integration_time_ms

        total = np.trapezoid(df["spectral_power_itnorm"], df["wavelength_nm"])
        band_vals = {
            name: integrate_band(df, *rng, value_col="spectral_power_itnorm")
            for name, rng in BANDS_NM.items()
        }
        peak_row = df.loc[df["spectral_power_itnorm"].idxmax()]
        theta_rad = np.arctan2(x_mm, VIEW_DISTANCE_MM)
        correction_factor = 1 / np.cos(theta_rad)
        record = {
            "file": path.name,
            "group": prefix,
            "scan_idx": idx,
            "x_mm": x_mm,
            "theta_deg": np.degrees(theta_rad),
            "illuminance_lx": float(metadata.get("照度E(lx)", np.nan)),
            "integration_time_ms": integration_time_ms,
            "total_area": total,
            "peak_nm": int(peak_row["wavelength_nm"]),
            "peak_value": float(peak_row["spectral_power"]),
            **band_vals,
            "correction_factor": correction_factor,
            "illuminance_lx_corr": float(metadata.get("照度E(lx)", np.nan)) * correction_factor,
            "total_area_corr": total * correction_factor,
            "B_corr": band_vals["B"] * correction_factor,
            "G_corr": band_vals["G"] * correction_factor,
            "R_corr": band_vals["R"] * correction_factor,
        }
        records.append(record)
        spectra[x_mm] = df.copy()
    summary = pd.DataFrame(records).sort_values("x_mm").reset_index(drop=True)
    return summary, spectra

def plot_spatial_distribution(summary: pd.DataFrame, prefix: str, out_path: Path):
    plt.figure(figsize=(8, 5))
    plt.plot(summary["x_mm"], summary["total_area_corr"], marker="o", label="R+G+B", color="black")
    plt.plot(summary["x_mm"], summary["B_corr"], marker="o", label="B", color="blue")
    plt.plot(summary["x_mm"], summary["G_corr"], marker="o", label="G", color="green")
    plt.plot(summary["x_mm"], summary["R_corr"], marker="o", label="R", color="red")
    plt.xlabel("Scan position x (mm)")
    plt.ylabel("Integrated intensity")
    if prefix == "gp":
        plt.title(f"Magenta-Green spatial distribution")
    else :
        plt.title(f"Blue-Red spatial distribution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_spectra_overlay(spectra: dict, prefix: str, out_path: Path):
    plt.figure(figsize=(8, 5))
    for x_mm in sorted(spectra):
        df = spectra[x_mm]
        plt.plot(df["wavelength_nm"], df["spectral_power"], label=f"x={x_mm} mm")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Spectral power")
    plt.title(f"{prefix.upper()} spectra at 7 scan positions")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

rb_summary, rb_spectra = summarize_group("rb")
gp_summary, gp_spectra = summarize_group("gp")

# summary = pd.concat([rb_summary, gp_summary], ignore_index=True)
# summary.to_csv("viewing_zone_summary_final.csv", index=False, encoding="utf-8-sig")

plot_spatial_distribution(rb_summary, "rb", Path("./image/rb_spatial_distribution_final.png"))
plot_spatial_distribution(gp_summary, "gp", Path("./image/gp_spatial_distribution_final.png"))
plot_spectra_overlay(rb_spectra, "rb", Path("./image/rb_spectra_overlay_final.png"))
plot_spectra_overlay(gp_spectra, "gp", Path("./image/gp_spectra_overlay_final.png"))

print("Done.")
