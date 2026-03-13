#===================================================================================================#
# Usage:
# python .\plot_viewing_zone.py
# python .\plot_viewing_zone.py --shim with --integrate R+G+B 
# python .\plot_viewing_zone.py --group rb --shim with --integrate R+B G+B --out .\image\rb_with_shim.png
# python .\plot_viewing_zone.py --group both --integrate R G B R+G+B
#===================================================================================================#

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(".")
VIEW_DISTANCE_MM = 100
POSITION_MAP = {"33": -90, "22": -60, "11": -30, "0": 0, "1": 30, "2": 60, "3": 90}
FIXED_BANDS_NM = {
    "B": (430.0, 500.0),
    "G": (500.0, 580.0),
    "R": (600.0, 680.0),
}
GROUP_TITLES = {
    "gp": "Magenta-Green spatial distribution",
    "rb": "Blue-Red spatial distribution",
}
COMBO_SPECS = {
    "R": {"bands": ("R",), "label": "R", "color": "red"},
    "G": {"bands": ("G",), "label": "G", "color": "green"},
    "B": {"bands": ("B",), "label": "B", "color": "blue"},
    "RG": {"bands": ("R", "G"), "label": "R+G", "color": "goldenrod"},
    "RB": {"bands": ("R", "B"), "label": "R+B", "color": "magenta"},
    "GB": {"bands": ("G", "B"), "label": "G+B", "color": "cyan"},
    "RGB": {"bands": ("R", "G", "B"), "label": "R+G+B", "color": "black"},
}
PLOT_ORDER = ["R", "G", "B", "RG", "RB", "GB", "RGB"]
SPECTRUM_HEADER_MARKERS = {"波長(nm)", "豕｢髟ｷ(nm)"}
INTEGRATION_TIME_KEYS = [
    "積分時間(ms)",
    "積分時間",
    "遨榊・譎る俣(ms)",
    "遨榊・譎る俣",
    "Integration Time(ms)",
    "Integration Time",
]
ILLUMINANCE_KEYS = ["照度E(lx)", "辣ｧ蠎ｦE(lx)"]


def read_spectrum_csv(path: Path):
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    metadata = {}
    rows = []
    in_spectrum = False

    for line in lines:
        parts = [part.strip() for part in line.split(",")]
        if not in_spectrum:
            if len(parts) >= 2 and parts[0] in SPECTRUM_HEADER_MARKERS:
                in_spectrum = True
                continue
            if len(parts) >= 2 and parts[0]:
                metadata[parts[0]] = parts[1]
            continue

        if len(parts) >= 2 and parts[0].isdigit():
            rows.append((int(parts[0]), float(parts[1])))

    df = pd.DataFrame(rows, columns=["wavelength_nm", "spectral_power"])
    return metadata, df


def get_integration_time_ms(metadata):
    for key in INTEGRATION_TIME_KEYS:
        if key in metadata:
            value = str(metadata[key]).replace("ms", "").strip()
            return float(value)
    raise KeyError(f"Cannot find integration time in metadata keys: {list(metadata.keys())}")


def get_illuminance_lx(metadata):
    for key in ILLUMINANCE_KEYS:
        if key in metadata:
            return float(metadata[key])
    return float(np.nan)


def parse_integrate_selection(value: str):
    raw = value.strip().upper().replace("+", "")
    if not raw:
        raise argparse.ArgumentTypeError("Integration mode cannot be empty.")
    if any(channel not in "RGB" for channel in raw):
        raise argparse.ArgumentTypeError(
            f"Unsupported integration mode '{value}'. Use R, G, B, R+G, R+B, G+B, or R+G+B."
        )
    if len(set(raw)) != len(raw):
        raise argparse.ArgumentTypeError(f"Duplicated channel in integration mode '{value}'.")

    canonical = "".join(channel for channel in "RGB" if channel in raw)
    if canonical not in COMBO_SPECS:
        raise argparse.ArgumentTypeError(
            f"Unsupported integration mode '{value}'. Use R, G, B, R+G, R+B, G+B, or R+G+B."
        )
    return canonical


def normalize_integrate_selections(selections):
    normalized = []
    seen = set()
    for selection in selections:
        canonical = parse_integrate_selection(selection)
        if canonical in seen:
            continue
        seen.add(canonical)
        normalized.append(canonical)
    return normalized


def integrate_band(df, lo, hi, value_col="spectral_power"):
    sub = df[(df["wavelength_nm"] >= lo) & (df["wavelength_nm"] <= hi)]
    if sub.empty:
        return 0.0
    return float(np.trapezoid(sub[value_col], sub["wavelength_nm"]))


def build_combo_values(band_values):
    combo_values = {}
    for combo, spec in COMBO_SPECS.items():
        combo_values[combo] = sum(band_values[band] for band in spec["bands"])
    return combo_values


def expand_plot_combos(selected_combos):
    expanded = set()
    for combo in selected_combos:
        expanded.update(COMBO_SPECS[combo]["bands"])
        expanded.add(combo)
    return [combo for combo in PLOT_ORDER if combo in expanded]


def parse_position_from_name(stem: str, prefix: str):
    match = re.match(rf"{prefix}(\d+)!?-", stem)
    if not match:
        raise ValueError(f"Cannot parse scan index from filename: {stem}")

    idx = match.group(1)
    if idx not in POSITION_MAP:
        raise ValueError(f"Unknown scan index {idx} in {stem}")
    return idx, POSITION_MAP[idx]


def summarize_group(prefix: str, use_shim: bool):
    records = []

    for path in sorted(BASE_DIR.glob(f"{prefix}*.csv")):
        has_shim = "!" in path.name
        if has_shim != use_shim:
            continue

        idx, x_mm = parse_position_from_name(path.stem, prefix)
        metadata, df = read_spectrum_csv(path)
        integration_time_ms = get_integration_time_ms(metadata)
        df["spectral_power_itnorm"] = df["spectral_power"] / integration_time_ms

        band_vals = {
            name: integrate_band(df, *rng, value_col="spectral_power_itnorm")
            for name, rng in FIXED_BANDS_NM.items()
        }
        combo_vals = build_combo_values(band_vals)
        theta_rad = np.arctan2(x_mm, VIEW_DISTANCE_MM)
        correction_factor = 1 / np.cos(theta_rad)
        combo_corr_vals = {
            f"{combo}_corr": value * correction_factor for combo, value in combo_vals.items()
        }
        records.append(
            {
                "file": path.name,
                "group": prefix,
                "scan_idx": idx,
                "x_mm": x_mm,
                "theta_deg": np.degrees(theta_rad),
                "illuminance_lx": get_illuminance_lx(metadata),
                "integration_time_ms": integration_time_ms,
                "correction_factor": correction_factor,
                **combo_vals,
                **combo_corr_vals,
            }
        )

    if not records:
        shim_label = "with '!'" if use_shim else "without '!'"
        raise FileNotFoundError(f"No {prefix} CSV files found {shim_label} in {BASE_DIR.resolve()}.")

    return pd.DataFrame(records).sort_values("x_mm").reset_index(drop=True)


def plot_spatial_distribution(summary: pd.DataFrame, prefix: str, out_path: Path, plot_combos):
    plt.figure(figsize=(8, 5))
    for combo in plot_combos:
        spec = COMBO_SPECS[combo]
        plt.plot(
            summary["x_mm"],
            summary[f"{combo}_corr"],
            marker="o",
            label=spec["label"],
            color=spec["color"],
        )
    plt.xlabel("Scan position x (mm)")
    plt.ylabel("Integrated intensity")
    plt.title(GROUP_TITLES.get(prefix, f"{prefix.upper()} spatial distribution"))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot the viewing-zone spatial distribution using fixed RGB wavelength bands."
    )
    parser.add_argument(
        "--group",
        choices=["gp", "rb", "both"],
        default="gp",
        help="Select which dataset prefix to process (default: gp).",
    )
    parser.add_argument(
        "--shim",
        choices=["without", "with"],
        default="without",
        help="Read files without '!' or with '!' in the filename (default: without).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path for a single generated spatial-distribution figure.",
    )
    parser.add_argument(
        "--integrate",
        nargs="+",
        type=parse_integrate_selection,
        default=["R", "G", "B", "RGB"],
        metavar="MODE",
        help="Integration mode(s): R, G, B, R+G, R+B, G+B, or R+G+B.",
    )
    return parser


def resolve_output_path(group: str, use_shim: bool) -> Path:
    if group == "gp":
        return Path("./image/gp_overlay_with_shim.png" if use_shim else "./image/gp_overlay.png")
    if use_shim:
        return Path(f"./image/{group}_spatial_distribution_with_shim.png")
    return Path(f"./image/{group}_spatial_distribution_final.png")


def main():
    parser = build_parser()
    args = parser.parse_args()
    selected_combos = normalize_integrate_selections(args.integrate)
    plot_combos = expand_plot_combos(selected_combos)
    use_shim = args.shim == "with"

    if args.out and args.group == "both":
        parser.error("--out can only be used when exactly one figure will be generated.")

    selected_groups = ["rb", "gp"] if args.group == "both" else [args.group]
    saved_paths = []

    for group in selected_groups:
        summary = summarize_group(group, use_shim)
        out_path = args.out if args.out else resolve_output_path(group, use_shim)
        plot_spatial_distribution(summary, group, out_path, plot_combos)
        saved_paths.append(out_path)

    for path in saved_paths:
        print(f"Saved plot to: {path}")

    print(
        "Done. "
        f"Shim mode: {args.shim}. "
        "Modes: "
        + ", ".join(COMBO_SPECS[combo]["label"] for combo in selected_combos)
        + "."
    )


if __name__ == "__main__":
    main()
