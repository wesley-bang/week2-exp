# python .\plot_spectrum.py ".\data\red255mini-214848.csv"
import argparse
import csv
import glob
import os
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MultipleLocator


HEADER_MARKERS = {
    "波長(nm)",
    "Wavelength(nm)",
    "Wavelength (nm)",
}


def read_spectrum(csv_path: str) -> Tuple[List[float], List[float], Optional[str]]:
    """
    Read spectral data (380-780 nm) from a CSV exported by the spectrometer.
    Returns (wavelengths, values, unit_label).
    """
    wavelengths: List[float] = []
    values: List[float] = []
    unit_label: Optional[str] = None

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        in_data = False
        for row in reader:
            if not row:
                continue
            if not in_data:
                first = row[0].strip()
                if first in HEADER_MARKERS:
                    in_data = True
                    if len(row) > 1:
                        unit_label = row[1].strip() or unit_label
                continue

            # After header: parse numeric rows (wavelength, value)
            if len(row) < 2:
                continue
            try:
                wl = float(row[0].strip())
                val = float(row[1].strip())
            except ValueError:
                continue

            if 380 <= wl <= 780:
                wavelengths.append(wl)
                values.append(val)

    if not wavelengths:
        raise ValueError(f"No spectral data found in {csv_path} (380-780 nm).")

    return wavelengths, values, unit_label


def wavelength_to_rgb(wl: float, gamma: float = 0.8) -> Tuple[float, float, float]:
    # Approximate visible spectrum color for 380–780 nm.
    if wl < 380 or wl > 780:
        return (0.0, 0.0, 0.0)

    if wl < 440:
        r = -(wl - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif wl < 490:
        r = 0.0
        g = (wl - 440) / (490 - 440)
        b = 1.0
    elif wl < 510:
        r = 0.0
        g = 1.0
        b = -(wl - 510) / (510 - 490)
    elif wl < 580:
        r = (wl - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif wl < 645:
        r = 1.0
        g = -(wl - 645) / (645 - 580)
        b = 0.0
    else:
        r = 1.0
        g = 0.0
        b = 0.0

    if wl < 420:
        factor = 0.3 + 0.7 * (wl - 380) / (420 - 380)
    elif wl <= 700:
        factor = 1.0
    else:
        factor = 0.3 + 0.7 * (780 - wl) / (780 - 700)

    r = (r * factor) ** gamma
    g = (g * factor) ** gamma
    b = (b * factor) ** gamma
    return (r, g, b)


def spectrum_colormap(n: int = 256) -> LinearSegmentedColormap:
    wls = np.linspace(380, 780, n)
    colors = [wavelength_to_rgb(wl) for wl in wls]
    return LinearSegmentedColormap.from_list("spectrum", colors, N=n)


def _plot_colored_line(
    wls: List[float],
    vals: List[float],
    ax: plt.Axes,
    cmap: LinearSegmentedColormap,
    norm: Normalize,
    label: str,
) -> LineCollection:
    points = np.column_stack([wls, vals])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=1.6, label=label)
    lc.set_array(np.array(wls[:-1]))
    ax.add_collection(lc)
    return lc


def plot_spectra(csv_paths: List[str], out_path: str, show: bool) -> None:
    unit_label: Optional[str] = None

    fig, ax = plt.subplots(figsize=(8, 4.5))
    cmap = spectrum_colormap()
    norm = Normalize(380, 780)

    for path in csv_paths:
        wls, vals, unit = read_spectrum(path)
        if unit and not unit_label:
            unit_label = unit
        label = os.path.splitext(os.path.basename(path))[0]
        if len(wls) >= 2:
            _plot_colored_line(wls, vals, ax, cmap, norm, label)
        else:
            ax.plot(wls, vals, linewidth=1.4, label=label)

    ax.autoscale_view()
    ax.set_xlim(380, 780)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    # if unit_label:
    #     # ax.set_ylabel(f"Spectral power {unit_label}")
    # else:
    #     # ax.set_ylabel("Spectral power")
    # if len(csv_paths) == 1:
    #     # ax.set_title(os.path.basename(csv_paths[0]))
    # else:
    #     # ax.set_title("Spectra")
    #     ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()


def make_out_path(csv_path: str) -> str:
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    prefix = base_name.split("-", 1)[0]
    out_dir = "image"
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{prefix}.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read 380-780 nm spectral data from CSV and plot."
    )
    parser.add_argument("csv_paths", nargs="*", help="CSV file path(s)")
    parser.add_argument(
        "--out",
        default=None,
        help="Output image path (default: image/<prefix>.png)",
    )
    parser.add_argument("--show", action="store_true", help="Show the plot window")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all CSV files under the data/ directory",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory to scan when using --all (default: data)",
    )
    args = parser.parse_args()

    if args.all:
        if args.out:
            print("Note: --out is ignored when using --all.")
        pattern = os.path.join(args.data_dir, "*.csv")
        csv_paths = sorted(glob.glob(pattern))
        if not csv_paths:
            raise SystemExit(f"No CSV files found in: {args.data_dir}")
        for path in csv_paths:
            out_path = make_out_path(path)
            plot_spectra([path], out_path, args.show)
            print(f"Saved plot to: {out_path}")
        return

    if not args.csv_paths:
        raise SystemExit("Please provide CSV file path(s), or use --all.")

    first = args.csv_paths[0]
    out_path = args.out or make_out_path(first)

    plot_spectra(args.csv_paths, out_path, args.show)
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
