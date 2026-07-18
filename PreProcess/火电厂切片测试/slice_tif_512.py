import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.windows import Window, transform as window_transform


# =========================
# Direct run config
# =========================
# Output is only the final image tiles for object-detection labeling/training.
RUN_CONFIG = {
    "INPUT_TIF": r"F:\3能源金三角基础设施识别\火力发电厂\1数据集制作与备注\1火电厂数据集制作\河南省\原tif\河南76.tif",

    # None means: create <input_tif_parent>\1_tiles_768
    "OUT_DIR": r"F:\3能源金三角基础设施识别\火力发电厂\1数据集制作与备注\1火电厂数据集制作\河南省\切片",

    "TILE_SIZE": 512,                   #512   256
    "STRIDE": 256,                      #768   384

    # Skip pure black / transparent edge tiles.
    "BLACK_THRESHOLD": 3,
    "MIN_VALID_RATIO": 0.20,

    "NAME_PREFIX": "河南76_",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Cut a GeoTIFF into object-detection image tiles.")
    parser.add_argument("input_tif", nargs="?", default=None, help="Input GeoTIFF path.")
    parser.add_argument("--out-dir", default=None, help="Output directory.")
    parser.add_argument("--tile-size", type=int, default=None, help="Tile size in pixels.")
    parser.add_argument("--stride", type=int, default=None, help="Sliding-window stride in pixels.")
    parser.add_argument("--black-threshold", type=int, default=None, help="Black edge threshold.")
    parser.add_argument("--min-valid-ratio", type=float, default=None, help="Minimum valid pixel ratio.")
    parser.add_argument("--name-prefix", default=None, help="Output filename prefix.")
    return parser.parse_args()


def build_args():
    if len(sys.argv) == 1:
        return SimpleNamespace(**RUN_CONFIG)

    cli = parse_args()
    return SimpleNamespace(
        INPUT_TIF=cli.input_tif or RUN_CONFIG["INPUT_TIF"],
        OUT_DIR=cli.out_dir if cli.out_dir is not None else RUN_CONFIG["OUT_DIR"],
        TILE_SIZE=cli.tile_size if cli.tile_size is not None else RUN_CONFIG["TILE_SIZE"],
        STRIDE=cli.stride if cli.stride is not None else RUN_CONFIG["STRIDE"],
        BLACK_THRESHOLD=(
            cli.black_threshold if cli.black_threshold is not None else RUN_CONFIG["BLACK_THRESHOLD"]
        ),
        MIN_VALID_RATIO=(
            cli.min_valid_ratio if cli.min_valid_ratio is not None else RUN_CONFIG["MIN_VALID_RATIO"]
        ),
        NAME_PREFIX=cli.name_prefix or RUN_CONFIG["NAME_PREFIX"],
    )


def start_positions(length, tile_size, stride):
    if length <= tile_size:
        return [0]

    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def valid_mask(rgb, alpha, black_threshold):
    valid = ~np.all(rgb <= black_threshold, axis=0)
    if alpha is not None:
        valid &= alpha > 0
    return valid


def print_progress(done, total, saved, skipped):
    ratio = done / total if total else 1.0
    print(
        f"\rProgress: {done}/{total} ({ratio * 100:5.1f}%) | saved {saved} | skipped {skipped}",
        end="",
        flush=True,
    )


def slice_tif(args):
    input_tif = Path(args.INPUT_TIF)
    if not input_tif.exists():
        raise FileNotFoundError(f"Input GeoTIFF not found: {input_tif}")

    out_dir = Path(args.OUT_DIR) if args.OUT_DIR else input_tif.with_name(
        f"{input_tif.stem}_tiles_{args.TILE_SIZE}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_tif) as src:
        if src.count < 3:
            raise ValueError("Input raster must have at least 3 bands.")

        xs = start_positions(src.width, args.TILE_SIZE, args.STRIDE)
        ys = start_positions(src.height, args.TILE_SIZE, args.STRIDE)
        total = len(xs) * len(ys)

        saved = 0
        skipped = 0
        done = 0
        print_progress(done, total, saved, skipped)

        for row, yoff in enumerate(ys):
            for col, xoff in enumerate(xs):
                done += 1
                window = Window(xoff, yoff, args.TILE_SIZE, args.TILE_SIZE)
                rgb = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
                alpha = src.read(4, window=window, boundless=True, fill_value=0) if src.count >= 4 else None

                valid_ratio = float(valid_mask(rgb, alpha, args.BLACK_THRESHOLD).mean())
                if valid_ratio < args.MIN_VALID_RATIO:
                    skipped += 1
                    print_progress(done, total, saved, skipped)
                    continue

                name = f"{args.NAME_PREFIX}{saved + 1}.tif"
                profile = src.profile.copy()
                profile.update(
                    driver="GTiff",
                    height=args.TILE_SIZE,
                    width=args.TILE_SIZE,
                    count=3,
                    transform=window_transform(window, src.transform),
                    dtype=rgb.dtype,
                    nodata=0,
                    compress="lzw",
                )
                with rasterio.open(out_dir / name, "w", **profile) as dst:
                    dst.write(rgb)

                saved += 1
                print_progress(done, total, saved, skipped)

    print()
    print(f"Done. Saved {saved} tiles, skipped {skipped} empty/edge tiles.")
    print(f"Output tiles: {out_dir}")


def main():
    slice_tif(build_args())


if __name__ == "__main__":
    main()
