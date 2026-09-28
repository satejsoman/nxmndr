#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Code-orchestrated re-run of the Anaximander case study (three inference paths).

Reproduces, without QGIS, what the plugin does for the agricultural field
delineation case study:

  1. capture  -- build the 1024x1024 EPSG:3857 input the plugin would capture from
                 its "Bing Maps" XYZ layer (same tile source as the QGIS project),
                 georeferenced exactly like the plugin's ``center_1024`` capture.
  2. gpt      -- cloud VLM path: POST the capture to the nxmndr Azure/OpenAI proxy
                 (``/openai/deployments/<name>/images/edits``), the same request the
                 Quick Labels widget sends, using the paper's instance-segmentation prompt.
  3. sam3     -- remote-GPU path: LoadModel(facebook/sam3, HuggingFace) + Predict per
                 512-px chip over gRPC, exactly the client-side tiling the plugin does.
  4. dam      -- local-CPU path: DelineateAnything (Ultralytics YOLO) per 512-px chip,
                 the plugin's ``_predict_tile_yolo`` logic ported verbatim.
  5. render   -- the four Figure-4 panels (input / gpt / sam3 / dam) + a 2x2 grid, and
                 georeferenced rasters that load straight into QGIS.

The harness can spawn its own ``nxmndr-server --azure-proxy`` (default) or be pointed
at existing endpoints (``--grpc-endpoint``, ``--proxy-url``), e.g. a JupyterHub-forwarded
GPU server for SAM 3.  Every stage is skippable/reusable, so re-rendering never
re-bills the API.

Example (server venv, from the ``nxmndr/nxmndr`` directory):

    export OPENAI_API_KEY=sk-...            # or Azure: az login + VISION_ENDPOINT_URL
    export HF_TOKEN=hf_...                  # facebook/sam3 is gated
    python examples/case_study_harness.py \\
        --dam-weights ../models/DelineateAnything.pt \\
        --out ../../anaximander/nxmndr_outputs/case_study

Outputs land in ``<out>/<timestamp>/`` with a ``manifest.json``.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

# Case-study extent (EPSG:3857) derived from the paper's Figure-4 crop: Overberg,
# Western Cape (centre 20.1537E, 34.2254S).  9.33 km in Mercator metres = 7.7 km ground.
DEFAULT_EXTENT = (2238831.0, -4063768.0, 2248162.0, -4054438.0)  # xmin, ymin, xmax, ymax
CAPTURE_SIZE = 1024
CHIP_SIZE = 512

# Same XYZ source as the "Bing Maps" layer in anaximander/nxmndr.qgz (quadkey scheme).
BING_TILE_URL = "http://ecn.t3.tiles.virtualearth.net/tiles/a{q}.jpeg?g=1"
TILE_PX = 256
MERC_HALF = 20037508.342789244
EPSG3857_WKT = (
    'PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",'
    'SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],'
    'UNIT["degree",0.0174532925199433]],PROJECTION["Mercator_1SP"],'
    'PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],'
    'PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1],'
    'AUTHORITY["EPSG","3857"]]'
)

# Prompt used for the paper's Quick Labels run (appendix "Case Study Details").
PAPER_PROMPT = (
    "You are an instance-segmentation model. Given the satellite image provided, output a "
    "PNG instance-segmentation mask with the same resolution as the input, where every "
    "distinct agricultural field parcel is detected and assigned its own unique solid RGB "
    "color. Each field must be represented as a separate instance, even if adjacent or "
    "similar in appearance. Colors must be visually distinct, with no transparency, "
    "blending, labels, text, borders, or legend; output only the mask. Each pixel in the "
    "image must belong either to one field instance or to a single neutral background color "
    "such as black (#000000) that is not reused for any instance. Return only the generated "
    "segmentation mask image."
)

DEFAULT_SAM_REPO = "facebook/sam3"


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[harness {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def write_world_file(path: Path, extent: Tuple[float, float, float, float], w: int, h: int) -> None:
    xmin, ymin, xmax, ymax = extent
    px = (xmax - xmin) / w
    py = (ymax - ymin) / h
    # ESRI world file: pixel-centre origin
    path.write_text(f"{px:.10f}\n0\n0\n{-py:.10f}\n{xmin + px / 2:.6f}\n{ymax - py / 2:.6f}\n")


def save_georeferenced(
    array: np.ndarray, stem: Path, extent: Tuple[float, float, float, float], nodata=None
) -> Path:
    """Write ``array`` (H,W) or (H,W,3) as GeoTIFF via rasterio if available, else PNG+PGW+PRJ."""
    h, w = array.shape[:2]
    try:
        import rasterio  # type: ignore
        from rasterio.transform import from_bounds  # type: ignore

        count = 1 if array.ndim == 2 else array.shape[2]
        data = array[None, ...] if array.ndim == 2 else np.transpose(array, (2, 0, 1))
        out = stem.with_suffix(".tif")
        with rasterio.open(
            out, "w", driver="GTiff", height=h, width=w, count=count, dtype=data.dtype,
            crs="EPSG:3857", transform=from_bounds(*extent, w, h), nodata=nodata, compress="deflate",
        ) as dst:
            dst.write(data)
        return out
    except Exception:
        out = stem.with_suffix(".png")
        if array.ndim == 2 and array.dtype == np.uint16:
            Image.fromarray(array, mode="I;16").save(out)
        else:
            Image.fromarray(array).save(out)
        write_world_file(stem.with_suffix(".pgw"), extent, w, h)
        stem.with_suffix(".prj").write_text(EPSG3857_WKT)
        return out


def palette(n: int, seed: int = 7) -> np.ndarray:
    """Deterministic saturated colours for instance ids 1..n (index 0 = black)."""
    rng = np.random.default_rng(seed)
    hsv = np.stack([rng.random(n), 0.65 + 0.35 * rng.random(n), 0.75 + 0.25 * rng.random(n)], 1)
    rgb = np.zeros((n + 1, 3), dtype=np.uint8)
    for i, (hh, ss, vv) in enumerate(hsv, start=1):
        r, g, b = _hsv_to_rgb(hh, ss, vv)
        rgb[i] = (int(r * 255), int(g * 255), int(b * 255))
    return rgb


def _hsv_to_rgb(h, s, v):
    i = int(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i % 6]


def instance_outline(label_map: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels whose 4-neighbourhood contains a different label."""
    lm = label_map
    edge = np.zeros_like(lm, dtype=bool)
    edge[:, 1:] |= lm[:, 1:] != lm[:, :-1]
    edge[1:, :] |= lm[1:, :] != lm[:-1, :]
    edge[:, :-1] |= lm[:, :-1] != lm[:, 1:]
    edge[:-1, :] |= lm[:-1, :] != lm[1:, :]
    return edge


# --------------------------------------------------------------------------------------
# Stage 1: capture (Bing XYZ tiles -> plugin-style 1024x1024 georeferenced input)
# --------------------------------------------------------------------------------------


def quadkey(x: int, y: int, z: int) -> str:
    q = []
    for i in range(z, 0, -1):
        d, m = 0, 1 << (i - 1)
        if x & m:
            d += 1
        if y & m:
            d += 2
        q.append(str(d))
    return "".join(q)


def merc_to_tile_px(xm: float, ym: float, z: int) -> Tuple[float, float]:
    """Web-Mercator metres -> global pixel coords at zoom z (origin top-left)."""
    n = TILE_PX * (1 << z)
    return (xm + MERC_HALF) / (2 * MERC_HALF) * n, (MERC_HALF - ym) / (2 * MERC_HALF) * n


def fetch_bing_capture(
    extent: Tuple[float, float, float, float], size: int, zoom: Optional[int], cache_dir: Path
) -> Tuple[Image.Image, int]:
    xmin, ymin, xmax, ymax = extent
    target_res = (xmax - xmin) / size  # Mercator m/px wanted
    if zoom is None:
        # smallest zoom whose native resolution is finer than the target (then downsample)
        zoom = 1
        while (2 * MERC_HALF) / (TILE_PX * (1 << zoom)) > target_res and zoom < 19:
            zoom += 1
    px0, py0 = merc_to_tile_px(xmin, ymax, zoom)
    px1, py1 = merc_to_tile_px(xmax, ymin, zoom)
    tx0, ty0 = int(px0 // TILE_PX), int(py0 // TILE_PX)
    tx1, ty1 = int(px1 // TILE_PX), int(py1 // TILE_PX)
    cache_dir.mkdir(parents=True, exist_ok=True)
    mosaic = Image.new("RGB", ((tx1 - tx0 + 1) * TILE_PX, (ty1 - ty0 + 1) * TILE_PX))
    sess = requests.Session()
    sess.headers["User-Agent"] = "nxmndr-case-study-harness/1.0 (+QGIS XYZ layer equivalent)"
    n = 0
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            q = quadkey(tx, ty, zoom)
            f = cache_dir / f"a{q}.jpeg"
            if not f.exists():
                r = sess.get(BING_TILE_URL.format(q=q), timeout=30)
                r.raise_for_status()
                f.write_bytes(r.content)
                n += 1
            mosaic.paste(Image.open(f).convert("RGB"), ((tx - tx0) * TILE_PX, (ty - ty0) * TILE_PX))
    log(f"capture: zoom {zoom}, {(tx1-tx0+1)*(ty1-ty0+1)} tiles ({n} fetched), "
        f"native {(2*MERC_HALF)/(TILE_PX*(1<<zoom)):.2f} m/px -> target {target_res:.2f} m/px")
    # sub-pixel crop of the mosaic to the exact extent, then resample to size x size.
    # Image.transform(EXTENT) only supports NEAREST/BILINEAR/BICUBIC, so crop at native
    # resolution first (BICUBIC, ~1:1) and do the ~2x downsample with LANCZOS afterwards.
    box = (px0 - tx0 * TILE_PX, py0 - ty0 * TILE_PX, px1 - tx0 * TILE_PX, py1 - ty0 * TILE_PX)
    native = (int(round(box[2] - box[0])), int(round(box[3] - box[1])))
    img = mosaic.transform(native, Image.Transform.EXTENT, box, Image.Resampling.BICUBIC)
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return img, zoom


def load_capture(path: Path) -> Tuple[Image.Image, Tuple[float, float, float, float]]:
    """Use an existing plugin capture (GeoTIFF or PNG+PGW) instead of fetching tiles."""
    src = Image.open(path)
    tags = getattr(src, "tag_v2", None)  # read GeoTIFF tags before convert() drops them
    img = src.convert("RGB")
    w, h = img.size
    if tags and 33550 in tags and 33922 in tags:
        sx, sy = tags[33550][0], tags[33550][1]
        x0, y0 = tags[33922][3], tags[33922][4]
        return img, (x0, y0 - sy * h, x0 + sx * w, y0)
    pgw = path.with_suffix(".pgw")
    if pgw.exists():
        a, _, _, d, cx, cy = [float(v) for v in pgw.read_text().split()]
        x0, y0 = cx - a / 2, cy - d / 2  # d is negative
        return img, (x0, y0 + d * h, x0 + a * w, y0)
    raise SystemExit(f"{path}: no georeferencing found (need GeoTIFF tags or a .pgw)")


# --------------------------------------------------------------------------------------
# Stage 2: local nxmndr-server lifecycle
# --------------------------------------------------------------------------------------


class ManagedServer:
    """Spawn ``nxmndr-server --azure-proxy`` in this interpreter's environment."""

    def __init__(self, grpc_port: int, http_port: int, log_path: Path, env: Dict[str, str]):
        self.grpc_port, self.http_port, self.log_path, self.env = grpc_port, http_port, log_path, env
        self.proc: Optional[subprocess.Popen] = None

    def start(self, timeout: float = 240.0) -> None:
        exe = shutil.which("nxmndr-server")
        cmd = [exe] if exe else [sys.executable, "-m", "nxmndr.server.server"]
        cmd += ["--grpc-port", str(self.grpc_port), "--http-port", str(self.http_port),
                "--host", "127.0.0.1", "--azure-proxy"]
        log(f"server: starting {' '.join(cmd)}  (log: {self.log_path})")
        self.proc = subprocess.Popen(cmd, env=self.env, stdout=open(self.log_path, "w"),
                                     stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise SystemExit(f"server exited early (code {self.proc.returncode}); see {self.log_path}")
            ok_http = ok_grpc = False
            try:
                ok_http = requests.get(f"http://127.0.0.1:{self.http_port}/health", timeout=2).status_code == 200
            except Exception:
                pass
            try:
                with socket.create_connection(("127.0.0.1", self.grpc_port), timeout=1):
                    ok_grpc = True
            except OSError:
                pass
            if ok_http and ok_grpc:
                log(f"server: ready after {time.time() - t0:.1f}s (grpc {self.grpc_port}, http {self.http_port})")
                return
            time.sleep(1.0)
        raise SystemExit(f"server did not become ready within {timeout}s; see {self.log_path}")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            log("server: stopped")


# --------------------------------------------------------------------------------------
# Stage 3: gpt-image path (via the proxy, identical to the Quick Labels request)
# --------------------------------------------------------------------------------------


def run_gpt(proxy_url: str, deployment: str, image: Image.Image, prompt: str, timeout: float) -> Tuple[Image.Image, dict]:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    url = f"{proxy_url.rstrip('/')}/openai/deployments/{deployment}/images/edits"
    log(f"gpt: POST {url} ({len(buf.getvalue())} bytes, size 1024x1024)")
    t0 = time.time()
    r = requests.post(url, files={"image": ("input.png", buf.getvalue(), "image/png")},
                      data={"prompt": prompt, "n": 1, "size": "1024x1024"}, timeout=timeout)
    dt = time.time() - t0
    if r.status_code != 200:
        raise SystemExit(f"gpt: HTTP {r.status_code}: {r.text[:800]}")
    payload = r.json()
    item = payload["data"][0]
    if "b64_json" in item:
        out = Image.open(io.BytesIO(base64.b64decode(item["b64_json"])))
    else:
        out = Image.open(io.BytesIO(requests.get(item["url"], timeout=60).content))
    out = out.convert("RGB")
    log(f"gpt: {out.size} image in {dt:.1f}s; usage={payload.get('usage')}")
    return out, {"seconds": round(dt, 1), "usage": payload.get("usage"), "size": out.size}


# --------------------------------------------------------------------------------------
# Stage 4: SAM 3 over gRPC (client-side 512-px tiling, as the plugin does)
# --------------------------------------------------------------------------------------


def run_sam3(endpoint: str, image: np.ndarray, repo: str, token: Optional[str], chip: int,
             timeout: float) -> Tuple[np.ndarray, dict]:
    from nxmndr import client as nx_client  # server package (grpc stubs)

    cli = nx_client.InferenceGrpcClient(endpoint, timeout=timeout)
    spec = {"format": "huggingface", "source": repo, "name": repo, "task": "segmentation"}
    if token:
        spec["token"] = token
    t0 = time.time()
    model_id = cli.load_model(model_id=repo, spec=spec)
    log(f"sam3: model loaded on {endpoint} as '{model_id}' in {time.time() - t0:.1f}s")
    h, w = image.shape[:2]
    label = np.zeros((h, w), dtype=np.uint16)
    next_id, n_masks, times = 1, 0, []
    for y in range(0, h, chip):
        for x in range(0, w, chip):
            tile = np.ascontiguousarray(image[y:y + chip, x:x + chip, :3])
            t1 = time.time()
            res = cli.predict(model_id=model_id, tensor=tile, options={})
            times.append(time.time() - t1)
            out = np.frombuffer(res.output, dtype=np.dtype(res.dtype)).reshape(res.shape)
            if out.ndim == 3 and out.shape[0] < out.shape[1]:  # (N, h, w) masks
                for m in out:
                    sel = m > 0.5 if np.issubdtype(m.dtype, np.floating) else m > 0
                    if sel.any():
                        label[y:y + chip, x:x + chip][sel & (label[y:y + chip, x:x + chip] == 0)] = next_id
                        next_id += 1
                        n_masks += 1
            elif out.ndim == 2:  # already a label map
                sub = label[y:y + chip, x:x + chip]
                nz = out > 0
                sub[nz] = out[nz].astype(np.uint16) + (next_id - 1)
                next_id += int(out.max())
                n_masks += int(out.max())
            log(f"sam3: tile ({y},{x}) -> {out.shape} {out.dtype} in {times[-1]:.1f}s")
    cli.close()
    return label, {"model_id": model_id, "masks": n_masks, "tile_seconds": [round(t, 1) for t in times]}


# --------------------------------------------------------------------------------------
# Stage 5: DelineateAnything (Ultralytics YOLO) -- plugin's _predict_tile_yolo, ported
# --------------------------------------------------------------------------------------


def run_dam(weights: Path, image: np.ndarray, chip: int, device: str) -> Tuple[np.ndarray, dict]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    h, w = image.shape[:2]
    label = np.zeros((h, w), dtype=np.uint16)
    next_id, times = 1, []
    for y in range(0, h, chip):
        for x in range(0, w, chip):
            tile = np.ascontiguousarray(image[y:y + chip, x:x + chip, :3].astype(np.uint8))
            t1 = time.time()
            try:
                results = model.predict(tile, verbose=False, retina_masks=True, device=device)
            except TypeError:
                results = model.predict(tile, verbose=False, device=device)
            times.append(time.time() - t1)
            masks = getattr(results[0], "masks", None) if results else None
            data = getattr(masks, "data", None) if masks is not None else None
            if data is None:
                log(f"dam: tile ({y},{x}) -> no masks")
                continue
            arr = data.detach().cpu().numpy()
            th, tw = tile.shape[:2]
            sub = label[y:y + th, x:x + tw]
            for m in arr:
                if m.shape != (th, tw):  # nearest-neighbour resize, as the plugin does
                    m = np.asarray(Image.fromarray(m.astype(np.float32)).resize((tw, th), Image.Resampling.NEAREST))
                sel = m > 0.5
                if sel.any():
                    sub[sel] = next_id  # last-write-wins, globally unique ids
                    next_id += 1
            log(f"dam: tile ({y},{x}) -> {arr.shape[0]} masks in {times[-1]:.1f}s")
    return label, {"instances": int(next_id - 1), "tile_seconds": [round(t, 1) for t in times]}


# --------------------------------------------------------------------------------------
# Stage 6: rendering (Figure-4 style panels)
# --------------------------------------------------------------------------------------


def render_overlay(base: np.ndarray, label: np.ndarray, alpha: float, outline: bool) -> Image.Image:
    n = int(label.max())
    colors = palette(max(n, 1))
    rgb = base.astype(np.float32)
    col = colors[np.clip(label, 0, n)].astype(np.float32)
    mask = label > 0
    rgb[mask] = (1 - alpha) * rgb[mask] + alpha * col[mask]
    if outline:
        rgb[instance_outline(label) & mask] = 0
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def render_grid(panels: List[Tuple[str, Image.Image]], out: Path, size: int = 1024, pad: int = 24) -> None:
    cols = 2
    rows = math.ceil(len(panels) / cols)
    grid = Image.new("RGB", (cols * size + (cols + 1) * pad, rows * (size + 48) + (rows + 1) * pad), "white")
    draw = ImageDraw.Draw(grid)
    font = None
    for name in ("DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(name, 36)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default(size=36) if "size" in ImageFont.load_default.__code__.co_varnames else ImageFont.load_default()
    for i, (title, img) in enumerate(panels):
        r, c = divmod(i, cols)
        x = pad + c * (size + pad)
        y = pad + r * (size + 48 + pad)
        draw.text((x, y), title, fill="black", font=font)
        grid.paste(img.resize((size, size), Image.Resampling.LANCZOS), (x, y + 48))
    grid.save(out)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


@dataclass
class Manifest:
    created: str
    extent_3857: Tuple[float, float, float, float]
    capture: dict
    gpt: Optional[dict] = None
    sam3: Optional[dict] = None
    dam: Optional[dict] = None
    endpoints: Optional[dict] = None
    platform: str = platform.platform()
    python: str = sys.version.split()[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("case_study_runs"), help="output root (a timestamped subdir is created)")
    ap.add_argument("--extent", type=float, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"), default=DEFAULT_EXTENT, help="EPSG:3857 extent")
    ap.add_argument("--capture", type=Path, help="reuse an existing georeferenced capture (GeoTIFF or PNG+PGW) instead of fetching Bing tiles")
    ap.add_argument("--zoom", type=int, help="Bing zoom level (default: auto, finest >= target resolution)")
    ap.add_argument("--tile-cache", type=Path, default=Path.home() / ".cache" / "nxmndr" / "bing_tiles")
    ap.add_argument("--paths", default="gpt,sam3,dam", help="comma list of paths to run (gpt,sam3,dam); rendering always runs")
    # gpt
    ap.add_argument("--proxy-url", help="existing proxy base URL (default: spawn a local server with --azure-proxy)")
    ap.add_argument("--deployment", default=os.environ.get("VISION_DEPLOYMENT_NAME", "gpt-image-1"), help="deployment/model name in the proxy URL")
    ap.add_argument("--prompt-file", type=Path, help="override the paper prompt")
    ap.add_argument("--gpt-timeout", type=float, default=300.0)
    # sam3
    ap.add_argument("--grpc-endpoint", help="existing nxmndr gRPC endpoint host:port for SAM 3 (default: spawn locally)")
    ap.add_argument("--sam-repo", default=DEFAULT_SAM_REPO)
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"))
    ap.add_argument("--sam-timeout", type=float, default=900.0)
    # dam
    ap.add_argument("--dam-weights", type=Path, default=Path("../models/DelineateAnything.pt"))
    ap.add_argument("--dam-device", default="cpu", help="ultralytics device: cpu, mps, cuda:0 ...")
    ap.add_argument("--chip", type=int, default=CHIP_SIZE, help="chip size for sam3/dam tiling (paper: 512 -> 4 tiles)")
    ap.add_argument("--reuse", type=Path, help="previous run dir: reuse its capture/outputs for paths not in --paths")
    args = ap.parse_args()

    paths = {p.strip() for p in args.paths.split(",") if p.strip()}
    run_dir = args.out / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"run dir: {run_dir}")

    # ---- capture ---------------------------------------------------------------------
    extent = tuple(args.extent)
    if args.capture:
        img, extent = load_capture(args.capture)
        cap_meta = {"source": str(args.capture), "mode": "existing"}
        log(f"capture: reusing {args.capture} extent={extent}")
    elif args.reuse and ((args.reuse / "input.tif").exists() or (args.reuse / "input.png").exists()):
        p = args.reuse / ("input.tif" if (args.reuse / "input.tif").exists() else "input.png")
        img, extent = load_capture(p)
        cap_meta = {"source": str(p), "mode": "reuse"}
    else:
        img, zoom = fetch_bing_capture(extent, CAPTURE_SIZE, args.zoom, args.tile_cache)
        cap_meta = {"source": BING_TILE_URL, "mode": "center_1024", "zoom": zoom,
                    "attribution": "Bing Maps aerial imagery (Microsoft) via XYZ tiles"}
    if img.size != (CAPTURE_SIZE, CAPTURE_SIZE):
        img = img.resize((CAPTURE_SIZE, CAPTURE_SIZE), Image.Resampling.LANCZOS)
    base = np.asarray(img, dtype=np.uint8)
    input_path = save_georeferenced(base, run_dir / "input", extent)
    img.save(run_dir / "panel_input.png")
    log(f"capture: wrote {input_path}")

    manifest = Manifest(created=time.strftime("%Y-%m-%dT%H:%M:%S"), extent_3857=extent, capture=cap_meta)
    prompt = args.prompt_file.read_text() if args.prompt_file else PAPER_PROMPT
    prev_manifest: dict = {}
    if args.reuse and (args.reuse / "manifest.json").exists():
        prev_manifest = json.loads((args.reuse / "manifest.json").read_text())

    def reuse_path(key: str, *globs: str) -> Optional[dict]:
        """Copy a previous run's rasters/prompt for ``key`` into run_dir; return its manifest entry."""
        for g in globs:
            for f in args.reuse.glob(g):
                shutil.copy2(f, run_dir / f.name)
        entry = prev_manifest.get(key)
        if isinstance(entry, dict):
            entry = {**entry, "reused_from": str(args.reuse)}
        return entry

    # ---- server -----------------------------------------------------------------------
    need_local = ("gpt" in paths and not args.proxy_url) or ("sam3" in paths and not args.grpc_endpoint)
    server: Optional[ManagedServer] = None
    proxy_url, grpc_endpoint = args.proxy_url, args.grpc_endpoint
    if need_local:
        env = dict(os.environ)
        have_url = env.get("VISION_ENDPOINT_URL") or env.get("ENDPOINT_URL")
        if "gpt" in paths and not args.proxy_url and not have_url:
            # default: OpenAI platform through the proxy (see docs/azure_setup.md); the proxy's
            # discovery needs <PREFIX>_ENDPOINT_URL, so key on that rather than on a deployment name
            if not env.get("OPENAI_API_KEY"):
                raise SystemExit("gpt path: set OPENAI_API_KEY (or VISION_* Azure vars) or pass --proxy-url")
            env.update({"VISION_DEPLOYMENT_NAME": args.deployment, "VISION_ENDPOINT_URL": "https://api.openai.com",
                        "VISION_TYPE": "vision"})
        if args.hf_token:
            env.setdefault("HF_TOKEN", args.hf_token)
        server = ManagedServer(free_port(), free_port(), run_dir / "server.log", env)
        server.start()
        proxy_url = proxy_url or f"http://127.0.0.1:{server.http_port}"
        grpc_endpoint = grpc_endpoint or f"127.0.0.1:{server.grpc_port}"
    manifest.endpoints = {"proxy_url": proxy_url, "grpc_endpoint": grpc_endpoint}

    panels: List[Tuple[str, Image.Image]] = [("Input (Bing Maps)", img)]
    try:
        # ---- gpt ------------------------------------------------------------------------
        gpt_img = None
        if "gpt" in paths:
            gpt_img, meta = run_gpt(proxy_url, args.deployment, img, prompt, args.gpt_timeout)
            gpt_img = gpt_img.resize((CAPTURE_SIZE, CAPTURE_SIZE), Image.Resampling.NEAREST)
            save_georeferenced(np.asarray(gpt_img), run_dir / f"quick_labels_{args.deployment}", extent)
            (run_dir / "prompt.txt").write_text(prompt)
            manifest.gpt = {"deployment": args.deployment, **meta}
        elif args.reuse and (args.reuse / "panel_gpt.png").exists():
            gpt_img = Image.open(args.reuse / "panel_gpt.png").convert("RGB")
            manifest.gpt = reuse_path("gpt", "quick_labels_*", "prompt.txt")
        if gpt_img is not None:
            gpt_img.save(run_dir / "panel_gpt.png")
            panels.append((f"{args.deployment} (cloud API)", gpt_img))

        # ---- sam3 -----------------------------------------------------------------------
        sam_label = None
        if "sam3" in paths:
            sam_label, meta = run_sam3(grpc_endpoint, base, args.sam_repo, args.hf_token, args.chip, args.sam_timeout)
            save_georeferenced(sam_label, run_dir / "sam3_instances", extent, nodata=0)
            manifest.sam3 = {"repo": args.sam_repo, "endpoint": grpc_endpoint, "chip": args.chip, **meta}
        elif args.reuse and (args.reuse / "panel_sam3.png").exists():
            panels.append(("SAM 3 (remote GPU)", Image.open(args.reuse / "panel_sam3.png").convert("RGB")))
            manifest.sam3 = reuse_path("sam3", "sam3_instances.*", "panel_sam3.png")
        if sam_label is not None:
            p = render_overlay(base, sam_label, alpha=0.55, outline=False)
            p.save(run_dir / "panel_sam3.png")
            panels.append(("SAM 3 (remote GPU)", p))

        # ---- dam ------------------------------------------------------------------------
        dam_label = None
        if "dam" in paths:
            dam_label, meta = run_dam(args.dam_weights, base, args.chip, args.dam_device)
            save_georeferenced(dam_label, run_dir / "delineate_anything_instances", extent, nodata=0)
            manifest.dam = {"weights": str(args.dam_weights), "device": args.dam_device, "chip": args.chip, **meta}
        elif args.reuse and (args.reuse / "panel_dam.png").exists():
            panels.append(("DelineateAnything (local CPU)", Image.open(args.reuse / "panel_dam.png").convert("RGB")))
            manifest.dam = reuse_path("dam", "delineate_anything_instances.*", "panel_dam.png")
        if dam_label is not None:
            p = render_overlay(base, dam_label, alpha=0.85, outline=True)
            p.save(run_dir / "panel_dam.png")
            panels.append(("DelineateAnything (local CPU)", p))
    finally:
        if server:
            server.stop()

    render_grid(panels, run_dir / "figure4_grid.png")
    (run_dir / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2, default=str))
    log(f"done: {len(panels)} panels -> {run_dir / 'figure4_grid.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
