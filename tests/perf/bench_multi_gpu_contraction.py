"""Multi-GPU scaling of the deposit-fold beamlet contraction: a self-contained benchmark.

Answers the question left open in ``docs/decisions.md`` ("Contraction by deposit fold",
"Several devices"): what ``run_beamlet_contraction(..., fold="deposit", devices=[...])``
gains on several identical GPUs, and how the engine's group rule (one group per device,
more once a group would exceed ``WarpEngine.fold_group_histories``) compares with fixed
numbers of groups per device. The rule's default comes from this script's first runs.
A manual script, not a pytest benchmark (no ``test_`` prefix, never collected): it
needs the hardware it measures and runs for tens of minutes.

**Problem.** Synthetic and deterministic from ``--seed``; public API only, no data files
(analytic water cross sections, the analytic Ali-Rogers 6 MV spectrum). A 32 x 21 x 28 cm
thorax-like phantom on 0.25 cm voxels (1.20 M voxels; water at 1.0 g/cm^3 in an
elliptic body, two lungs at 0.26, a spine at 1.6, outside 0.0012: the analytic source is
water-only, so every medium is water at its density), a 90-angle coplanar arc (4 deg)
through a single-layer MLC (5 mm leaves at the isocentre; leaves of "water" at
19.3 g/cm^3, the analytic stand-in for tungsten), and a ``SegmentedEdgeSource`` over it:
every leaf tip of every control point is one beamlet. Two sizes stand in for the study's
two 4-degree cases:

- ``large``: 39 leaf pairs, 7020 edges, a 104 x 66 x 92 scoring box (631 488 voxels)
  - the study's 1THA232: 7020 edges on 628 524 voxels, 0.155 s per call at 1000
  histories per edge on an RTX 4070 Laptop;
- ``small``: 17 leaf pairs, 3060 edges, a 60 x 46 x 46 box (126 960 voxels) - 1THA210:
  3060 edges on 128 340 voxels, 0.065 s.

The voxel weights are smooth and signed, like an objective gradient (an underdosed
target, an overdosed organ at risk, a weak normal-tissue term), and zero in "air".

**Arms**, per problem and histories per edge: the deposit fold on 1, 2, ..., N GPUs
(``cuda:0`` .. ``cuda:N-1``) at the engine's own group size ("auto"); for more than one
device also at each ``--groups-per-device`` value G, passed as an explicit
``beamlet_group_size = ceil(edges / (devices * G))``. Every call's actual group size is
read back and compared with that; the engine still bounds an explicit group by its
energy books, which shows as ``GROUP BOUNDED``. The dense fold on one
GPU once per size as the baseline (dropped with a note if its warm-up exceeds
``--dense-max-seconds`` or fails). ``--include-cpu`` adds ``cpu`` to every device set;
off by default: a cpu shard that pulls a group holds the whole call hostage, so those
arms run only up to ``--cpu-max-histories``.

**Protocol.** One engine per problem. Every arm is warmed up (``--warmup`` calls: kernel
compile and first-call set-up on every device), then ``--repeats`` rounds run every arm
in turn (round-robin inside the process, so drift hits all arms alike). The engine keeps
one pool of transport queues per device, sized to the last call's group; alternating
arms of different group sizes would refit it on every call, which an optimizer calling
one configuration never pays, so each timed call is preceded by ``--settle`` untimed
calls of the same arm (default 1; ``--settle 0`` times the switching cost too). The
clock stops after ``warp.synchronize()`` (the result's arrays are host NumPy arrays, so
the call has already waited for its devices; the synchronize only makes that explicit).

**Checks.** Environment (pyradmc file and git commit, Warp, driver, GPUs); nvidia-smi
clocks, temperature, utilization, memory and throttle reasons before, during (a sampler
thread, ``--sample-interval``) and after; a warning if a GPU is in use at the start.
Per call: transport passes, group size and count, which device ran which groups (from
outside the engine: the engine's per-device shard method is wrapped on the instance, so
every group a device pulls from the shared queue is recorded with its time), and the
provenance fold fields. Values, sigma and energy books of every multi-GPU arm are
compared bitwise with the 1-GPU result of the same round and seed: group size is
bit-inert and identical GPUs should be too, so a mismatch is a finding (reported with
the largest relative difference), not a failure. Arms with a cpu shard that transported
a group are expected to differ (no cross-target bit identity, AGENTS.md 2.3).

Writes a JSON file (``--out``) with every per-repeat time and the environment record,
rewritten after every block so an interrupted run keeps what it measured. Run::

    python tests/perf/bench_multi_gpu_contraction.py --quick   # about a minute
    python tests/perf/bench_multi_gpu_contraction.py           # the full sweep

The checkout this file lives in is put first on ``sys.path`` (``--installed`` keeps the
installed ``pyradmc`` instead), so the benchmark measures the code next to it.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]

# -- the synthetic problem -------------------------------------------------------------

SAD = 100.0  # focal spot to isocentre (cm)
GRID_SHAPE = (128, 84, 112)
VOXEL = 0.25  # cm
N_ANGLES = 90  # 4 deg arc
LEAF_WIDTH = 0.5  # cm at the isocentre
MLC_Z = (50.0, 56.0)  # slab from the focal spot (cm)
TO_MID = 0.5 * sum(MLC_Z) / SAD  # isocentre plane -> MLC mid-plane
MLC_DENSITY = 19.3  # water at tungsten density: the analytic source is water-only
AIR_DENSITY = 0.0012
LUNG_DENSITY = 0.26
BONE_DENSITY = 1.6


@dataclass(frozen=True)
class Problem:
    """One synthetic problem size and the study case it stands in for."""

    name: str
    n_pairs: int
    field_u: float  # cm at the isocentre, along leaf travel
    box_shape: tuple[int, int, int]
    target: tuple[float, float, float]  # target semi-axes (cm)
    study: str


PROBLEMS = {
    "large": Problem(
        "large", 39, 20.0, (104, 66, 92), (5.0, 7.0, 4.5),
        "1THA232 4 deg: 7020 edges, 628 524 box voxels, 0.155 s at 1000/edge (RTX 4070 Laptop)",
    ),
    "small": Problem(
        "small", 17, 10.0, (60, 46, 46), (2.5, 3.0, 2.5),
        "1THA210 4 deg: 3060 edges, 128 340 box voxels, 0.065 s at 1000/edge (RTX 4070 Laptop)",
    ),
}  # fmt: skip


def _axes(angle_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(u, v, w)`` of the beam at a gantry angle (rotation about +y)."""
    a = math.radians(angle_deg)
    w = np.array([math.sin(a), 0.0, math.cos(a)])
    u = np.array([math.cos(a), 0.0, -math.sin(a)])
    return u, np.array([0.0, 1.0, 0.0]), w


def _centres(shape: Sequence[int], origin: Sequence[float]) -> list[np.ndarray]:
    return [o + (np.arange(n) + 0.5) * VOXEL for n, o in zip(shape, origin, strict=True)]


def _phantom_density() -> np.ndarray:
    """Elliptic body, two lungs, a spine; everything else air-density water."""
    origin = [-0.5 * n * VOXEL for n in GRID_SHAPE]
    x, y, z = np.meshgrid(*_centres(GRID_SHAPE, origin), indexing="ij")
    body = (x / 15.0) ** 2 + (z / 11.0) ** 2 <= 1.0
    lungs = (((np.abs(x) - 6.5) / 4.5) ** 2 + (y / 8.0) ** 2 + ((z + 0.5) / 6.5) ** 2 <= 1.0) & body
    spine = (x**2 + (z - 8.0) ** 2 <= 1.5**2) & body
    density = np.where(body, 1.0, AIR_DENSITY)
    density[lungs] = LUNG_DENSITY
    density[spine] = BONE_DENSITY
    return density


def _tips(problem: Problem, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Per-angle apertures around an elliptic target, at the MLC mid-plane: (A, P) each."""
    theta = np.radians(np.arange(N_ANGLES) * 360.0 / N_ANGLES)[:, None]
    v = (np.arange(problem.n_pairs) + 0.5 - 0.5 * problem.n_pairs) * LEAF_WIDTH
    v_half = 0.45 * problem.n_pairs * LEAF_WIDTH
    centre = 0.08 * problem.field_u * np.sin(theta)
    shape = np.sqrt(np.clip(1.0 - (v[None, :] / v_half) ** 2, 0.0, 1.0))
    half = 0.35 * problem.field_u * shape * (1.0 + 0.15 * np.cos(2.0 * theta))
    half = half + 0.3 * rng.standard_normal(half.shape)
    limit = 0.5 * problem.field_u - 0.2
    half = np.clip(half, 0.3, limit - np.abs(centre))
    return (centre - half) * TO_MID, (centre + half) * TO_MID


def _weights(problem: Problem, scoring: Any, box_origin: Sequence[float]) -> np.ndarray:
    """Smooth signed weights, like dL/dD: target underdosed, an OAR overdosed, normal tissue."""
    x, y, z = np.meshgrid(*_centres(problem.box_shape, box_origin), indexing="ij")
    ax, ay, az = problem.target
    r2 = ((x - 1.0) / ax) ** 2 + (y / ay) ** 2 + ((z + 1.0) / az) ** 2
    w = -np.exp(-r2) * (1.0 + 0.25 * np.sin(0.9 * x + 0.4 * y))
    oar = ((x - 1.0 + 1.4 * ax) ** 2 + (z + 1.0) ** 2) / 2.0 + (y / ay) ** 2 / 2.0  # beside it
    w += 0.4 * np.exp(-oar)
    w += 0.05 * (1.0 - np.exp(-r2 / 9.0))
    mass = np.asarray(scoring.voxel_mass).reshape(problem.box_shape)
    w[mass < 0.01 * VOXEL**3] = 0.0  # no gradient in air
    return np.ascontiguousarray(w, dtype=np.float64)


def build_problem(problem: Problem, seed: int) -> dict[str, Any]:
    """Grid, cross sections, scoring box, weights and edge source of one problem."""
    from pyradmc.data.analytic import AnalyticCrossSections
    from pyradmc.data.materials import WATER
    from pyradmc.geometry.collimation import MLC, BeamFrame, BeamLimitingStack
    from pyradmc.geometry.grid import VoxelGrid
    from pyradmc.geometry.segment_edges import SegmentedEdgeSource
    from pyradmc.geometry.segments import SegmentedFanSource
    from pyradmc.geometry.source import SpectralBeamSource
    from pyradmc.geometry.spectrum import ali_rogers_mv
    from pyradmc.scoring.grid import ScoringGrid

    origin = tuple(-0.5 * n * VOXEL for n in GRID_SHAPE)
    grid = VoxelGrid(
        shape=GRID_SHAPE,
        spacing=(VOXEL, VOXEL, VOXEL),
        density=_phantom_density(),
        material=np.full(GRID_SHAPE, WATER, dtype=np.int32),
        origin=origin,
    )
    xs = AnalyticCrossSections(geometry_densities=grid.max_density_by_material())
    low = [(n - b) // 2 for n, b in zip(GRID_SHAPE, problem.box_shape, strict=True)]
    box_origin = tuple(o + lo * VOXEL for o, lo in zip(origin, low, strict=True))
    scoring = ScoringGrid.rebin(grid, problem.box_shape, (VOXEL, VOXEL, VOXEL), box_origin)
    weights = _weights(problem, scoring, box_origin)

    spectrum = ali_rogers_mv("varian-6mv")
    width_v = problem.n_pairs * LEAF_WIDTH
    leaf_edges = tuple(float(e) for e in np.linspace(-0.5, 0.5, problem.n_pairs + 1) * width_v)
    leaf_edges = tuple(e * TO_MID for e in leaf_edges)
    tips_neg, tips_pos = _tips(problem, np.random.default_rng(seed))
    angles = np.arange(N_ANGLES) * 360.0 / N_ANGLES
    fans, frames = [], []
    for angle in angles:
        u, v, w = _axes(float(angle))
        focal = tuple(-SAD * w)  # isocentre at the origin
        fans.append(
            SpectralBeamSource(
                spectrum, focal_point=focal, center=(0.0, 0.0, 0.0), width_u=problem.field_u,
                width_v=width_v, u_axis=tuple(u), v_axis=tuple(v),
            )
        )  # fmt: skip
        frames.append(BeamFrame(origin=focal, u_axis=tuple(u), v_axis=tuple(v), w_axis=tuple(w)))
    template = BeamLimitingStack(
        frame=frames[0],
        devices=(
            MLC(
                z_top=MLC_Z[0], z_bottom=MLC_Z[1], leaf_edges_v=leaf_edges,
                tips_neg=tuple(float(t) for t in tips_neg[0]),
                tips_pos=tuple(float(t) for t in tips_pos[0]),
                tip_radius=8.0, material=WATER, density=MLC_DENSITY, focused_sides=True,
            ),
        ),
    )  # fmt: skip
    segments = SegmentedFanSource(
        fans, template, xs, frames=frames, tips_neg=tips_neg, tips_pos=tips_pos
    )
    source = SegmentedEdgeSource(segments)
    return {
        "grid": grid,
        "xs": xs,
        "scoring": scoring,
        "weights": weights,
        "source": source,
        "record": {
            "name": problem.name,
            "stands_in_for": problem.study,
            "edges": int(source.n_beamlets),
            "angles": N_ANGLES,
            "leaf_pairs": problem.n_pairs,
            "transport_grid": list(GRID_SHAPE),
            "transport_voxels": int(np.prod(GRID_SHAPE)),
            "voxel_cm": VOXEL,
            "scoring_box": list(problem.box_shape),
            "scoring_voxels": int(np.prod(problem.box_shape)),
            "nonzero_weights": int(np.count_nonzero(weights)),
            "max_energy_mev": float(source.max_energy),
            "seed": seed,
        },
    }


# -- nvidia-smi ------------------------------------------------------------------------

GPU_FIELDS = (
    "index", "uuid", "name", "pci.bus_id", "driver_version", "pstate", "persistence_mode",
    "clocks.sm", "clocks.max.sm", "clocks.mem", "clocks.max.mem", "temperature.gpu",
    "utilization.gpu", "memory.used", "memory.total", "power.draw", "power.limit",
)  # fmt: skip
REASON_FIELDS = ("clocks_event_reasons.active", "clocks_throttle_reasons.active")
REASONS = {
    0x1: "gpu_idle", 0x2: "applications_clocks", 0x4: "sw_power_cap", 0x8: "hw_slowdown",
    0x10: "sync_boost", 0x20: "sw_thermal", 0x40: "hw_thermal", 0x80: "hw_power_brake",
    0x100: "display_clocks",
}  # fmt: skip
SLOWDOWN = {"sw_power_cap", "hw_slowdown", "sw_thermal", "hw_thermal", "hw_power_brake"}
_MISSING = {"", "n/a", "[n/a]", "[not supported]", "not supported", "[unknown error]"}


def _parse(value: str) -> Any:
    value = value.strip()
    if value.lower() in _MISSING:
        return None
    for kind in (int, float):
        with contextlib.suppress(ValueError):
            return kind(value)
    return value


def decode_reasons(mask: Any) -> list[str] | None:
    """Named throttle/event reasons of an nvidia-smi hex mask (``None`` if unknown)."""
    if not isinstance(mask, str) or not mask.startswith("0x"):
        return None
    bits = int(mask, 16)
    names = [name for bit, name in REASONS.items() if bits & bit]
    other = bits & ~sum(REASONS)
    return names + ([f"0x{other:x}"] if other else [])


class NvidiaSmi:
    """Query nvidia-smi, tolerating fields this card or driver does not report."""

    def __init__(self) -> None:
        self.exe = shutil.which("nvidia-smi")
        self.fields: list[str] = []
        if self.exe is None:
            return
        wanted = list(GPU_FIELDS)
        for reason in REASON_FIELDS:  # renamed in newer drivers; take the first valid
            if self._run([f"--query-gpu=index,{reason}"]) is not None:
                wanted.append(reason)
                break
        if self._run(["--query-gpu=" + ",".join(wanted)]) is not None:
            self.fields = wanted
        else:  # some field is unknown to this driver: keep the valid ones
            self.fields = [f for f in wanted if self._run([f"--query-gpu={f}"]) is not None]

    def _run(self, query: list[str]) -> str | None:
        if self.exe is None:
            return None
        try:
            done = subprocess.run(
                [self.exe, *query, "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=30, check=False,
            )  # fmt: skip
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout if done.returncode == 0 else None

    def gpus(self) -> list[dict[str, Any]]:
        """One record per GPU; reason masks decoded under ``reasons``."""
        out = self._run(["--query-gpu=" + ",".join(self.fields)]) if self.fields else None
        if out is None:
            return []
        rows = []
        for line in out.strip().splitlines():
            row = dict(zip(self.fields, (_parse(v) for v in line.split(",")), strict=False))
            mask = next((row[f] for f in REASON_FIELDS if f in row), None)
            row["reasons"] = decode_reasons(mask)
            row["time"] = time.time()
            rows.append(row)
        return rows

    def compute_apps(self) -> list[dict[str, Any]]:
        """Compute processes nvidia-smi lists (pid, name, GPU, MiB or None)."""
        out = self._run(["--query-compute-apps=gpu_uuid,pid,process_name,used_memory"])
        apps = []
        for line in (out or "").strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                apps.append(
                    {"gpu_uuid": parts[0], "pid": _parse(parts[1]),
                     "name": Path(parts[2]).name, "used_mib": _parse(parts[3])}
                )  # fmt: skip
        return apps


class ClockSampler:
    """Sample nvidia-smi from a thread while timed rounds run."""

    def __init__(self, smi: NvidiaSmi, interval: float) -> None:
        self.smi, self.interval = smi, interval
        self.samples: list[dict[str, Any]] = []
        self.label = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.interval <= 0.0 or not self.smi.fields:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            label = self.label
            for row in self.smi.gpus():
                row["label"] = label
                self.samples.append(row)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=60)


def summarize_samples(samples: list[dict[str, Any]], label: str | None = None) -> dict[str, Any]:
    """Per GPU: SM clock range under load (>= 50 % utilization), reasons seen, peak temp."""
    out: dict[str, Any] = {}
    for row in samples:
        if label is not None and row.get("label") != label:
            continue
        key = str(row.get("index"))
        entry = out.setdefault(key, {"n": 0, "loaded": [], "reasons": set(), "temp_max": None})
        entry["n"] += 1
        util, clock, temp = (
            row.get("utilization.gpu"),
            row.get("clocks.sm"),
            row.get("temperature.gpu"),
        )
        if isinstance(util, int | float) and util >= 50 and isinstance(clock, int | float):
            entry["loaded"].append(clock)
        entry["reasons"].update(row.get("reasons") or [])
        if isinstance(temp, int | float):
            entry["temp_max"] = max(temp, entry["temp_max"] or temp)
    for entry in out.values():
        loaded = entry.pop("loaded")
        entry["loaded_samples"] = len(loaded)
        entry["sm_clock_loaded_min"] = min(loaded) if loaded else None
        entry["sm_clock_loaded_max"] = max(loaded) if loaded else None
        entry["reasons"] = sorted(entry["reasons"])
        entry["slowdown_seen"] = sorted(set(entry["reasons"]) & SLOWDOWN)
    return out


# -- environment -----------------------------------------------------------------------


def _git(*args: str) -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        done = subprocess.run(
            [git, "-C", str(REPO), *args], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def environment(wp: Any, smi: NvidiaSmi, gpus: list[str]) -> dict[str, Any]:
    """Interpreter, package, checkout, Warp, driver and GPU record."""
    import pyradmc

    module_file = Path(pyradmc.__file__).resolve()
    status = _git("status", "--porcelain", "--untracked-files=no")
    devices = []
    for name in gpus:
        d = wp.get_device(name)
        devices.append(
            {"device": name, "name": d.name, "arch": getattr(d, "arch", None),
             "sm_count": getattr(d, "sm_count", None), "uuid": str(getattr(d, "uuid", "")),
             "pci_bus_id": getattr(d, "pci_bus_id", None),
             "total_memory_mib": int(d.total_memory) // 2**20,
             "free_memory_mib": int(d.free_memory) // 2**20}
        )  # fmt: skip
    smi_rows = smi.gpus()
    return {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "pyradmc_version": getattr(pyradmc, "__version__", None),
        "pyradmc_file": str(module_file),
        "pyradmc_from_this_checkout": module_file.is_relative_to(REPO),
        "checkout": str(REPO),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty_tracked_files": None if status is None else len(status.splitlines()),
        "warp": wp.__version__,
        "warp_cuda_toolkit": list(wp.get_cuda_toolkit_version() or ()),
        "warp_cuda_driver": list(wp.get_cuda_driver_version() or ()),
        "nvidia_driver": smi_rows[0].get("driver_version") if smi_rows else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": devices,
        "identical_gpus": len({(d["name"], d["arch"]) for d in devices}) <= 1,
    }


def idle_warnings(
    rows: list[dict[str, Any]], apps: list[dict[str, Any]], busy_mib: int
) -> list[str]:
    """Warnings for GPUs already in use (memory held, utilization, compute processes)."""
    warnings = []
    for row in rows:
        used, util = row.get("memory.used"), row.get("utilization.gpu")
        if isinstance(used, int | float) and used > busy_mib:
            warnings.append(f"GPU {row.get('index')} has {used} MiB in use before the benchmark")
        if isinstance(util, int | float) and util > 10:
            warnings.append(f"GPU {row.get('index')} is {util} % utilized before the benchmark")
    for app in apps:
        mib = app["used_mib"]
        if mib is None or mib > busy_mib:
            held = "unknown memory" if mib is None else f"{mib} MiB"
            warnings.append(f"compute process {app['name']} (pid {app['pid']}) holds {held}")
    return warnings


# -- the engine, observed from outside -------------------------------------------------


class ShardSpy:
    """Record passes and which device pulled which beamlet group, per call.

    Wraps two private methods on the engine *instance* (the class is untouched):
    ``_compute_dij`` (one call per transport pass) and ``_run_dij_shard`` (one per device
    per pass; its ``next_group`` pulls from the shared group queue). Benchmark-only.
    """

    def __init__(self, engine: Any) -> None:
        self.ok = all(hasattr(engine, n) for n in ("_compute_dij", "_run_dij_shard"))
        self._lock = threading.Lock()
        self.passes: list[list[dict[str, Any]]] = []
        self.t0 = 0.0
        if not self.ok:
            return
        compute, run_shard = engine._compute_dij, engine._run_dij_shard

        def compute_spy(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                self.passes.append([])
            return compute(*args, **kwargs)

        def shard_spy(device: str, next_group: Callable[[], int | None], **kwargs: Any) -> Any:
            record: dict[str, Any] = {
                "device": device,
                "group_size": kwargs.get("beamlet_group_size"),
                "start": time.perf_counter() - self.t0,
                "pulls": [],
            }

            def pull() -> int | None:
                group = next_group()
                record["pulls"].append((time.perf_counter() - self.t0, group))
                return group

            try:
                return run_shard(device, pull, **kwargs)
            finally:
                record["end"] = time.perf_counter() - self.t0
                with self._lock:
                    self.passes[-1].append(record)

        object.__setattr__(engine, "_compute_dij", compute_spy)
        object.__setattr__(engine, "_run_dij_shard", shard_spy)

    def begin(self, t0: float) -> None:
        self.passes, self.t0 = [], t0

    def summary(self, devices: Sequence[str], n_beamlets: int, n_hist: int) -> dict[str, Any]:
        """Passes, group size/count and per-device work of the pass that was returned."""
        if not self.ok or not self.passes:
            return {"passes": len(self.passes) if self.ok else None}
        last = self.passes[-1]
        size = next((r["group_size"] for r in last if r["group_size"]), None)
        shards = {}
        for r in sorted(last, key=lambda r: list(devices).index(r["device"])):
            starts = [g for _, g in r["pulls"] if g is not None]
            beamlets = sum(min(size, n_beamlets - g) for g in starts) if size else 0
            first = r["pulls"][0][0] if r["pulls"] else r["end"]
            shards[r["device"]] = {
                "groups": len(starts),
                "beamlets": beamlets,
                "histories": beamlets * n_hist,
                "share": beamlets / n_beamlets,
                "setup_s": first - r["start"],  # shard start to its first pull
                "done_s": r["end"],  # from the call's start
            }
        return {
            "passes": len(self.passes),
            "group_size": size,
            "n_groups": -(-n_beamlets // size) if size else None,
            "shards": shards,
        }


@dataclass
class Arm:
    """One configuration: devices, groups per device (None: the engine's rule), fold."""

    label: str
    devices: list[str]
    groups: int | None
    fold: str = "deposit"
    n_gpus: int = 1
    has_cpu: bool = False
    note: str | None = None
    times: list[float] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    active: bool = True


def build_arms(n_gpus: int, groups: Sequence[int], include_cpu: bool, dense: bool) -> list[Arm]:
    """1 GPU; 2..N GPUs, auto and at each G; optionally with cpu added; dense on 1 GPU."""
    arms = [Arm("1 GPU", ["cuda:0"], None)]
    for k in range(2, n_gpus + 1):
        gpus = [f"cuda:{i}" for i in range(k)]
        arms.append(Arm(f"{k} GPU auto", gpus, None, n_gpus=k))
        arms += [Arm(f"{k} GPU G={g}", gpus, g, n_gpus=k) for g in groups]
    if include_cpu:
        for k in range(1, n_gpus + 1):
            devs = [*(f"cuda:{i}" for i in range(k)), "cpu"]
            arms += [Arm(f"{k} GPU+cpu G={g}", devs, g, n_gpus=k, has_cpu=True) for g in groups]
    if dense:
        arms.append(Arm("dense 1 GPU", ["cuda:0"], None, fold="dense"))
    return arms


def digest(result: Any) -> dict[str, Any]:
    """Hashes and books of a contraction result (for the JSON and the bit check)."""
    return {
        "values_sha256": hashlib.sha256(np.ascontiguousarray(result.values).tobytes()).hexdigest(),
        "sigma_sha256": hashlib.sha256(np.ascontiguousarray(result.sigma).tobytes()).hexdigest(),
        "books": [
            result.energy_emitted, result.energy_deposited,
            result.energy_escaped, result.energy_unscored,
        ],
    }  # fmt: skip


def compare(result: Any, reference: Any) -> dict[str, Any]:
    """Bitwise equality of values, sigma and books, and the largest relative difference."""
    scale = float(np.abs(reference.values).max()) or 1.0
    return {
        "values_equal": bool(np.array_equal(result.values, reference.values)),
        "sigma_equal": bool(np.array_equal(result.sigma, reference.sigma)),
        "books_equal": digest(result)["books"] == digest(reference)["books"],
        "max_rel_diff": float(np.abs(result.values - reference.values).max()) / scale,
    }


class Runner:
    """Calls one problem's engine for one arm and records what happened."""

    def __init__(self, wp: Any, engine: Any, problem: dict[str, Any]) -> None:
        self.wp, self.engine = wp, engine
        self.problem = problem
        self.spy = ShardSpy(engine)

    def call(self, arm: Arm, n_hist: int, n_batches: int, seed: int) -> tuple[float, Any, dict]:
        wp, source = self.wp, self.problem["source"]
        gc.collect()
        wp.synchronize()
        t0 = time.perf_counter()
        self.spy.begin(t0)
        result = self.engine.run_beamlet_contraction(
            source, self.problem["weights"], n_histories_per_beamlet=n_hist,
            n_batches=n_batches, seed=seed, scoring_grid=self.problem["scoring"],
            devices=arm.devices, fold=arm.fold,
            beamlet_group_size=expected_group(arm, source.n_beamlets),
        )  # fmt: skip
        wp.synchronize()
        seconds = time.perf_counter() - t0
        info = self.spy.summary(arm.devices, source.n_beamlets, n_hist)
        p = result.provenance
        info["provenance"] = {
            k: getattr(p, k, None)
            for k in ("device", "fold", "fold_requested", "fold_scale_exponent", "fold_bands",
                      "fold_band_width")
        }  # fmt: skip
        return seconds, result, info


# -- one block: a problem at one history count -----------------------------------------


def expected_group(arm: Arm, n_beamlets: int) -> int | None:
    """The explicit group of a G arm, ``ceil(edges / (devices * G))``; None: engine's rule."""
    if arm.groups is None:
        return None
    return max(1, -(-n_beamlets // (len(arm.devices) * arm.groups)))


def run_block(
    runner: Runner, arms: list[Arm], n_hist: int, args: argparse.Namespace, log: Callable
) -> dict[str, Any]:
    """Warm up every arm, then the interleaved rounds; returns the block's record."""
    n_beamlets = runner.problem["source"].n_beamlets
    for arm in arms:
        if arm.has_cpu and n_hist > args.cpu_max_histories:
            arm.active, arm.note = False, f"skipped: cpu arms run up to {args.cpu_max_histories}"
    log(f"  warm-up ({args.warmup} per arm) ...")
    for arm in (a for a in arms if a.active):
        seconds = 0.0
        for w in range(args.warmup):
            try:
                seconds, _, info = runner.call(arm, n_hist, args.batches, args.seed + 20_000 + w)
            except Exception as error:  # an arm that cannot run is a note, not a crash
                arm.active, arm.note = False, f"failed in warm-up: {type(error).__name__}: {error}"
                log(f"    {arm.label}: {arm.note}")
                break
            log(f"    {arm.label:<18} warm-up {w + 1}: {seconds:8.4f} s  {_share_text(info)}")
        if arm.active and arm.fold == "dense" and seconds > args.dense_max_seconds:
            arm.active = False
            arm.note = (
                f"dropped: warm-up {seconds:.1f} s > --dense-max-seconds {args.dense_max_seconds}"
            )
            log(f"    {arm.label}: {arm.note}")
    live = [a for a in arms if a.active]
    reference = arms[0]
    if not reference.active:
        raise RuntimeError(f"the 1-GPU reference arm cannot run: {reference.note}")
    log(f"  {args.repeats} rounds x {len(live)} arms (settle {args.settle}) ...")
    for r in range(args.repeats):
        seed = args.seed + r
        ref_result = None
        line = []
        for arm in live:
            if not arm.active:
                continue
            try:
                for s in range(args.settle):
                    runner.call(arm, n_hist, args.batches, args.seed + 10_000 + 100 * r + s)
                seconds, result, info = runner.call(arm, n_hist, args.batches, seed)
            except Exception as error:
                arm.active, arm.note = (
                    False,
                    f"failed in round {r}: {type(error).__name__}: {error}",
                )
                log(f"    {arm.label}: {arm.note}")
                continue
            if arm is reference:
                ref_result = result
            info.update(round=r, seed=seed, seconds=seconds, **digest(result))
            if ref_result is not None and arm is not reference:
                info["vs_1gpu"] = compare(result, ref_result)
            arm.times.append(seconds)
            arm.calls.append(info)
            line.append(f"{arm.label} {seconds:.4f}")
        log(f"    round {r + 1}/{args.repeats}: " + ", ".join(line))
    best_ref = min(reference.times) if reference.times else None
    records = []
    for arm in arms:
        records.append(_arm_record(arm, n_beamlets, best_ref))
    return {"histories_per_edge": n_hist, "batches": args.batches, "arms": records}


def _share_text(info: dict[str, Any]) -> str:
    shards = info.get("shards") or {}
    parts = [f"{d} {s['groups']}g" for d, s in shards.items()]
    size = info.get("group_size")
    head = f"group {size} x {info.get('n_groups')}" if size else ""
    return f"{head}  {' '.join(parts)}  passes {info.get('passes')}"


def _arm_record(arm: Arm, n_beamlets: int, best_ref: float | None) -> dict:
    times = arm.times
    record: dict[str, Any] = {
        "label": arm.label,
        "devices": arm.devices,
        "groups_per_device": arm.groups,
        "fold": arm.fold,
        "n_gpus": arm.n_gpus,
        "has_cpu": arm.has_cpu,
        "note": arm.note,
        "times_s": times,
        "calls": arm.calls,
    }
    if not times:
        return record
    best = min(times)
    record.update(best_s=best, median_s=statistics.median(times), max_s=max(times))
    if best_ref is not None:
        record["speedup_best"] = best_ref / best
        record["efficiency"] = best_ref / best / arm.n_gpus
    sizes = {c.get("group_size") for c in arm.calls}
    record["group_sizes"] = sorted(s for s in sizes if s is not None)
    record["n_groups"] = sorted({c.get("n_groups") for c in arm.calls if c.get("n_groups")})
    want = expected_group(arm, n_beamlets) if arm.fold == "deposit" else None
    record["expected_group_size"] = want
    record["override_effective"] = None if want is None else record["group_sizes"] == [want]
    record["passes"] = sorted({c.get("passes") for c in arm.calls if c.get("passes") is not None})
    devices: dict[str, dict[str, list[float]]] = {}
    for c in arm.calls:
        for d, s in (c.get("shards") or {}).items():
            e = devices.setdefault(d, {"groups": [], "share": [], "setup_s": [], "done_s": []})
            for k in e:
                e[k].append(s[k])
    record["work"] = {
        d: {
            "groups": sorted(set(e["groups"])),
            "share_min": min(e["share"]),
            "share_max": max(e["share"]),
            "setup_s_median": statistics.median(e["setup_s"]),
            "done_s_median": statistics.median(e["done_s"]),
        }
        for d, e in devices.items()
    }
    gpu_shares = [record["work"][d]["share_max"] for d in arm.devices if d.startswith("cuda")]
    record["one_gpu_did_everything"] = arm.n_gpus > 1 and any(s >= 1.0 for s in gpu_shares)
    cpu = record["work"].get("cpu")
    record["cpu_transported"] = None if cpu is None else cpu["share_max"] > 0.0
    vs = [c["vs_1gpu"] for c in arm.calls if "vs_1gpu" in c]
    if vs:
        record["bitwise_equal_rounds"] = sum(
            v["values_equal"] and v["sigma_equal"] and v["books_equal"] for v in vs
        )
        record["compared_rounds"] = len(vs)
        record["max_rel_diff"] = max(v["max_rel_diff"] for v in vs)
        record["books_equal_rounds"] = sum(v["books_equal"] for v in vs)
    record["provenance"] = arm.calls[-1].get("provenance")
    return record


# -- reporting -------------------------------------------------------------------------


def _bits(rec: dict[str, Any]) -> str:
    if "compared_rounds" not in rec:
        return "ref" if rec["label"] == "1 GPU" else "-"
    n, k, diff = rec["compared_rounds"], rec["bitwise_equal_rounds"], rec["max_rel_diff"]
    books = "books equal" if rec["books_equal_rounds"] == n else "BOOKS DIFFER"
    if rec["fold"] == "dense":  # agrees to the fold's rounding only, by design
        return f"rounding {diff:.1e}, {books}"
    if k == n:
        return f"equal {k}/{n}"
    if rec.get("cpu_transported"):  # cpu and GPU never agree bitwise (AGENTS.md 2.3)
        return f"cpu ran groups: {diff:.1e}"
    return f"DIFF {n - k}/{n} ({diff:.1e})"


def print_block(problem: dict[str, Any], block: dict[str, Any]) -> None:
    """The block's timing table and its work-share table."""
    info = problem["record"]
    print(
        f"\n== {info['name']}: {info['edges']} edges, {info['scoring_voxels']:,} scoring voxels, "
        f"{block['histories_per_edge']:,} histories/edge, {block['batches']} batches =="
    )
    head = (
        f"{'arm':<20}{'group x n':>12}{'best s':>10}{'median s':>10}{'max s':>10}"
        f"{'speed-up':>10}{'eff.':>7}{'passes':>8}  {'bits vs 1 GPU':<18}"
    )
    print(head)
    print("-" * len(head))
    for rec in block["arms"]:
        if "best_s" not in rec:
            print(f"{rec['label']:<20}  {rec['note']}")
            continue
        gx = (
            "/".join(str(s) for s in rec["group_sizes"])
            + " x "
            + "/".join(str(n) for n in rec["n_groups"])
        )
        flag = "" if rec.get("override_effective") in (None, True) else "  GROUP BOUNDED"
        flag += "  ONE GPU DID ALL" if rec.get("one_gpu_did_everything") else ""
        flag += "  cpu took no group" if rec.get("cpu_transported") is False else ""
        passes = "/".join(str(p) for p in rec["passes"])
        print(
            f"{rec['label']:<20}{gx:>12}{rec['best_s']:>10.4f}{rec['median_s']:>10.4f}"
            f"{rec['max_s']:>10.4f}{rec.get('speedup_best', float('nan')):>10.2f}"
            f"{rec.get('efficiency', float('nan')):>7.2f}{passes:>8}  {_bits(rec):<18}{flag}"
        )
    multi = [r for r in block["arms"] if len(r["devices"]) > 1 and "work" in r]
    if multi:
        print("work per device (share of histories min-max, groups; set-up = shard start to")
        print("first group pulled; done = shard end from call start; medians, timed calls):")
    for rec in multi:
        parts = [
            f"{d} {w['share_min']:.0%}-{w['share_max']:.0%} {'/'.join(map(str, w['groups']))}g "
            f"set-up {w['setup_s_median']:.3f} done {w['done_s_median']:.3f}"
            for d, w in rec["work"].items()
        ]
        print(f"  {rec['label']:<18} " + " | ".join(parts))


def print_crossover(results: list[dict[str, Any]], n_gpus: int) -> None:
    """Best speed-up over the arms of a GPU count, per problem and history count."""
    if n_gpus < 2:
        return
    print("\n== speed-up vs 1 GPU, best arm per GPU count [G, or auto] ==")
    print(
        f"{'problem':<8}{'hist/edge':>11}"
        + "".join(f"{f'{k} GPU':>14}" for k in range(2, n_gpus + 1))
    )
    for res in results:
        for block in res["blocks"]:
            cells = []
            for k in range(2, n_gpus + 1):
                cand = [a for a in block["arms"] if a["n_gpus"] == k and not a["has_cpu"]
                        and "speedup_best" in a]  # fmt: skip
                if cand:
                    top = max(cand, key=lambda a: a["speedup_best"])
                    g = top["groups_per_device"] or "auto"
                    cells.append(f"{top['speedup_best']:.2f} [{g}]")
                else:
                    cells.append("-")
            print(f"{res['problem']['name']:<8}{block['histories_per_edge']:>11,}"
                  + "".join(f"{c:>14}" for c in cells))  # fmt: skip


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer | np.floating | np.bool_):
        return value.item()
    if isinstance(value, set | tuple):
        return list(value)
    return str(value)


# -- main ------------------------------------------------------------------------------


def _ints(text: str) -> list[int]:
    return [int(t) for t in text.replace("_", "").split(",") if t.strip()]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                        help="smoke test (~1 min): small problem, 1000 and 10000 histories/edge, "
                             "2 repeats, 1 warm-up, G 2,4")  # fmt: skip
    parser.add_argument("--problems", default="large,small", help="comma list of large, small")
    parser.add_argument("--histories", default="1000,10000,100000",
                        help="histories per edge, comma list")  # fmt: skip
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2, help="untimed calls per arm and block")
    parser.add_argument("--settle", type=int, default=1,
                        help="untimed calls of the same arm before each timed call")  # fmt: skip
    parser.add_argument("--groups-per-device", default="2,4,8",
                        help="fixed groups per device for multi-device arms")  # fmt: skip
    parser.add_argument("--max-gpus", type=int, default=None, help="use cuda:0 .. cuda:N-1")
    parser.add_argument("--include-cpu", action="store_true", help="add cpu to every device set")
    parser.add_argument("--cpu-max-histories", type=int, default=1000,
                        help="cpu arms only up to this many histories per edge")  # fmt: skip
    parser.add_argument("--no-dense", action="store_true", help="skip the dense-fold baseline")
    parser.add_argument("--dense-max-seconds", type=float, default=60.0,
                        help="drop the dense arm if its warm-up call takes longer")  # fmt: skip
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--sample-interval", type=float, default=1.0,
                        help="nvidia-smi sampling during the rounds (s); 0 disables")  # fmt: skip
    parser.add_argument("--busy-mib", type=int, default=256,
                        help="warn if a GPU holds more than this before the start")  # fmt: skip
    parser.add_argument("--out", default=None, help="JSON path (default: timestamped, in cwd)")
    parser.add_argument("--installed", action="store_true",
                        help="import the installed pyradmc, not this checkout's")  # fmt: skip
    args = parser.parse_args(argv)
    if args.quick:
        args.problems, args.histories = "small", "1000,10000"
        args.repeats, args.warmup, args.groups_per_device = 2, 1, "2,4"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.installed and (REPO / "pyradmc" / "__init__.py").is_file():
        sys.path.insert(0, str(REPO))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out or f"bench_multi_gpu_contraction_{stamp}.json").resolve()

    def log(message: str) -> None:
        print(message, flush=True)

    smi = NvidiaSmi()
    before = smi.gpus()  # before this process touches CUDA
    apps = smi.compute_apps()
    warnings = idle_warnings(before, apps, args.busy_mib)
    if not before:
        warnings.append("nvidia-smi not found or not answering: no idle check, clocks or "
                        "throttle reasons are recorded")  # fmt: skip

    import warp as wp

    wp.init()  # its banner names the toolkit, driver and devices: kept in the log
    from pyradmc.backends.warp import engine as module

    n_cuda = wp.get_cuda_device_count()
    if n_cuda < 1:
        log("no CUDA device visible to Warp; this benchmark needs at least one GPU")
        return 2
    n_gpus = min(n_cuda, args.max_gpus or n_cuda)
    gpus = [f"cuda:{i}" for i in range(n_gpus)]
    env = environment(wp, smi, gpus)
    env["fold_group_histories_default"] = module._FOLD_GROUP_HISTORIES
    for d in env["gpus"]:
        if d["free_memory_mib"] < 0.85 * d["total_memory_mib"]:
            warnings.append(f"{d['device']} reports {d['free_memory_mib']} of "
                            f"{d['total_memory_mib']} MiB free at start")  # fmt: skip
    if not env["pyradmc_from_this_checkout"]:
        warnings.append(f"pyradmc is imported from {env['pyradmc_file']}, not from {REPO}")

    log(f"pyradmc {env['pyradmc_version']} from {env['pyradmc_file']}")
    log(f"checkout {REPO}  commit {env['git_commit']}  branch {env['git_branch']}  "
        f"dirty tracked files {env['git_dirty_tracked_files']}")  # fmt: skip
    log(f"warp {env['warp']} (toolkit {env['warp_cuda_toolkit']}, driver CUDA "
        f"{env['warp_cuda_driver']}), NVIDIA driver {env['nvidia_driver']}, "
        f"python {platform.python_version()}, {env['platform']}")  # fmt: skip
    for d in env["gpus"]:
        log(f"  {d['device']}: {d['name']} sm_{d['arch']} {d['sm_count']} SMs, "
            f"{d['total_memory_mib']} MiB ({d['free_memory_mib']} free)")  # fmt: skip
    log(f"identical GPUs: {env['identical_gpus']}; fold_group_histories default "
        f"{module._FOLD_GROUP_HISTORIES:,}")  # fmt: skip
    for row in before:
        log(f"  nvidia-smi GPU {row.get('index')}: {row.get('clocks.sm')}/"
            f"{row.get('clocks.max.sm')} MHz, {row.get('temperature.gpu')} C, "
            f"{row.get('utilization.gpu')} %, {row.get('memory.used')} MiB, reasons "
            f"{row.get('reasons')}")  # fmt: skip
    for w in warnings:
        log(f"WARNING: {w}")

    groups = _ints(args.groups_per_device)
    record: dict[str, Any] = {
        "benchmark": "bench_multi_gpu_contraction",
        "settings": vars(args),
        "environment": env,
        "warnings": warnings,
        "nvidia_smi_before": before,
        "compute_apps_before": apps,
        "results": [],
    }
    sampler = ClockSampler(smi, args.sample_interval)

    def save() -> None:
        record["clock_samples"] = sampler.samples
        out.write_text(json.dumps(record, indent=1, default=_json_default), encoding="utf-8")

    from pyradmc.backends.warp.engine import WarpEngine

    sampler.start()
    t_start = time.perf_counter()
    try:
        for name in [p.strip() for p in args.problems.split(",") if p.strip()]:
            problem = build_problem(PROBLEMS[name], args.seed)
            info = problem["record"]
            log(f"\n{name}: {info['edges']} edges ({info['angles']} angles x "
                f"{2 * info['leaf_pairs']}), scoring box {info['scoring_box']} = "
                f"{info['scoring_voxels']:,} voxels ({info['nonzero_weights']:,} weighted), "
                f"transport {info['transport_voxels']:,} voxels")  # fmt: skip
            log(f"  stands in for {info['stands_in_for']}")
            engine = WarpEngine(grid=problem["grid"], cross_sections=problem["xs"], device="cuda:0")
            runner = Runner(wp, engine, problem)
            if not runner.spy.ok:
                warnings.append("engine internals changed: per-device work cannot be recorded")
            result = {"problem": info, "blocks": []}
            record["results"].append(result)
            for n_hist in _ints(args.histories):
                if n_hist % args.batches:
                    log(f"  skipping {n_hist}: not divisible by {args.batches} batches")
                    continue
                arms = build_arms(n_gpus, groups, args.include_cpu, not args.no_dense)
                sampler.label = f"{name}/{n_hist}"
                log(f"\n[{name}, {n_hist:,} histories/edge]")
                block = run_block(runner, arms, n_hist, args, log)
                block["clocks"] = summarize_samples(sampler.samples, sampler.label)
                block["chunk_sizes"] = dict(getattr(engine, "_auto_chunk", {}))
                result["blocks"].append(block)
                save()
                print_block(problem, block)
                for gpu, c in block["clocks"].items():
                    slow = f"  SLOWDOWN {c['slowdown_seen']}" if c["slowdown_seen"] else ""
                    log(f"  clocks GPU {gpu}: {c['loaded_samples']} loaded samples, SM "
                        f"{c['sm_clock_loaded_min']}-{c['sm_clock_loaded_max']} MHz, peak "
                        f"{c['temp_max']} C, reasons {c['reasons']}{slow}")  # fmt: skip
            engine.release_device_memory()
            del engine, runner, problem
            gc.collect()
    except KeyboardInterrupt:
        log("\ninterrupted; writing what was measured")
        record["interrupted"] = True
    finally:
        sampler.stop()
        record["wall_s"] = time.perf_counter() - t_start
        record["nvidia_smi_after"] = smi.gpus()
        save()
    print_crossover(record["results"], n_gpus)
    for row in record["nvidia_smi_after"]:
        log(f"  after: GPU {row.get('index')}: {row.get('clocks.sm')} MHz, "
            f"{row.get('temperature.gpu')} C, {row.get('memory.used')} MiB, "
            f"reasons {row.get('reasons')}")  # fmt: skip
    for w in warnings:
        log(f"WARNING: {w}")
    log(f"\nwall {record['wall_s']:.0f} s; wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
