"""
ROOTFileLoader
==============
Backends: PyROOT (preferred) or uproot (fallback).

Draw modes for TTree loading:
  filter   — Draw("energy", "channelID==N")          separate branches, cut per channel
  array    — Draw("energy[channelID]")                array branch indexed by channel
  custom   — user supplies draw_expr (%d = channel)  full control
              e.g. "sqrt(energy[%d])" or "adc>>h(%d,cut)"

User controls:
  channel_ids    — explicit list, or auto-discovered from data
  ch_first/last/step — range specification
  max_entries    — limit entries read per channel (0 = all)
  n_bins         — histogram bins
"""

from __future__ import annotations
import sys, os, subprocess
import numpy as np
from dataclasses import dataclass
from typing import Optional

# ── Backend detection ──────────────────────────────────────────────────── #

_BACKEND = None

def _find_root_python_path() -> Optional[str]:
    candidates = []
    rootsys = os.environ.get("ROOTSYS", "")
    if rootsys:
        candidates.append(rootsys)
    try:
        prefix = subprocess.check_output(
            ["root-config", "--prefix"],
            stderr=subprocess.DEVNULL, timeout=5
        ).decode().strip()
        if prefix:
            candidates.append(prefix)
    except Exception:
        pass
    candidates += [
        "/usr/local", "/opt/root", "/opt/homebrew",
        "/usr/local/opt/root", os.path.expanduser("~/root"),
        os.path.expanduser("~/ROOT"), "/opt/local",
    ]
    python_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    suffixes = [
        f"lib/{python_ver}/site-packages",
        f"lib/{python_ver}/dist-packages",
        "lib/python3/site-packages", "lib",
    ]
    for prefix in candidates:
        for suffix in suffixes:
            path = os.path.join(prefix, suffix)
            if os.path.isdir(path) and any(
                    f.startswith("ROOT") for f in os.listdir(path)):
                return path
    return None

def _detect_backend() -> str:
    global _BACKEND
    if _BACKEND:
        return _BACKEND
    try:
        import ROOT
        ROOT.gROOT.SetBatch(True)
        ROOT.gErrorIgnoreLevel = ROOT.kError
        _BACKEND = "pyroot"
        return _BACKEND
    except ImportError:
        pass
    root_py_path = _find_root_python_path()
    if root_py_path and root_py_path not in sys.path:
        sys.path.insert(0, root_py_path)
        lib_dir = os.path.normpath(
            os.path.join(root_py_path, "..", "lib"))
        env_key = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" \
                  else "LD_LIBRARY_PATH"
        old = os.environ.get(env_key, "")
        os.environ[env_key] = f"{lib_dir}:{old}" if old else lib_dir
        try:
            import ROOT
            ROOT.gROOT.SetBatch(True)
            ROOT.gErrorIgnoreLevel = ROOT.kError
            _BACKEND = "pyroot"
            return _BACKEND
        except Exception:
            pass
    try:
        import uproot
        _BACKEND = "uproot"
        return _BACKEND
    except ImportError:
        pass
    raise RuntimeError(
        "No ROOT backend found.\n\n"
        "Option A — PyROOT:\n"
        "  source /path/to/root/bin/thisroot.sh\n"
        "  python3 main.py\n\n"
        "Option B — uproot (no ROOT needed):\n"
        "  pip install uproot awkward\n"
        "  python3 main.py\n\n"
        f"Tried ROOT path: {root_py_path or 'not found'}"
    )

def get_backend() -> str:
    return _detect_backend()


# ── Draw mode constants ────────────────────────────────────────────────── #

DRAW_MODES = {
    "filter": 'Draw("energy", "channelID==N")  — separate branches, filter per channel',
    "array":  'Draw("energy[channelID]")        — array branch indexed by channel',
    "custom": 'Custom expression  (%d replaced by channel number)',
}


# ── Data class ─────────────────────────────────────────────────────────── #

@dataclass
class ChannelSpectrum:
    channel_id:  int
    name:        str
    bin_centers: np.ndarray
    counts:      np.ndarray
    n_entries:   int
    source:      str = "unknown"
    draw_expr:   str = ""        # the actual Draw expression used


# ── Loader ─────────────────────────────────────────────────────────────── #

class ROOTFileLoader:

    def __init__(self):
        self.filename: str = ""
        self.mode:     str = ""
        self.backend:  str = ""
        self.draw_mode: str = "filter"
        self.spectra: dict[int, ChannelSpectrum] = {}
        self._file = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def open(self, filename: str) -> dict:
        self.filename = filename
        self.backend  = _detect_backend()
        if self.backend == "pyroot":
            return self._open_pyroot(filename)
        else:
            return self._open_uproot(filename)

    def load_from_th1(self, hist_names: Optional[list] = None):
        self.mode    = "th1"
        self.spectra = {}
        if self.backend == "pyroot":
            self._load_th1_pyroot(hist_names)
        else:
            self._load_th1_uproot(hist_names)

    def load_from_ttree(self,
                         tree_name:      str,
                         channel_branch: str,
                         adc_branch:     str,
                         n_bins:         int  = 1024,
                         draw_mode:      str  = "filter",
                         custom_expr:    str  = "",
                         channel_ids:    Optional[list] = None,
                         ch_first:       int  = 0,
                         ch_last:        int  = -1,
                         ch_step:        int  = 1,
                         max_entries:    int  = 0):
        """
        Load per-channel spectra from a TTree.

        draw_mode:
          "filter"  — Draw("adc_branch", "channel_branch==N")
          "array"   — Draw("adc_branch[N]")   for array branches
          "custom"  — custom_expr with %d replaced by channel number
                      e.g.  "sqrt(energy[%d])"

        channel_ids: explicit list of channel IDs to load.
                     If None, use ch_first/ch_last/ch_step range,
                     or auto-discover from data (filter mode only).

        max_entries: max events to read per channel (0 = all).
        """
        self.mode      = "ttree"
        self.draw_mode = draw_mode
        self.spectra   = {}

        if self.backend == "pyroot":
            self._load_ttree_pyroot(
                tree_name, channel_branch, adc_branch,
                n_bins, draw_mode, custom_expr,
                channel_ids, ch_first, ch_last, ch_step, max_entries)
        else:
            # uproot only supports filter mode natively
            self._load_ttree_uproot(
                tree_name, channel_branch, adc_branch,
                n_bins, draw_mode, custom_expr,
                channel_ids, ch_first, ch_last, ch_step, max_entries)

    def get_channel_ids(self) -> list[int]:
        return sorted(self.spectra.keys())

    def get_spectrum(self, channel_id: int) -> Optional[ChannelSpectrum]:
        return self.spectra.get(channel_id)

    def close(self):
        if self._file is not None:
            try:
                if self.backend == "pyroot":
                    self._file.Close()
                else:
                    self._file.close()
            except Exception:
                pass
            self._file = None

    # ------------------------------------------------------------------ #
    # Channel ID resolution helper
    # ------------------------------------------------------------------ #

    def _resolve_channel_ids(self, channel_ids, ch_first, ch_last,
                               ch_step, discovered: Optional[list]) -> list:
        """
        Priority:
          1. Explicit channel_ids list
          2. ch_first / ch_last / ch_step range  (if ch_last >= 0)
          3. Auto-discovered from data
        """
        if channel_ids:
            return sorted(channel_ids)
        if ch_last >= 0:
            return list(range(ch_first, ch_last + 1, max(1, ch_step)))
        if discovered:
            return sorted(discovered)
        return []

    # ================================================================== #
    # PyROOT backend
    # ================================================================== #

    def _open_pyroot(self, filename: str) -> dict:
        import ROOT
        self._file = ROOT.TFile.Open(filename, "READ")
        if not self._file or self._file.IsZombie():
            raise IOError(f"Cannot open ROOT file: {filename}")

        info = {"trees": [], "histograms": [],
                "filename": filename, "backend": "pyroot"}

        for key in self._file.GetListOfKeys():
            cls  = key.GetClassName()
            name = key.GetName()
            if cls in ("TTree", "TNtuple"):
                tree     = self._file.Get(name)
                branches = [b.GetName() for b in tree.GetListOfBranches()]
                info["trees"].append({
                    "name":     name,
                    "branches": branches,
                    "entries":  int(tree.GetEntries())
                })
            elif cls in ("TH1F", "TH1D", "TH1I", "TH1"):
                h = self._file.Get(name)
                info["histograms"].append({
                    "name":    name,
                    "nbins":   h.GetNbinsX(),
                    "xmin":    h.GetXaxis().GetXmin(),
                    "xmax":    h.GetXaxis().GetXmax(),
                    "entries": int(h.GetEntries())
                })
        return info

    def _load_th1_pyroot(self, hist_names: Optional[list]):
        import re
        all_keys = [k.GetName() for k in self._file.GetListOfKeys()
                    if k.GetClassName() in ("TH1F", "TH1D", "TH1I", "TH1")]
        targets = hist_names if hist_names else all_keys
        for idx, name in enumerate(targets):
            h = self._file.Get(name)
            if not h:
                continue
            n       = h.GetNbinsX()
            centers = np.array([h.GetXaxis().GetBinCenter(i + 1)
                                 for i in range(n)])
            counts  = np.array([h.GetBinContent(i + 1) for i in range(n)])
            m       = re.search(r"(\d+)$", name)
            ch_id   = int(m.group(1)) if m else idx
            self.spectra[ch_id] = ChannelSpectrum(
                channel_id=ch_id, name=name,
                bin_centers=centers, counts=counts,
                n_entries=int(h.GetEntries()), source="th1")

    def _load_ttree_pyroot(self, tree_name, channel_branch, adc_branch,
                            n_bins, draw_mode, custom_expr,
                            channel_ids, ch_first, ch_last, ch_step,
                            max_entries):
        import ROOT
        tree = self._file.Get(tree_name)
        if not tree:
            raise ValueError(f"TTree '{tree_name}' not found.")

        ROOT.gROOT.cd()
        entry_limit = int(max_entries) if max_entries > 0 else -1

        # ── filter mode: Draw("adc", "channelID==N") ────────────────── #
        if draw_mode == "filter":
            # Auto-discover channel IDs if not specified
            discovered = None
            if not channel_ids and ch_last < 0:
                n_total = int(tree.GetEntries())
                tree.SetEstimate(n_total + 1)
                n_drawn = tree.Draw(channel_branch, "", "goff",
                                     entry_limit if entry_limit > 0
                                     else n_total)
                if n_drawn > 0:
                    discovered = sorted(set(
                        int(tree.GetV1()[i]) for i in range(n_drawn)))

            targets = self._resolve_channel_ids(
                channel_ids, ch_first, ch_last, ch_step, discovered)
            if not targets:
                raise ValueError(
                    "No channels found. Set channel range manually.")

            for ch_id in targets:
                cut   = f"{channel_branch}=={ch_id}"
                limit = entry_limit if entry_limit > 0 else \
                        int(tree.GetEntries())
                tree.SetEstimate(limit + 1)
                n = tree.Draw(adc_branch, cut, "goff", limit)
                if n <= 0:
                    continue
                vals = np.array([tree.GetV1()[i] for i in range(n)])
                self._fill_spectrum(ch_id, vals, n_bins,
                                     draw_expr=f'Draw("{adc_branch}","{cut}")')

        # ── array mode: Draw("energy[N]") ───────────────────────────── #
        elif draw_mode == "array":
            if ch_last < 0 and not channel_ids:
                raise ValueError(
                    "Array mode requires a channel range. "
                    "Set First/Last channel.")
            targets = self._resolve_channel_ids(
                channel_ids, ch_first, ch_last, ch_step, None)

            for ch_id in targets:
                expr  = adc_branch.replace("%d", str(ch_id)) \
                        if "%d" in adc_branch \
                        else f"{adc_branch}[{ch_id}]"
                limit = entry_limit if entry_limit > 0 else \
                        int(tree.GetEntries())
                tree.SetEstimate(limit + 1)
                n = tree.Draw(expr, "", "goff", limit)
                if n <= 0:
                    continue
                vals = np.array([tree.GetV1()[i] for i in range(n)])
                self._fill_spectrum(ch_id, vals, n_bins,
                                     draw_expr=f'Draw("{expr}")')

        # ── custom mode: user expression with %d ────────────────────── #
        elif draw_mode == "custom":
            if not custom_expr.strip():
                raise ValueError(
                    "Custom draw mode requires an expression. "
                    'Example: "energy[%d]" or "sqrt(adc),channelID==%d"')
            if ch_last < 0 and not channel_ids:
                raise ValueError(
                    "Custom mode requires a channel range. "
                    "Set First/Last channel.")
            targets = self._resolve_channel_ids(
                channel_ids, ch_first, ch_last, ch_step, None)

            # Parse: expr may be "draw_var,selection" or just "draw_var"
            # %d in either part is replaced by ch_id
            for ch_id in targets:
                filled = custom_expr.replace("%d", str(ch_id))
                parts  = [p.strip() for p in filled.split(",", 1)]
                var    = parts[0]
                cut    = parts[1] if len(parts) > 1 else ""
                limit  = entry_limit if entry_limit > 0 else \
                         int(tree.GetEntries())
                tree.SetEstimate(limit + 1)
                n = tree.Draw(var, cut, "goff", limit)
                if n <= 0:
                    continue
                vals = np.array([tree.GetV1()[i] for i in range(n)])
                self._fill_spectrum(ch_id, vals, n_bins,
                                     draw_expr=f'Draw("{var}","{cut}")')
        else:
            raise ValueError(f"Unknown draw mode: {draw_mode}")

    def _fill_spectrum(self, ch_id: int, vals: np.ndarray,
                        n_bins: int, draw_expr: str = ""):
        if len(vals) == 0:
            return
        adc_min = float(vals.min())
        adc_max = float(vals.max())
        if adc_min == adc_max:
            adc_max = adc_min + 1.0
        counts, edges = np.histogram(vals, bins=n_bins,
                                      range=(adc_min, adc_max))
        centers = 0.5 * (edges[:-1] + edges[1:])
        self.spectra[ch_id] = ChannelSpectrum(
            channel_id=ch_id,
            name=f"ch_{ch_id:04d}",
            bin_centers=centers,
            counts=counts,
            n_entries=len(vals),
            source="ttree",
            draw_expr=draw_expr
        )

    # ================================================================== #
    # uproot backend
    # ================================================================== #

    def _open_uproot(self, filename: str) -> dict:
        import uproot
        self._file = uproot.open(filename)
        info = {"trees": [], "histograms": [],
                "filename": filename, "backend": "uproot"}
        for name, obj in self._file.items():
            clean = name.split(";")[0]
            cls   = type(obj).__name__
            if "TTree" in cls:
                try:
                    info["trees"].append({
                        "name":     clean,
                        "branches": list(obj.keys()),
                        "entries":  int(obj.num_entries)
                    })
                except Exception:
                    pass
            elif any(h in cls for h in ("TH1", "Histogram")):
                try:
                    vals, edges = obj.to_numpy()
                    info["histograms"].append({
                        "name":    clean,
                        "nbins":   len(vals),
                        "xmin":    float(edges[0]),
                        "xmax":    float(edges[-1]),
                        "entries": int(obj.member("fEntries"))
                    })
                except Exception:
                    pass
        return info

    def _load_th1_uproot(self, hist_names: Optional[list]):
        import re
        all_items = {n.split(";")[0]: o for n, o in self._file.items()}
        all_hist_names = [
            n for n, o in all_items.items()
            if any(h in type(o).__name__ for h in ("TH1", "Histogram"))
        ]
        targets = hist_names if hist_names else all_hist_names
        for idx, name in enumerate(targets):
            obj = all_items.get(name)
            if obj is None:
                continue
            try:
                counts, edges = obj.to_numpy()
                centers = 0.5 * (edges[:-1] + edges[1:])
                import re as re2
                m     = re2.search(r"(\d+)$", name)
                ch_id = int(m.group(1)) if m else idx
                self.spectra[ch_id] = ChannelSpectrum(
                    channel_id=ch_id, name=name,
                    bin_centers=centers, counts=counts,
                    n_entries=int(obj.member("fEntries")), source="th1")
            except Exception:
                continue

    def _load_ttree_uproot(self, tree_name, channel_branch, adc_branch,
                            n_bins, draw_mode, custom_expr,
                            channel_ids, ch_first, ch_last, ch_step,
                            max_entries):
        """
        uproot backend: supports filter and array modes.
        Custom mode with expressions falls back to filter mode with a warning.
        """
        tree = None
        for name, obj in self._file.items():
            if name.split(";")[0] == tree_name:
                tree = obj
                break
        if tree is None:
            raise ValueError(f"TTree '{tree_name}' not found.")

        if draw_mode == "array":
            # Array branch — read the 2D array and index per channel
            entry_kw = {"entry_stop": max_entries} if max_entries > 0 else {}
            arr = tree[adc_branch].array(library="np", **entry_kw)
            # arr shape: (n_events, n_channels) or similar
            if arr.ndim == 1:
                raise ValueError(
                    f"Branch '{adc_branch}' is not an array branch. "
                    "Use 'filter' mode instead.")
            discovered = list(range(arr.shape[1]))
            targets = self._resolve_channel_ids(
                channel_ids, ch_first, ch_last, ch_step, discovered)
            for ch_id in targets:
                if ch_id >= arr.shape[1]:
                    continue
                vals = arr[:, ch_id].astype(float)
                self._fill_spectrum(ch_id, vals, n_bins,
                                     draw_expr=f"{adc_branch}[{ch_id}]")

        else:
            # filter mode (and fallback for custom)
            if draw_mode == "custom":
                import warnings
                warnings.warn(
                    "uproot backend: custom expressions not supported. "
                    "Falling back to filter mode.", RuntimeWarning)

            entry_kw = {"entry_stop": max_entries} if max_entries > 0 else {}
            arrays  = tree.arrays(
                [channel_branch, adc_branch], library="np", **entry_kw)
            ch_arr  = arrays[channel_branch].astype(int)
            adc_arr = arrays[adc_branch].astype(float)

            discovered = sorted(set(ch_arr.tolist()))
            targets    = self._resolve_channel_ids(
                channel_ids, ch_first, ch_last, ch_step, discovered)

            for ch_id in targets:
                mask = ch_arr == ch_id
                vals = adc_arr[mask]
                self._fill_spectrum(
                    ch_id, vals, n_bins,
                    draw_expr=f'Draw("{adc_branch}","{channel_branch}=={ch_id}")')
