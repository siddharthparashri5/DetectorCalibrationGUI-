"""
ROOTFileLoader
==============
Backends: PyROOT (preferred) or uproot (fallback).

Draw modes for TTree loading:
  filter   — read both branches once, split per channel in numpy
  array    — Draw("energy[N]") per channel (array-indexed branch)
  custom   — user expression with %d replaced by channel number

Channel-range resolution priority
  1. Explicit channel_ids list
  2. ch_first … ch_last range  (when ch_last >= ch_first)
  3. Auto-discovered from data  (filter mode only)
"""

from __future__ import annotations
import sys, os, subprocess
import numpy as np
from dataclasses import dataclass
from typing import Optional

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
        lib_dir = os.path.normpath(os.path.join(root_py_path, "..", "lib"))
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


DRAW_MODES = {
    "filter": 'Read both branches once, split per channel in numpy',
    "array":  'Draw("energy[N]") — array branch indexed by channel',
    "custom": 'Custom expression  (%d replaced by channel number)',
}

_PYROOT_CHUNK = 500_000


@dataclass
class ChannelSpectrum:
    channel_id:  int
    name:        str
    bin_centers: np.ndarray
    counts:      np.ndarray
    n_entries:   int
    source:      str = "unknown"
    draw_expr:   str = ""


class ROOTFileLoader:

    def __init__(self):
        self.filename:  str = ""
        self.mode:      str = ""
        self.backend:   str = ""
        self.draw_mode: str = "filter"
        self.spectra:   dict = {}
        self._file = None

    def open(self, filename: str) -> dict:
        self.filename = filename
        self.backend  = _detect_backend()
        return self._open_pyroot(filename) if self.backend == "pyroot" \
               else self._open_uproot(filename)

    def load_from_th1(self, hist_names=None):
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
                         channel_ids=None,
                         ch_first:       int  = 0,
                         ch_last:        int  = -1,
                         ch_step:        int  = 1,
                         max_entries:    int  = 0):
        self.mode      = "ttree"
        self.draw_mode = draw_mode
        self.spectra   = {}
        if self.backend == "pyroot":
            self._load_ttree_pyroot(
                tree_name, channel_branch, adc_branch,
                n_bins, draw_mode, custom_expr,
                channel_ids, ch_first, ch_last, ch_step, max_entries)
        else:
            self._load_ttree_uproot(
                tree_name, channel_branch, adc_branch,
                n_bins, draw_mode, custom_expr,
                channel_ids, ch_first, ch_last, ch_step, max_entries)

    def get_channel_ids(self) -> list:
        return sorted(self.spectra.keys())

    def get_spectrum(self, channel_id: int):
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

    def _resolve_channel_ids(self, channel_ids, ch_first, ch_last, ch_step, discovered):
        """
        Priority:
          1. Explicit channel_ids list
          2. ch_first..ch_last inclusive range (when ch_last >= ch_first)
             FIX: was ch_last >= 0, now ch_last >= ch_first so that
             ch_first=10, ch_last=-1 correctly falls through to auto-discover
          3. Auto-discovered list from data
        """
        if channel_ids:
            return sorted(channel_ids)
        if ch_last >= ch_first:
            return list(range(ch_first, ch_last + 1, max(1, ch_step)))
        if discovered:
            return sorted(discovered)
        return []

    def _open_pyroot(self, filename: str) -> dict:
        import ROOT
        self._file = ROOT.TFile.Open(filename, "READ")
        if not self._file or self._file.IsZombie():
            raise IOError(f"Cannot open ROOT file: {filename}")
        info = {"trees": [], "histograms": [], "filename": filename, "backend": "pyroot"}
        for key in self._file.GetListOfKeys():
            cls  = key.GetClassName()
            name = key.GetName()
            if cls in ("TTree", "TNtuple"):
                tree     = self._file.Get(name)
                branches = [b.GetName() for b in tree.GetListOfBranches()]
                info["trees"].append({
                    "name":     name,
                    "branches": branches,
                    "entries":  int(tree.GetEntries()),
                })
            elif cls in ("TH1F", "TH1D", "TH1I", "TH1"):
                h = self._file.Get(name)
                info["histograms"].append({
                    "name":    name,
                    "nbins":   h.GetNbinsX(),
                    "xmin":    h.GetXaxis().GetXmin(),
                    "xmax":    h.GetXaxis().GetXmax(),
                    "entries": int(h.GetEntries()),
                })
        return info

    def _load_th1_pyroot(self, hist_names):
        import re
        all_keys = [k.GetName() for k in self._file.GetListOfKeys()
                    if k.GetClassName() in ("TH1F", "TH1D", "TH1I", "TH1")]
        targets = hist_names if hist_names else all_keys
        for idx, name in enumerate(targets):
            h = self._file.Get(name)
            if not h:
                continue
            n       = h.GetNbinsX()
            centers = np.array([h.GetXaxis().GetBinCenter(i + 1) for i in range(n)])
            counts  = np.array([h.GetBinContent(i + 1) for i in range(n)])
            m       = re.search(r"(\d+)$", name)
            ch_id   = int(m.group(1)) if m else idx
            self.spectra[ch_id] = ChannelSpectrum(
                channel_id=ch_id, name=name,
                bin_centers=centers, counts=counts,
                n_entries=int(h.GetEntries()), source="th1")

    def _load_ttree_pyroot(self, tree_name, channel_branch, adc_branch,
                            n_bins, draw_mode, custom_expr,
                            channel_ids, ch_first, ch_last, ch_step, max_entries):
        import ROOT
        tree = self._file.Get(tree_name)
        if not tree:
            raise ValueError(f"TTree '{tree_name}' not found.")
        ROOT.gROOT.cd()
        n_total     = int(tree.GetEntries())
        entry_limit = int(max_entries) if max_entries > 0 else n_total

        if draw_mode == "filter":
            ch_arr, adc_arr = self._pyroot_read_two_branches(
                tree, channel_branch, adc_branch, entry_limit)
            if len(ch_arr) == 0:
                raise ValueError(f"No data read from '{channel_branch}' / '{adc_branch}'.")
            discovered = sorted(set(ch_arr.tolist()))
            targets    = self._resolve_channel_ids(
                channel_ids, ch_first, ch_last, ch_step, discovered)
            if not targets:
                raise ValueError(
                    "No target channels could be determined.\n"
                    "Set First/Last channel in the load dialog, or leave "
                    "both at default to auto-discover.")
            data_set = set(discovered)
            present  = [c for c in targets if c in data_set]
            if not present:
                raise ValueError(
                    f"Requested channels {targets[:8]}… not found in data.\n"
                    f"Branch '{channel_branch}' contains: {discovered[:8]}…")
            for ch_id in present:
                mask = ch_arr == ch_id
                vals = adc_arr[mask]
                if len(vals) == 0:
                    continue
                self._fill_spectrum(ch_id, vals, n_bins,
                                     draw_expr=f'{adc_branch}[{channel_branch}=={ch_id}]')

        elif draw_mode == "array":
            targets = self._resolve_channel_ids(
                channel_ids, ch_first, ch_last, ch_step, None)
            if not targets:
                raise ValueError(
                    "Array mode requires a channel range.\n"
                    "Set First and Last channel in the load dialog.")
            for ch_id in targets:
                expr = (adc_branch.replace("%d", str(ch_id))
                        if "%d" in adc_branch
                        else f"{adc_branch}[{ch_id}]")
                vals = self._pyroot_draw_chunked(tree, expr, "", entry_limit)
                if vals is None or len(vals) == 0:
                    continue
                self._fill_spectrum(ch_id, vals, n_bins, draw_expr=f'Draw("{expr}")')

        elif draw_mode == "custom":
            if not custom_expr.strip():
                raise ValueError(
                    "Custom draw mode requires an expression.\n"
                    'e.g. "energy[%d]"  or  "sqrt(adc),channelID==%d"')
            targets = self._resolve_channel_ids(
                channel_ids, ch_first, ch_last, ch_step, None)
            if not targets:
                raise ValueError(
                    "Custom mode requires a channel range.\n"
                    "Set First and Last channel in the load dialog.")
            for ch_id in targets:
                filled = custom_expr.replace("%d", str(ch_id))
                parts  = [p.strip() for p in filled.split(",", 1)]
                var    = parts[0]
                cut    = parts[1] if len(parts) > 1 else ""
                vals   = self._pyroot_draw_chunked(tree, var, cut, entry_limit)
                if vals is None or len(vals) == 0:
                    continue
                self._fill_spectrum(ch_id, vals, n_bins,
                                     draw_expr=f'Draw("{var}","{cut}")')
        else:
            raise ValueError(f"Unknown draw mode: '{draw_mode}'")

    def _pyroot_read_two_branches(self, tree, branch_ch, branch_adc, entry_limit):
        n_total    = int(tree.GetEntries())
        n_read     = min(entry_limit, n_total)
        chunk_size = min(_PYROOT_CHUNK, n_read)
        ch_parts, adc_parts = [], []
        offset = 0
        while offset < n_read:
            this_chunk = min(chunk_size, n_read - offset)
            tree.SetEstimate(this_chunk + 1)
            n = tree.Draw(f"{branch_adc}:{branch_ch}", "", "goff", this_chunk, offset)
            if n > 0:
                adc_parts.append(np.frombuffer(tree.GetV1(), dtype=np.float64, count=n).copy())
                ch_parts.append(np.frombuffer(tree.GetV2(), dtype=np.float64, count=n).copy())
            offset += this_chunk
        if not ch_parts:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float64)
        return (np.concatenate(ch_parts).astype(np.int32),
                np.concatenate(adc_parts))

    def _pyroot_draw_chunked(self, tree, var, cut, entry_limit):
        n_total    = int(tree.GetEntries())
        n_read     = min(entry_limit, n_total)
        chunk_size = min(_PYROOT_CHUNK, n_read)
        parts, offset = [], 0
        while offset < n_read:
            this_chunk = min(chunk_size, n_read - offset)
            tree.SetEstimate(this_chunk + 1)
            n = tree.Draw(var, cut, "goff", this_chunk, offset)
            if n > 0:
                parts.append(np.frombuffer(tree.GetV1(), dtype=np.float64, count=n).copy())
            offset += this_chunk
        return np.concatenate(parts) if parts else None

    def _fill_spectrum(self, ch_id, vals, n_bins, draw_expr=""):
        if len(vals) == 0:
            return
        adc_min = float(vals.min())
        adc_max = float(vals.max())
        if adc_min == adc_max:
            adc_max = adc_min + 1.0
        counts, edges = np.histogram(vals, bins=n_bins, range=(adc_min, adc_max))
        centers = 0.5 * (edges[:-1] + edges[1:])
        self.spectra[ch_id] = ChannelSpectrum(
            channel_id  = ch_id,
            name        = f"ch_{ch_id:04d}",
            bin_centers = centers,
            counts      = counts,
            n_entries   = len(vals),
            source      = "ttree",
            draw_expr   = draw_expr,
        )

    def _open_uproot(self, filename: str) -> dict:
        import uproot
        self._file = uproot.open(filename)
        info = {"trees": [], "histograms": [], "filename": filename, "backend": "uproot"}
        for name, obj in self._file.items():
            clean = name.split(";")[0]
            cls   = type(obj).__name__
            if "TTree" in cls:
                try:
                    info["trees"].append({
                        "name":     clean,
                        "branches": list(obj.keys()),
                        "entries":  int(obj.num_entries),
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
                        "entries": int(obj.member("fEntries")),
                    })
                except Exception:
                    pass
        return info

    def _load_th1_uproot(self, hist_names):
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
                m     = re.search(r"(\d+)$", name)
                ch_id = int(m.group(1)) if m else idx
                self.spectra[ch_id] = ChannelSpectrum(
                    channel_id=ch_id, name=name,
                    bin_centers=centers, counts=counts,
                    n_entries=int(obj.member("fEntries")), source="th1")
            except Exception:
                continue

    def _load_ttree_uproot(self, tree_name, channel_branch, adc_branch,
                            n_bins, draw_mode, custom_expr,
                            channel_ids, ch_first, ch_last, ch_step, max_entries):
        tree = None
        for name, obj in self._file.items():
            if name.split(";")[0] == tree_name:
                tree = obj
                break
        if tree is None:
            raise ValueError(f"TTree '{tree_name}' not found.")

        if draw_mode == "array":
            entry_kw = {"entry_stop": max_entries} if max_entries > 0 else {}
            arr = tree[adc_branch].array(library="np", **entry_kw)
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
            if draw_mode == "custom":
                import warnings
                warnings.warn("uproot: custom expressions not supported, using filter mode.",
                              RuntimeWarning)
            entry_kw = {"entry_stop": max_entries} if max_entries > 0 else {}
            ch_parts, adc_parts = [], []
            for batch in tree.iterate([channel_branch, adc_branch],
                                       library="np", step_size="50MB", **entry_kw):
                ch_parts.append(batch[channel_branch].astype(np.int32))
                adc_parts.append(batch[adc_branch].astype(np.float64))
            if not ch_parts:
                raise ValueError(f"No data from '{channel_branch}' / '{adc_branch}'.")
            ch_arr  = np.concatenate(ch_parts)
            adc_arr = np.concatenate(adc_parts)
            discovered = sorted(set(ch_arr.tolist()))
            targets    = self._resolve_channel_ids(
                channel_ids, ch_first, ch_last, ch_step, discovered)
            if not targets:
                raise ValueError("No target channels could be determined.")
            for ch_id in targets:
                mask = ch_arr == ch_id
                vals = adc_arr[mask]
                if len(vals) == 0:
                    continue
                self._fill_spectrum(ch_id, vals, n_bins,
                                     draw_expr=f'Draw("{adc_branch}","{channel_branch}=={ch_id}")')
