"""Embodiment loader for robots/<name>/embodiment.yaml (JSON content). Stdlib only (runs in both pythons).

Single source of truth for joint names/order, action spaces, angle offsets and URDF limits of the dual
YellowArm+ORCA robot. Everything (sim, dataset, policy server, robot-agent) must take joint order from here.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import numpy as np

from mhh_paths import DUAL_EMBODIMENT, DUAL_URDF


class Embodiment:
    def __init__(self, path=DUAL_EMBODIMENT):
        self.path = path
        self.spec = json.load(open(path, encoding="utf-8"))
        self.id = self.spec["id"]
        self.control_hz = self.spec.get("control_hz", 25)
        self.groups = self.spec["joint_groups"]
        self.offsets = self.spec.get("source_angle_offsets_rad", {})
        self.urdf = path.parent / self.spec["urdf"]
        self.urdf_limits = self._parse_limits()  # as written in the URDF
        self.overrides = self.spec.get("sim_overrides", {})
        self.limits = self._apply_overrides(self.urdf_limits)  # what the simulation uses (URDF + sim_overrides)

    def _parse_limits(self):
        out = {}
        for j in ET.parse(self.urdf).getroot().findall("joint"):
            if j.get("type") in ("revolute", "prismatic", "continuous"):
                l = j.find("limit")
                out[j.get("name")] = dict(lower=float(l.get("lower", -np.inf)), upper=float(l.get("upper", np.inf)),
                                          effort=float(l.get("effort", 0)), velocity=float(l.get("velocity", 0)))
        return out

    def _apply_overrides(self, lim):
        """sim_overrides in embodiment.yaml: joint_limits_rad {name: [lo, hi]}, effort_nm {name: Nm}."""
        out = {k: dict(v) for k, v in lim.items()}
        for n, (lo, hi) in self.overrides.get("joint_limits_rad", {}).items():
            out[n].update(lower=float(lo), upper=float(hi))
        for n, e in self.overrides.get("effort_nm", {}).items():
            out[n]["effort"] = float(e)
        return out

    def action_space(self, name="joint_full"):
        """Ordered joint names of an action space (concatenation of its groups)."""
        return [j for g in self.spec["action_spaces"][name]["groups"] for j in self.groups[g]]

    def lower_upper(self, names):
        return (np.array([self.limits[n]["lower"] for n in names]), np.array([self.limits[n]["upper"] for n in names]))

    def clamp(self, names, q):
        lo, hi = self.lower_upper(names)
        return np.clip(q, lo, hi)

    def home(self, names, margin=0.05):
        """Neutral pose: the CAD/URDF zero clipped into [lower+margin, upper-margin].

        Not the range centre: J1 (shoulder pitch) spans ~352 deg and its centre is the arm pointing straight up.
        Zero is arms hanging down (J1 then sits 8 deg + margin from it, toward +Y with sim_overrides, see
        tasks/URDF_CAD_JOINT_MAPPING.md)."""
        lo, hi = self.lower_upper(names)
        return np.clip(np.zeros(len(names)), lo + margin, hi - margin)


if __name__ == "__main__":
    e = Embodiment()
    n = e.action_space()
    print(e.id, "joints:", len(n), "ctrl_hz:", e.control_hz, "urdf:", e.urdf.name)
    lo, hi = e.lower_upper(n)
    bad = [x for x in n if x not in e.limits]
    print("missing limits:", bad)
    h = e.home(n)
    print("home inside limits:", bool(np.all(h >= lo) and np.all(h <= hi)))
    print("arm joints with 0 outside range:", [x for x in n if "yellowarm" in x and not (e.limits[x]["lower"] <= 0 <= e.limits[x]["upper"])])
