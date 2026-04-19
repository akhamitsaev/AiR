from typing import Callable

import numpy as np
from scipy.optimize import minimize_scalar

from lib.broom_racing import Configuration, XYZConfiguration


KAPPA_MAX = 1.0
RHO_MIN = 1.0 / KAPPA_MAX
PHI_MIN = -np.pi / 4
PHI_MAX = np.pi / 4
EPS = 1e-12


def _mod2pi(x: float) -> float:
    return x % (2.0 * np.pi)


def _wrap_angle(x: float) -> float:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _x(c): return float(c.x)
def _y(c): return float(c.y)
def _z(c): return float(c.z)
def _theta(c): return float(c.theta)
def _phi(c): return float(c.phi)


def _cfg(x, y, z, theta, phi):
    return Configuration(float(x), float(y), float(z), float(theta), float(phi))


def _vec_to_angles(dx: float, dy: float, dz: float):
    theta = np.arctan2(dy, dx)
    phi = np.arctan2(dz, np.hypot(dx, dy))
    return float(theta), float(phi)


def _goal_dir(goal: Configuration) -> np.ndarray:
    ct, st = np.cos(goal.theta), np.sin(goal.theta)
    cp, sp = np.cos(goal.phi), np.sin(goal.phi)
    return np.array([ct * cp, st * cp, sp], dtype=float)


# ============================================================
# 2D Dubins path
# ============================================================

def _dubins_LSL(alpha, beta, d):
    p2 = 2 + d * d - 2 * np.cos(alpha - beta) + 2 * d * (np.sin(alpha) - np.sin(beta))
    if p2 < 0:
        return None
    tmp = np.arctan2(np.cos(beta) - np.cos(alpha), d + np.sin(alpha) - np.sin(beta))
    t = _mod2pi(-alpha + tmp)
    p = np.sqrt(p2)
    q = _mod2pi(beta - tmp)
    return t, p, q


def _dubins_RSR(alpha, beta, d):
    p2 = 2 + d * d - 2 * np.cos(alpha - beta) + 2 * d * (-np.sin(alpha) + np.sin(beta))
    if p2 < 0:
        return None
    tmp = np.arctan2(np.cos(alpha) - np.cos(beta), d - np.sin(alpha) + np.sin(beta))
    t = _mod2pi(alpha - tmp)
    p = np.sqrt(p2)
    q = _mod2pi(-beta + tmp)
    return t, p, q


def _dubins_LSR(alpha, beta, d):
    p2 = -2 + d * d + 2 * np.cos(alpha - beta) + 2 * d * (np.sin(alpha) + np.sin(beta))
    if p2 < 0:
        return None
    p = np.sqrt(p2)
    tmp = np.arctan2(-np.cos(alpha) - np.cos(beta), d + np.sin(alpha) + np.sin(beta)) - np.arctan2(-2.0, p)
    t = _mod2pi(-alpha + tmp)
    q = _mod2pi(-_mod2pi(beta) + tmp)
    return t, p, q


def _dubins_RSL(alpha, beta, d):
    p2 = d * d - 2 + 2 * np.cos(alpha - beta) - 2 * d * (np.sin(alpha) + np.sin(beta))
    if p2 < 0:
        return None
    p = np.sqrt(p2)
    tmp = np.arctan2(np.cos(alpha) + np.cos(beta), d - np.sin(alpha) - np.sin(beta)) - np.arctan2(2.0, p)
    t = _mod2pi(alpha - tmp)
    q = _mod2pi(beta - tmp)
    return t, p, q


def _dubins_RLR(alpha, beta, d):
    tmp = (6.0 - d * d + 2 * np.cos(alpha - beta) + 2 * d * (np.sin(alpha) - np.sin(beta))) / 8.0
    if abs(tmp) > 1.0:
        return None
    p = _mod2pi(2.0 * np.pi - np.arccos(tmp))
    tmp2 = np.arctan2(np.cos(alpha) - np.cos(beta), d - np.sin(alpha) + np.sin(beta))
    t = _mod2pi(alpha - tmp2 + 0.5 * p)
    q = _mod2pi(alpha - beta - t + p)
    return t, p, q


def _dubins_LRL(alpha, beta, d):
    tmp = (6.0 - d * d + 2 * np.cos(alpha - beta) + 2 * d * (-np.sin(alpha) + np.sin(beta))) / 8.0
    if abs(tmp) > 1.0:
        return None
    p = _mod2pi(2.0 * np.pi - np.arccos(tmp))
    tmp2 = np.arctan2(np.cos(alpha) - np.cos(beta), d + np.sin(alpha) - np.sin(beta))
    t = _mod2pi(-alpha - tmp2 + 0.5 * p)
    q = _mod2pi(_mod2pi(beta) - alpha - t + p)
    return t, p, q


_DUBINS_FNS = {
    "LSL": _dubins_LSL,
    "RSR": _dubins_RSR,
    "LSR": _dubins_LSR,
    "RSL": _dubins_RSL,
    "RLR": _dubins_RLR,
    "LRL": _dubins_LRL,
}


class _Dubins2D:
    def __init__(self, qi: np.ndarray, qf: np.ndarray, rho: float):
        self.qi = np.asarray(qi, dtype=float)
        self.qf = np.asarray(qf, dtype=float)
        self.rho = float(rho)

        dx = self.qf[0] - self.qi[0]
        dy = self.qf[1] - self.qi[1]
        D = np.hypot(dx, dy)
        d = D / self.rho
        theta = np.arctan2(dy, dx)
        alpha = _mod2pi(self.qi[2] - theta)
        beta = _mod2pi(self.qf[2] - theta)

        best = None
        for case, fn in _DUBINS_FNS.items():
            out = fn(alpha, beta, d)
            if out is None:
                continue
            t, p, q = out
            L = self.rho * (t + p + q)
            if best is None or L < best["length"]:
                best = {
                    "case": case,
                    "t": float(t),
                    "p": float(p),
                    "q": float(q),
                    "length": float(L),
                    "theta": float(theta),
                    "alpha": float(alpha),
                }

        if best is None:
            raise RuntimeError("No 2D Dubins path found")

        self.case = best["case"]
        self.t = best["t"]
        self.p = best["p"]
        self.q = best["q"]
        self.length = best["length"]
        self._theta_ref = best["theta"]
        self._alpha = best["alpha"]

    def _segment_step(self, x, y, th, seg_type, seg_len):
        if seg_type == "L":
            x2 = x + np.sin(th + seg_len) - np.sin(th)
            y2 = y - np.cos(th + seg_len) + np.cos(th)
            th2 = th + seg_len
        elif seg_type == "R":
            x2 = x - np.sin(th - seg_len) + np.sin(th)
            y2 = y + np.cos(th - seg_len) - np.cos(th)
            th2 = th - seg_len
        elif seg_type == "S":
            x2 = x + np.cos(th) * seg_len
            y2 = y + np.sin(th) * seg_len
            th2 = th
        else:
            raise ValueError(seg_type)
        return x2, y2, th2

    def at(self, s: float) -> np.ndarray:
        s = float(np.clip(s, 0.0, self.length))
        sn = s / self.rho

        seg_types = list(self.case)
        seg_lengths = [self.t, self.p, self.q]

        x = 0.0
        y = 0.0
        th = self._alpha

        for stype, slen in zip(seg_types, seg_lengths):
            if sn <= 0:
                break
            ds = min(sn, slen)
            x, y, th = self._segment_step(x, y, th, stype, ds)
            sn -= ds

        c = np.cos(self._theta_ref)
        s0 = np.sin(self._theta_ref)
        wx = self.qi[0] + self.rho * (c * x - s0 * y)
        wy = self.qi[1] + self.rho * (s0 * x + c * y)
        wth = _wrap_angle(th + self._theta_ref)
        return np.array([wx, wy, wth], dtype=float)


# ============================================================
# Decoupled 3D planner in the style of Vána 2020
# ============================================================

class _Dubins3DDecoupled:
    def __init__(self, start: Configuration, goal: Configuration, rho_min: float = RHO_MIN):
        self.start = start
        self.goal = goal
        self.rho_min = float(rho_min)

        self.Dlat, self.Dlon = self._construct()

        self.length = self.Dlon.length

    def _try_construct(self, rho_h: float):
        qi_xy = np.array([self.start.x, self.start.y, self.start.theta], dtype=float)
        qf_xy = np.array([self.goal.x, self.goal.y, self.goal.theta], dtype=float)
        Dlat = _Dubins2D(qi_xy, qf_xy, rho_h)

        val = 1.0 / (self.rho_min * self.rho_min) - 1.0 / (rho_h * rho_h)
        if val <= 1e-10:
            return None

        kappa_v = np.sqrt(val)
        rho_v = 1.0 / kappa_v

        qi_v = np.array([0.0, self.start.z, self.start.phi], dtype=float)
        qf_v = np.array([Dlat.length, self.goal.z, self.goal.phi], dtype=float)
        Dlon = _Dubins2D(qi_v, qf_v, rho_v)

        # Following the reference implementation logic: reject cyclic vertical solutions.
        if Dlon.case in ("RLR", "LRL"):
            return None

        # Enforce pitch limits by sampling vertical profile.
        n = max(200, int(np.ceil(Dlon.length / 0.02)))
        for i in range(n + 1):
            qv = Dlon.at(Dlon.length * i / n)
            phi = qv[2]
            if phi < PHI_MIN - 1e-9 or phi > PHI_MAX + 1e-9:
                return None

        return Dlat, Dlon

    def _construct(self):
        # Find a feasible horizontal radius upper bound.
        a = self.rho_min
        b = self.rho_min

        fa = self._try_construct(a)
        fb = self._try_construct(b)

        while fb is None and b < 1000.0 * self.rho_min:
            b *= 2.0
            fb = self._try_construct(b)

        if fb is None:
            raise RuntimeError("No feasible decoupled 3D Dubins path")

        if fa is None:
            # Search on [rho_min, b] for minimum vertical length.
            def objective(rh):
                out = self._try_construct(float(rh))
                if out is None:
                    return 1e9
                _, Dlon = out
                return Dlon.length

            res = minimize_scalar(objective, bounds=(self.rho_min, b), method="bounded", options={"xatol": 1e-6, "maxiter": 80})
            best = self._try_construct(float(res.x))
            if best is None:
                best = fb
            return best

        # Both feasible already -> optimize on [a, b]
        def objective(rh):
            out = self._try_construct(float(rh))
            if out is None:
                return 1e9
            _, Dlon = out
            return Dlon.length

        res = minimize_scalar(objective, bounds=(a, b), method="bounded", options={"xatol": 1e-6, "maxiter": 80})
        best = self._try_construct(float(res.x))
        if best is None:
            return fa if fa[1].length <= fb[1].length else fb
        return best

    def at(self, s: float) -> Configuration:
        s = float(np.clip(s, 0.0, self.length))
        qv = self.Dlon.at(s)
        sigma = qv[0]
        z = qv[1]
        phi = qv[2]

        qh = self.Dlat.at(sigma)
        x = qh[0]
        y = qh[1]
        theta = qh[2]

        return _cfg(x, y, z, theta, phi)


def _make_curve_from_planner(planner: _Dubins3DDecoupled):
    total = planner.length

    def curve(s: np.ndarray) -> Configuration:
        arr = np.asarray(s, dtype=float)
        flat = arr.reshape(-1)

        xs = np.empty_like(flat, dtype=float)
        ys = np.empty_like(flat, dtype=float)
        zs = np.empty_like(flat, dtype=float)
        thetas = np.empty_like(flat, dtype=float)
        phis = np.empty_like(flat, dtype=float)

        for i, si in enumerate(flat):
            cfg = planner.at(float(np.clip(si, 0.0, 1.0)) * total)
            xs[i] = cfg.x
            ys[i] = cfg.y
            zs[i] = cfg.z
            thetas[i] = cfg.theta
            phis[i] = cfg.phi

        if flat.size == 1:
            return _cfg(xs[0], ys[0], zs[0], thetas[0], phis[0])

        shp = arr.shape
        return Configuration(
            xs.reshape(shp),
            ys.reshape(shp),
            zs.reshape(shp),
            thetas.reshape(shp),
            phis.reshape(shp),
        )

    curve.total_length = total
    return curve


# ============================================================
# Public API
# ============================================================

def gate_pass(
    start: Configuration,
    goal: Configuration,
) -> Callable[[np.ndarray], Configuration]:
    # Exact straight shortcut
    d = goal.position() - start.position()
    if np.linalg.norm(d) > EPS:
        th, ph = _vec_to_angles(d[0], d[1], d[2])
        if (
            abs(_wrap_angle(th - start.theta)) < 1e-9
            and abs(ph - start.phi) < 1e-9
            and abs(_wrap_angle(goal.theta - start.theta)) < 1e-9
            and abs(goal.phi - start.phi) < 1e-9
        ):
            L = float(np.linalg.norm(d))

            def curve(s: np.ndarray) -> Configuration:
                arr = np.asarray(s, dtype=float)
                flat = arr.reshape(-1)
                t = np.clip(flat, 0.0, 1.0)
                xs = start.x + t * (goal.x - start.x)
                ys = start.y + t * (goal.y - start.y)
                zs = start.z + t * (goal.z - start.z)
                ths = np.full_like(xs, start.theta)
                phs = np.full_like(xs, start.phi)
                if flat.size == 1:
                    return _cfg(xs[0], ys[0], zs[0], ths[0], phs[0])
                shp = arr.shape
                return Configuration(xs.reshape(shp), ys.reshape(shp), zs.reshape(shp), ths.reshape(shp), phs.reshape(shp))

            curve.total_length = L
            return curve

    planner = _Dubins3DDecoupled(start, goal, rho_min=RHO_MIN)
    return _make_curve_from_planner(planner)


def catch_snitch(
    start: Configuration,
    goal_xyz: XYZConfiguration,
) -> Callable[[np.ndarray], Configuration]:
    d = goal_xyz.position() - start.position()
    th_goal, ph_goal = _vec_to_angles(d[0], d[1], d[2])

    # Heuristic final direction: line-of-sight to the goal.
    goal_cfg = Configuration(goal_xyz.x, goal_xyz.y, goal_xyz.z, th_goal, ph_goal)
    planner = _Dubins3DDecoupled(start, goal_cfg, rho_min=RHO_MIN)
    return _make_curve_from_planner(planner)


def catch_ball_and_gate(
    start: Configuration,
    intermediate_goal_xyz: XYZConfiguration,
    final_goal: Configuration,
) -> Callable[[np.ndarray], Configuration]:
    first = catch_snitch(start, intermediate_goal_xyz)
    mid = first(np.array([1.0]))
    second = gate_pass(mid, final_goal)

    total = first.total_length + second.total_length

    def curve(s: np.ndarray) -> Configuration:
        arr = np.asarray(s, dtype=float)
        flat = arr.reshape(-1)

        xs = np.empty_like(flat, dtype=float)
        ys = np.empty_like(flat, dtype=float)
        zs = np.empty_like(flat, dtype=float)
        thetas = np.empty_like(flat, dtype=float)
        phis = np.empty_like(flat, dtype=float)

        for i, si in enumerate(flat):
            l = np.clip(float(si), 0.0, 1.0) * total
            if l <= first.total_length:
                cfg = first(np.array([l / max(first.total_length, EPS)]))
            else:
                cfg = second(np.array([(l - first.total_length) / max(second.total_length, EPS)]))
            xs[i] = cfg.x
            ys[i] = cfg.y
            zs[i] = cfg.z
            thetas[i] = cfg.theta
            phis[i] = cfg.phi

        if flat.size == 1:
            return _cfg(xs[0], ys[0], zs[0], thetas[0], phis[0])

        shp = arr.shape
        return Configuration(
            xs.reshape(shp),
            ys.reshape(shp),
            zs.reshape(shp),
            thetas.reshape(shp),
            phis.reshape(shp),
        )

    curve.total_length = total
    return curve
