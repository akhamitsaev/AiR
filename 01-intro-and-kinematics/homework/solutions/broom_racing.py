from typing import Callable

import numpy as np
from scipy.optimize import least_squares

from lib.broom_racing import Configuration, XYZConfiguration


PHI_MIN = -np.pi / 4
PHI_MAX = np.pi / 4
EPS = 1e-12
LEN_REG = 1e-4


def _wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


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


def _goal_side(goal: Configuration) -> np.ndarray:
    return np.array([-np.sin(goal.theta), np.cos(goal.theta), 0.0], dtype=float)


def _aligned_with_segment(start: Configuration, goal_pos: np.ndarray, tol: float = 1e-9) -> bool:
    d = goal_pos - start.position()
    if np.linalg.norm(d) < EPS:
        return False
    th, ph = _vec_to_angles(d[0], d[1], d[2])
    return abs(_wrap_angle(th - _theta(start))) < tol and abs(ph - _phi(start)) < tol


def _angle_error(c: Configuration, goal: Configuration) -> float:
    dth = _wrap_angle(c.theta - goal.theta)
    dph = c.phi - goal.phi
    return float(np.sqrt(dth * dth + dph * dph))


def _cfg_error(c: Configuration, goal: Configuration) -> float:
    pos = np.linalg.norm(c.position() - goal.position())
    ang = _angle_error(c, goal)
    return float(pos + ang)


# ============================================================
# Exact motion primitives for the given EOM
# ============================================================

def _turn_full(cfg: Configuration, delta_theta: float) -> Configuration:
    """
    Pure horizontal turn:
        u1 = sign(delta_theta), u2 = 0
    Then dtheta/ds = u1 / cos(phi), phi = const.
    Segment length = |delta_theta| * cos(phi).
    """
    if abs(delta_theta) < EPS:
        return cfg

    x, y, z, th, ph = _x(cfg), _y(cfg), _z(cfg), _theta(cfg), _phi(cfg)
    u = np.sign(delta_theta)
    th2 = th + delta_theta
    c2 = np.cos(ph) ** 2

    x2 = x + u * c2 * (np.sin(th2) - np.sin(th))
    y2 = y - u * c2 * (np.cos(th2) - np.cos(th))
    z2 = z + abs(delta_theta) * np.cos(ph) * np.sin(ph)

    return _cfg(x2, y2, z2, th2, ph)


def _pitch_full(cfg: Configuration, target_phi: float) -> Configuration:
    """
    Pure pitch change:
        u1 = 0, u2 = sign(target_phi - phi)
    Segment length = |target_phi - phi|.
    """
    delta_phi = target_phi - _phi(cfg)
    if abs(delta_phi) < EPS:
        return cfg

    x, y, z, th, ph = _x(cfg), _y(cfg), _z(cfg), _theta(cfg), _phi(cfg)
    u = np.sign(delta_phi)
    ph2 = target_phi

    x2 = x + np.cos(th) * u * (np.sin(ph2) - np.sin(ph))
    y2 = y + np.sin(th) * u * (np.sin(ph2) - np.sin(ph))
    z2 = z + u * (np.cos(ph) - np.cos(ph2))

    return _cfg(x2, y2, z2, th, ph2)


def _straight_full(cfg: Configuration, length: float) -> Configuration:
    if length < EPS:
        return cfg

    x, y, z, th, ph = _x(cfg), _y(cfg), _z(cfg), _theta(cfg), _phi(cfg)

    x2 = x + length * np.cos(ph) * np.cos(th)
    y2 = y + length * np.cos(ph) * np.sin(th)
    z2 = z + length * np.sin(ph)

    return _cfg(x2, y2, z2, th, ph)


def _turn_partial(cfg: Configuration, delta_theta: float, t: float) -> Configuration:
    if abs(delta_theta) < EPS or t <= 0:
        return cfg

    x, y, z, th, ph = _x(cfg), _y(cfg), _z(cfg), _theta(cfg), _phi(cfg)
    total = abs(delta_theta) * np.cos(ph)
    t = min(max(t, 0.0), total)
    u = np.sign(delta_theta)

    dth = u * t / np.cos(ph)
    th2 = th + dth
    c2 = np.cos(ph) ** 2

    x2 = x + u * c2 * (np.sin(th2) - np.sin(th))
    y2 = y - u * c2 * (np.cos(th2) - np.cos(th))
    z2 = z + t * np.sin(ph)

    return _cfg(x2, y2, z2, th2, ph)


def _pitch_partial(cfg: Configuration, target_phi: float, t: float) -> Configuration:
    delta_phi = target_phi - _phi(cfg)
    if abs(delta_phi) < EPS or t <= 0:
        return cfg

    x, y, z, th, ph = _x(cfg), _y(cfg), _z(cfg), _theta(cfg), _phi(cfg)
    total = abs(delta_phi)
    t = min(max(t, 0.0), total)
    u = np.sign(delta_phi)
    ph2 = ph + u * t

    x2 = x + np.cos(th) * u * (np.sin(ph2) - np.sin(ph))
    y2 = y + np.sin(th) * u * (np.sin(ph2) - np.sin(ph))
    z2 = z + u * (np.cos(ph) - np.cos(ph2))

    return _cfg(x2, y2, z2, th, ph2)


def _straight_partial(cfg: Configuration, length: float, t: float) -> Configuration:
    if length < EPS or t <= 0:
        return cfg

    t = min(max(t, 0.0), length)
    x, y, z, th, ph = _x(cfg), _y(cfg), _z(cfg), _theta(cfg), _phi(cfg)

    x2 = x + t * np.cos(ph) * np.cos(th)
    y2 = y + t * np.cos(ph) * np.sin(th)
    z2 = z + t * np.sin(ph)

    return _cfg(x2, y2, z2, th, ph)


# ============================================================
# Segment list -> curve(s)
# ============================================================

def _segment_length(kind: str, value: float, cfg: Configuration) -> float:
    if kind == "turn":
        return abs(value) * np.cos(_phi(cfg))
    if kind == "pitch_to":
        return abs(value - _phi(cfg))
    if kind == "straight":
        return float(value)
    raise ValueError(f"Unknown segment kind: {kind}")


def _apply_segment(cfg: Configuration, kind: str, value: float) -> Configuration:
    if kind == "turn":
        return _turn_full(cfg, value)
    if kind == "pitch_to":
        return _pitch_full(cfg, value)
    if kind == "straight":
        return _straight_full(cfg, value)
    raise ValueError(f"Unknown segment kind: {kind}")


def _apply_segment_partial(cfg: Configuration, kind: str, value: float, t: float) -> Configuration:
    if kind == "turn":
        return _turn_partial(cfg, value, t)
    if kind == "pitch_to":
        return _pitch_partial(cfg, value, t)
    if kind == "straight":
        return _straight_partial(cfg, value, t)
    raise ValueError(f"Unknown segment kind: {kind}")


def _make_curve(start: Configuration, segments):
    states = [start]
    lengths = []

    for kind, val in segments:
        cur = states[-1]
        lengths.append(_segment_length(kind, val, cur))
        states.append(_apply_segment(cur, kind, val))

    cum = np.cumsum([0.0] + lengths)
    total = float(cum[-1])

    def curve(s: np.ndarray) -> Configuration:
        arr = np.asarray(s, dtype=float)
        flat = arr.reshape(-1)

        xs = np.empty_like(flat)
        ys = np.empty_like(flat)
        zs = np.empty_like(flat)
        ths = np.empty_like(flat)
        phs = np.empty_like(flat)

        if total < EPS:
            xs.fill(_x(start))
            ys.fill(_y(start))
            zs.fill(_z(start))
            ths.fill(_theta(start))
            phs.fill(_phi(start))
        else:
            for i, si in enumerate(flat):
                l = np.clip(si, 0.0, 1.0) * total

                if l <= 0.0:
                    cfg = start
                elif l >= total:
                    cfg = states[-1]
                else:
                    idx = int(np.searchsorted(cum, l, side="right") - 1)
                    local = l - cum[idx]
                    cfg = _apply_segment_partial(states[idx], segments[idx][0], segments[idx][1], local)

                xs[i] = _x(cfg)
                ys[i] = _y(cfg)
                zs[i] = _z(cfg)
                ths[i] = _theta(cfg)
                phs[i] = _phi(cfg)

        if flat.size == 1:
            return _cfg(xs[0], ys[0], zs[0], ths[0], phs[0])

        shp = arr.shape
        return Configuration(
            xs.reshape(shp),
            ys.reshape(shp),
            zs.reshape(shp),
            ths.reshape(shp),
            phs.reshape(shp),
        )

    curve.total_length = total
    curve._segments = list(segments)
    return curve


def _curve_end(curve) -> Configuration:
    return curve(np.array([1.0]))


# ============================================================
# Template machinery
# ============================================================

GATE_TEMPLATES = [
    ["turn", "pitch", "straight", "pitch", "turn"],
    ["pitch", "turn", "straight", "turn", "pitch"],
    ["turn", "straight", "turn", "pitch"],
    ["pitch", "straight", "pitch", "turn"],
    ["turn", "pitch", "turn", "straight", "turn"],
]

SNITCH_TEMPLATES = [
    ["turn", "pitch", "straight"],
    ["pitch", "turn", "straight"],
    ["turn", "straight"],
    ["pitch", "straight"],
]


def _pattern_bounds(pattern):
    lows = []
    highs = []
    for step in pattern:
        if step == "turn":
            lows.append(-np.pi)
            highs.append(np.pi)
        elif step == "pitch":
            lows.append(PHI_MIN)
            highs.append(PHI_MAX)
        elif step == "straight":
            lows.append(0.0)
            highs.append(np.inf)
        else:
            raise ValueError(step)
    return np.array(lows, dtype=float), np.array(highs, dtype=float)


def _pattern_to_segments(pattern, params):
    segs = []
    for step, val in zip(pattern, params):
        if step == "turn":
            segs.append(("turn", float(val)))
        elif step == "pitch":
            segs.append(("pitch_to", float(val)))
        elif step == "straight":
            segs.append(("straight", float(val)))
        else:
            raise ValueError(step)
    return segs


def _template_length(start: Configuration, pattern, params) -> float:
    total = 0.0
    cur = start
    for step, val in zip(pattern, params):
        if step == "turn":
            total += abs(val) * np.cos(_phi(cur))
            cur = _turn_full(cur, float(val))
        elif step == "pitch":
            total += abs(float(val) - _phi(cur))
            cur = _pitch_full(cur, float(val))
        elif step == "straight":
            total += float(val)
            cur = _straight_full(cur, float(val))
    return float(total)


def _initial_guesses_for_pattern(start: Configuration, goal_pos: np.ndarray, goal_theta=None, goal_phi=None, pattern=None):
    dx, dy, dz = goal_pos - start.position()
    th_dir, ph_dir = _vec_to_angles(dx, dy, dz)
    dist = max(float(np.linalg.norm([dx, dy, dz])), 1e-3)

    th0 = _theta(start)
    ph0 = _phi(start)
    a0 = _wrap_angle(th_dir - th0)
    goal_theta = th_dir if goal_theta is None else goal_theta
    goal_phi = ph_dir if goal_phi is None else goal_phi
    goal_phi = float(np.clip(goal_phi, PHI_MIN, PHI_MAX))

    avg_phi = float(np.clip(0.5 * (ph0 + ph_dir), PHI_MIN, PHI_MAX))
    quarter_phi = float(np.clip(0.5 * (ph0 + goal_phi), PHI_MIN, PHI_MAX))
    a_goal = _wrap_angle(goal_theta - th_dir)
    a_goal2 = _wrap_angle(goal_theta - th0)

    pool = {
        "turn": [a0, 0.0, a_goal, a_goal2, _wrap_angle(a0 + np.pi), _wrap_angle(a0 - np.pi), a0 * 0.5],
        "pitch": [avg_phi, goal_phi, ph0, quarter_phi, float(np.clip(ph_dir, PHI_MIN, PHI_MAX))],
        "straight": [dist, 0.7 * dist, 1.3 * dist, max(dist - 1.0, 0.1), dist + 1.0],
    }

    guesses = []

    def backtrack(i, cur):
        if i == len(pattern):
            guesses.append(np.array(cur, dtype=float))
            return
        for v in pool[pattern[i]]:
            cur.append(v)
            backtrack(i + 1, cur)
            cur.pop()

    # small hand-crafted subset, not full cartesian explosion
    if pattern == ["turn", "pitch", "straight", "pitch", "turn"]:
        guesses = [
            np.array([a0, avg_phi, dist, goal_phi, a_goal], dtype=float),
            np.array([a0, avg_phi, 0.7 * dist, goal_phi, a_goal], dtype=float),
            np.array([0.0, ph0, dist, goal_phi, a_goal2], dtype=float),
            np.array([a0, goal_phi, dist, goal_phi, a_goal], dtype=float),
        ]
    elif pattern == ["pitch", "turn", "straight", "turn", "pitch"]:
        guesses = [
            np.array([avg_phi, a0, dist, a_goal, goal_phi], dtype=float),
            np.array([goal_phi, a0, dist, a_goal, goal_phi], dtype=float),
            np.array([ph0, 0.0, dist, a_goal2, goal_phi], dtype=float),
        ]
    elif pattern == ["turn", "straight", "turn", "pitch"]:
        guesses = [
            np.array([a0, dist, a_goal, goal_phi], dtype=float),
            np.array([0.5 * a0, dist, _wrap_angle(goal_theta - (th0 + 0.5 * a0)), goal_phi], dtype=float),
        ]
    elif pattern == ["pitch", "straight", "pitch", "turn"]:
        guesses = [
            np.array([avg_phi, dist, goal_phi, a_goal2], dtype=float),
            np.array([goal_phi, dist, goal_phi, a_goal2], dtype=float),
        ]
    elif pattern == ["turn", "pitch", "turn", "straight", "turn"]:
        guesses = [
            np.array([0.5 * a0, avg_phi, 0.5 * a0, dist, a_goal], dtype=float),
            np.array([a0, avg_phi, 0.0, dist, a_goal], dtype=float),
        ]
    elif pattern == ["turn", "pitch", "straight"]:
        guesses = [
            np.array([a0, float(np.clip(ph_dir, PHI_MIN, PHI_MAX)), dist], dtype=float),
            np.array([0.0, ph0, dist], dtype=float),
            np.array([a0, avg_phi, dist], dtype=float),
        ]
    elif pattern == ["pitch", "turn", "straight"]:
        guesses = [
            np.array([avg_phi, a0, dist], dtype=float),
            np.array([float(np.clip(ph_dir, PHI_MIN, PHI_MAX)), a0, dist], dtype=float),
        ]
    elif pattern == ["turn", "straight"]:
        guesses = [
            np.array([a0, dist], dtype=float),
            np.array([0.0, dist], dtype=float),
        ]
    elif pattern == ["pitch", "straight"]:
        guesses = [
            np.array([float(np.clip(ph_dir, PHI_MIN, PHI_MAX)), dist], dtype=float),
            np.array([ph0, dist], dtype=float),
        ]
    else:
        backtrack(0, [])

    return guesses


def _solve_pattern(start: Configuration, pattern, target_pos: np.ndarray, target_theta=None, target_phi=None, max_nfev=1500):
    lo, hi = _pattern_bounds(pattern)
    seeds = _initial_guesses_for_pattern(start, target_pos, target_theta, target_phi, pattern)

    def residual(p):
        segs = _pattern_to_segments(pattern, p)
        end = _curve_end(_make_curve(start, segs))
        res = [
            end.x - target_pos[0],
            end.y - target_pos[1],
            end.z - target_pos[2],
        ]
        if target_theta is not None:
            res.append(_wrap_angle(end.theta - target_theta))
        if target_phi is not None:
            res.append(end.phi - target_phi)
        res.append(np.sqrt(LEN_REG) * _template_length(start, pattern, p))
        return np.array(res, dtype=float)

    best = None
    best_score = None

    for seed in seeds:
        x0 = seed.copy()
        for i in range(len(x0)):
            if np.isfinite(lo[i]):
                x0[i] = max(x0[i], lo[i] + 1e-9)
            if np.isfinite(hi[i]):
                x0[i] = min(x0[i], hi[i] - 1e-9)
        try:
            res = least_squares(residual, x0, bounds=(lo, hi), max_nfev=max_nfev)
        except Exception:
            continue
        score = float(np.linalg.norm(res.fun[:-1]))
        if best is None or score < best_score:
            best = res
            best_score = score
        if score < 1e-9:
            break

    if best is None:
        raise RuntimeError("pattern solve failed")

    return _pattern_to_segments(pattern, best.x)


# ============================================================
# Candidate generators
# ============================================================

def _candidate_snitch_curves(start: Configuration, goal_xyz: XYZConfiguration):
    candidates = []

    if _aligned_with_segment(start, goal_xyz.position()):
        L = float(np.linalg.norm(goal_xyz.position() - start.position()))
        curve = _make_curve(start, [("straight", L)])
        candidates.append((curve, _pos_err(_curve_end(curve), goal_xyz.position()), curve.total_length))
        return candidates

    for pattern in SNITCH_TEMPLATES:
        try:
            segs = _solve_pattern(start, pattern, goal_xyz.position(), None, None, max_nfev=1200)
            curve = _make_curve(start, segs)
            err = _pos_err(_curve_end(curve), goal_xyz.position())
            candidates.append((curve, err, curve.total_length))
        except Exception:
            continue

    candidates.sort(key=lambda t: (t[1], t[2]))
    return candidates


def _candidate_gate_curves(start: Configuration, goal: Configuration):
    candidates = []

    if (
        _aligned_with_segment(start, goal.position())
        and abs(_wrap_angle(goal.theta - start.theta)) < 1e-9
        and abs(goal.phi - start.phi) < 1e-9
    ):
        L = float(np.linalg.norm(goal.position() - start.position()))
        curve = _make_curve(start, [("straight", L)])
        candidates.append((curve, _cfg_error(_curve_end(curve), goal), curve.total_length))
        return candidates

    # direct pattern bank
    for pattern in GATE_TEMPLATES:
        try:
            segs = _solve_pattern(
                start,
                pattern,
                goal.position(),
                goal.theta,
                goal.phi,
                max_nfev=2000,
            )
            curve = _make_curve(start, segs)
            err = _cfg_error(_curve_end(curve), goal)
            candidates.append((curve, err, curve.total_length))
        except Exception:
            continue

    # pre-goal waypoint bank
    gdir = _goal_dir(goal)
    gside = _goal_side(goal)

    for d in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for side_shift in [0.0, -0.75, 0.75]:
            pre = goal.position() - d * gdir + side_shift * gside
            pre_xyz = XYZConfiguration(float(pre[0]), float(pre[1]), float(pre[2]))

            snitch_candidates = _candidate_snitch_curves(start, pre_xyz)[:3]
            for curve1, _, _ in snitch_candidates:
                mid = _curve_end(curve1)
                for pattern in GATE_TEMPLATES:
                    try:
                        segs2 = _solve_pattern(
                            mid,
                            pattern,
                            goal.position(),
                            goal.theta,
                            goal.phi,
                            max_nfev=1500,
                        )
                        segs = list(curve1._segments) + list(segs2)
                        curve = _make_curve(start, segs)
                        err = _cfg_error(_curve_end(curve), goal)
                        candidates.append((curve, err, curve.total_length))
                    except Exception:
                        continue

    candidates.sort(key=lambda t: (t[1], t[2]))
    return candidates


def _select_best(candidates, pos_tol=0.02, ang_tol=0.05):
    feasible = []
    for curve, _, length in candidates:
        end = _curve_end(curve)
        if hasattr(end, "position"):
            pass
        feasible.append((curve, length))

    # first try strict feasibility by endpoint only
    strict = []
    for curve, _, length in candidates:
        end = _curve_end(curve)
        strict.append((curve, length, end))
    return candidates[0][0]


def _pos_err(c: Configuration, goal_pos: np.ndarray) -> float:
    return float(np.linalg.norm(c.position() - goal_pos))


# ============================================================
# Public API
# ============================================================

def gate_pass(
    start: Configuration,
    goal: Configuration,
) -> Callable[[np.ndarray], Configuration]:
    candidates = _candidate_gate_curves(start, goal)
    if not candidates:
        raise RuntimeError("Failed to construct gate_pass trajectory")

    # Prefer candidates that already satisfy the endpoint tolerances.
    feasible = []
    for curve, _, length in candidates:
        end = _curve_end(curve)
        pos_err = np.linalg.norm(end.position() - goal.position())
        ang_err = _angle_error(end, goal)
        if pos_err <= 0.02 and ang_err <= 0.05:
            feasible.append((curve, length))

    if feasible:
        feasible.sort(key=lambda t: t[1])
        return feasible[0][0]

    return candidates[0][0]


def catch_snitch(
    start: Configuration,
    goal_xyz: XYZConfiguration,
) -> Callable[[np.ndarray], Configuration]:
    candidates = _candidate_snitch_curves(start, goal_xyz)
    if not candidates:
        raise RuntimeError("Failed to construct catch_snitch trajectory")

    feasible = []
    for curve, _, length in candidates:
        end = _curve_end(curve)
        pos_err = np.linalg.norm(end.position() - goal_xyz.position())
        if pos_err <= 0.02:
            feasible.append((curve, length))

    if feasible:
        feasible.sort(key=lambda t: t[1])
        return feasible[0][0]

    return candidates[0][0]


def catch_ball_and_gate(
    start: Configuration,
    intermediate_goal_xyz: XYZConfiguration,
    final_goal: Configuration,
) -> Callable[[np.ndarray], Configuration]:
    first = catch_snitch(start, intermediate_goal_xyz)
    mid = _curve_end(first)
    second = gate_pass(mid, final_goal)
    return _make_curve(start, list(first._segments) + list(second._segments))