import numpy as np
from lib.beads import JOINT_LIMIT_RADIUS, bead_configuration_violations, bounding_sphere_radius, build_necklace


def _project_joint_to_cap(ax, ay, margin=1e-6):
    limit_cos = np.cos(JOINT_LIMIT_RADIUS + 1e-5)
    lim = JOINT_LIMIT_RADIUS * 0.999
    ax = float(np.clip(ax, -lim, lim)); ay = float(np.clip(ay, -lim, lim))
    if np.cos(ax)*np.cos(ay) >= limit_cos:
        return ax, ay
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo+hi)/2
        tx, ty = mid*ax, mid*ay
        if np.cos(tx)*np.cos(ty) >= limit_cos:
            lo = mid
        else:
            hi = mid
    return lo*ax, lo*ay


def _project_angles(a):
    a = np.asarray(a, dtype=np.float64).copy()
    for i in range(len(a)):
        a[i,0], a[i,1] = _project_joint_to_cap(a[i,0], a[i,1])
    return a


def _valid(link_lengths, angles):
    return len(bead_configuration_violations(link_lengths, angles)) == 0


def _score(link_lengths, angles):
    if not _valid(link_lengths, angles):
        return None
    return bounding_sphere_radius(link_lengths, angles)


def _heuristic_seed(link_lengths):
    n_joints = len(link_lengths)-1
    grid = np.linspace(-1.0, 1.0, 9)
    seeds = []
    for ax in grid:
        for ay in grid:
            seeds += [np.tile([ax, ay], (n_joints,1)),
                      np.array([[ax, ay if i%2==0 else -ay] for i in range(n_joints)], float),
                      np.array([[ax if i%2==0 else -ax, ay] for i in range(n_joints)], float),
                      np.array([[ax if i%2==0 else -ax, ay if i%2==0 else -ay] for i in range(n_joints)], float)]
    seeds += [np.tile([0.95,-0.38], (n_joints,1)), np.tile([1.00,-0.35], (n_joints,1)), np.tile([-0.35,1.00], (n_joints,1)), np.zeros((n_joints,2))]
    best = None; best_r = np.inf
    for s in seeds:
        s = _project_angles(s)
        r = _score(link_lengths, s)
        if r is not None and r < best_r:
            best_r = r; best = s.copy()
    return best


def _positions_flat(link_lengths, angles):
    return build_necklace(link_lengths, angles).reshape(-1)


def _num_jac(link_lengths, angles, eps=2e-4):
    x0 = _positions_flat(link_lengths, angles)
    nvar = angles.size
    J = np.zeros((x0.size, nvar))
    base = angles.reshape(-1)
    for k in range(nvar):
        p = base.copy(); p[k]+=eps
        m = base.copy(); m[k]-=eps
        fp = _positions_flat(link_lengths, _project_angles(p.reshape(angles.shape)))
        fm = _positions_flat(link_lengths, _project_angles(m.reshape(angles.shape)))
        J[:,k] = (fp-fm)/(2*eps)
    return J


def _damped_step(link_lengths, angles, target_flat, damping):
    x = _positions_flat(link_lengths, angles)
    e = target_flat - x
    J = _num_jac(link_lengths, angles)
    A = J.T@J + damping*np.eye(J.shape[1])
    b = J.T@e
    return np.linalg.solve(A,b).reshape(angles.shape)


def _merit(link_lengths, angles, target_flat, mu=0.05):
    x = _positions_flat(link_lengths, angles)
    return np.sum((target_flat-x)**2) + mu*bounding_sphere_radius(link_lengths, angles)**2


def optimal_bead_config(link_lengths: np.ndarray) -> np.ndarray:
    link_lengths = np.asarray(link_lengths, dtype=np.float64)
    n_joints = len(link_lengths)-1
    if n_joints <= 0:
        return np.zeros((0,2))
    angles = _heuristic_seed(link_lengths)
    best = angles.copy(); best_r = bounding_sphere_radius(link_lengths, angles)

    # Use lecture-style projected damped Newton-Raphson on stacked bead positions.
    for shrink in [0.98,0.95,0.92,0.89,0.86,0.83,0.80,0.77,0.74]:
        improved = True
        while improved:
            improved = False
            pts = build_necklace(link_lengths, angles)
            c = pts.mean(axis=0, keepdims=True)
            target = c + shrink*(pts-c)
            # Add a light z-compression bias after centering to encourage ball-like packing.
            target[:,2] = c[0,2] + 0.90*(target[:,2]-c[0,2])
            target_flat = target.reshape(-1)
            base_mer = _merit(link_lengths, angles, target_flat)
            base_r = bounding_sphere_radius(link_lengths, angles)
            for damp in [10.0,3.0,1.0,0.3,0.1]:
                step = _damped_step(link_lengths, angles, target_flat, damp)
                norm = np.linalg.norm(step)
                if norm > 0.5*np.sqrt(step.size):
                    step *= (0.5*np.sqrt(step.size)/norm)
                accepted=False
                for alpha in [1.0,0.5,0.25,0.125,0.0625]:
                    cand = _project_angles(angles + alpha*step)
                    if not _valid(link_lengths, cand):
                        continue
                    cand_r = bounding_sphere_radius(link_lengths, cand)
                    cand_mer = _merit(link_lengths, cand, target_flat)
                    if cand_mer < base_mer - 1e-8 or cand_r < base_r - 1e-8:
                        angles = cand
                        improved = True; accepted=True
                        if cand_r < best_r:
                            best = cand.copy(); best_r = cand_r
                        break
                if accepted:
                    break
    return best
