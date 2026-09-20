"""Route-Guided CBF-QP Repair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple

import cvxpy as cp
import numpy as np

Phase = Literal["normal", "entry", "exit", "rejoin"]
Side = Literal["plus", "minus"]



@dataclass
class BypassGateConfig:
    # blocked iff min_k φ < 1+ε. ε 값
    bypass_eps: float = 0.05

    # 식 (10): ||p_t - w_active|| < δ
    bypass_reach_radius: float = 0.04

    # 식 (6): λclr=1.0, λprog=1.0, λlen=0.3, λdev=0.5.
    bypass_lambda_clear: float = 1.0
    bypass_lambda_prog: float = 1.0
    bypass_lambda_len: float = 0.3
    bypass_lambda_dev: float = 0.5

    # 식 (8)–(9)의 route gain. 아래 값은 실험 기본값.
    bypass_kp: float = 2.0
    bypass_wp_weight: float = 4.0  # 논문 λ_r

    # entry/exit를 만들려면 필요해서 유지.
    bypass_waypoint_margin: float = 0.03  # m
    bypass_gate_alpha: float = 0.5


@dataclass
class BypassGateSession:
    #  entry → exit → rejoin → (β_t=0) 표준 CBF-QP.
    phase: Phase = "normal"
    side: Optional[Side] = None
    w_entry: Optional[np.ndarray] = None
    w_exit: Optional[np.ndarray] = None
    p_rejoin: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.phase = "normal"
        self.side = None
        self.w_entry = None
        self.w_exit = None
        self.p_rejoin = None


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _as_vec3(x: np.ndarray) -> np.ndarray:
    v = np.asarray(x, dtype=float).reshape(-1)[:3]
    return v.copy()


def _safe_normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return fallback.copy()
    return v / n


def ellipsoid_qinv(R2: np.ndarray, Q2_diag: np.ndarray) -> np.ndarray:
    """World-frame Q^{-1} from oriented semi-axes (AEGIS fit_ellipse output)."""
    axes = np.asarray(Q2_diag, dtype=float).reshape(-1)[:3]
    axes = np.maximum(axes, 1e-6)
    inv_diag = 1.0 / np.square(axes)
    return R2 @ np.diag(inv_diag) @ R2.T


def phi_point(p: np.ndarray, c: np.ndarray, qinv: np.ndarray) -> float:
    d = _as_vec3(p) - _as_vec3(c)
    return float(d @ qinv @ d)


def rho_direction(d: np.ndarray, qinv: np.ndarray) -> float:
    """Ellipsoid center → boundary distance along unit d.
    ρ(d) = 1 / sqrt(d^T Q^{-1} d) 로 경계를 잡는다.
    """
    u = _safe_normalize(_as_vec3(d), np.array([1.0, 0.0, 0.0]))
    q = float(u @ qinv @ u)
    return 1.0 / np.sqrt(max(q, 1e-12))


def rot90_xy(t: np.ndarray) -> np.ndarray:
    """t의 table-plane (z=0) 좌측 단위 법선.
    논문은 s ∈ {left, right}만 적음. 좌우 축을 어떻게 만드는지는 없음.
    """
    txy = np.array([t[0], t[1], 0.0], dtype=float)
    if np.linalg.norm(txy) < 1e-8:
        txy = np.array([1.0, 0.0, 0.0])
    s = np.array([-txy[1], txy[0], 0.0])
    return _safe_normalize(s, np.array([0.0, 1.0, 0.0]))


def accumulate_nominal_path(ee_pos: np.ndarray, action_chunk: np.ndarray) -> np.ndarray:
    """VLA short-horizon path: EEF + cumulative chunk translations.
    p_hat_{t+k} = p_ee + sum_{j<k} Delta p^{vla}_{t+j}
    """
    p = _as_vec3(ee_pos)
    chunk = np.asarray(action_chunk, dtype=float)
    if chunk.ndim == 1:
        chunk = chunk[None, :]
    points = [p]
    acc = p.copy()
    for row in chunk:
        acc = acc + np.asarray(row[:3], dtype=float)
        points.append(acc.copy())
    return np.stack(points, axis=0)


def detect_path_blocked(
    path: np.ndarray,
    c: np.ndarray,
    qinv: np.ndarray,
    eps: float,
) -> Tuple[bool, int, float]:
    """식 (4): min_k φ(p̂_{t+k}) < 1+ε 이면 blocked.

    inflated ellipsoid라고 적음. 이 함수는 Q를 키우지 않고 φ < 1+ε 근접만 본다.
    """
    phis = np.array([phi_point(p, c, qinv) for p in path[1:]], dtype=float)
    if phis.size == 0:
        return False, -1, np.inf
    k = int(np.argmin(phis))
    best = float(phis[k])
    return best < 1.0 + float(eps), k, best


def select_rejoin_point(
    path: np.ndarray,
    ee_pos: np.ndarray,
    c: np.ndarray,
    qinv: np.ndarray,
    eps_safe: float,
) -> Optional[np.ndarray]:
    """rejoin waypoint w_rejoin.
    구현: 마지막 blocked 점 이후, φ > 1+ε 이고 g^T (p - p_ee) > 0.
    """
    p_ee = _as_vec3(ee_pos)
    fut = path[1:]
    if fut.shape[0] == 0:
        return None
    g = _safe_normalize(fut[-1] - p_ee, np.array([1.0, 0.0, 0.0]))
    phis = np.array([phi_point(p, c, qinv) for p in fut])
    blocked = phis < 1.0 + float(eps_safe)
    start = int(np.max(np.where(blocked)[0]) + 1) if np.any(blocked) else 0

    for p in fut[start:]:
        if phi_point(p, c, qinv) > 1.0 + float(eps_safe) and float(np.dot(g, p - p_ee)) > 0.0:
            return _as_vec3(p)
    for p in fut[::-1]:
        if phi_point(p, c, qinv) > 1.0 + float(eps_safe) and float(np.dot(g, p - p_ee)) > 0.0:
            return _as_vec3(p)
    return None


def build_gate_waypoints(
    c: np.ndarray,
    t: np.ndarray,
    s: np.ndarray,
    qinv: np.ndarray,
    alpha: float,
    margin: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """left/right entry·exit. 논문 식 (5)는 이름만 있고 좌표 식이 없음.
        w_entry = c - α ρ(t) t + (ρ(s) + m) s
        w_exit  = c + α ρ(t) t + (ρ(s) + m) s
    """
    t_u = _safe_normalize(t, np.array([1.0, 0.0, 0.0]))
    s_u = _safe_normalize(s, np.array([0.0, 1.0, 0.0]))
    rho_t = rho_direction(t_u, qinv)
    rho_s = rho_direction(s_u, qinv)
    c = _as_vec3(c)
    w_entry = c - alpha * rho_t * t_u + (rho_s + margin) * s_u
    w_exit = c + alpha * rho_t * t_u + (rho_s + margin) * s_u
    return w_entry, w_exit


def _nearest_path_dist(p: np.ndarray, path: np.ndarray) -> float:
    d = path - _as_vec3(p)[None, :]
    return float(np.min(np.linalg.norm(d, axis=1)))


def score_side(
    p_ee: np.ndarray,
    w_entry: np.ndarray,
    w_exit: np.ndarray,
    p_rejoin: np.ndarray,
    path: np.ndarray,
    c: np.ndarray,
    qinv: np.ndarray,
    cfg: BypassGateConfig,
) -> float:
    """식 (6): J = λclr C + λprog P − λlen L − λdev D."""
    p_ee = _as_vec3(p_ee)
    samples = [w_entry, w_exit, 0.5 * (w_entry + w_exit)]
    C = min(phi_point(p, c, qinv) for p in samples)
    t = _safe_normalize(p_rejoin - p_ee, np.array([1.0, 0.0, 0.0]))
    P = float(np.dot(w_exit - p_ee, t))
    L = (
        float(np.linalg.norm(w_entry - p_ee))
        + float(np.linalg.norm(w_exit - w_entry))
        + float(np.linalg.norm(p_rejoin - w_exit))
    )
    D = 0.5 * (_nearest_path_dist(w_entry, path) + _nearest_path_dist(w_exit, path))

    return (
        cfg.bypass_lambda_clear * C
        + cfg.bypass_lambda_prog * P
        - cfg.bypass_lambda_len * L
        - cfg.bypass_lambda_dev * D
    )


def plan_bypass_gate(
    p_ee: np.ndarray,
    path: np.ndarray,
    c: np.ndarray,
    qinv: np.ndarray,
    cfg: BypassGateConfig,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Side]]:
    p_rejoin = select_rejoin_point(path, p_ee, c, qinv, cfg.bypass_eps)
    if p_rejoin is None:
        return None

    t = _safe_normalize(p_rejoin - _as_vec3(p_ee), np.array([1.0, 0.0, 0.0]))
    s_plus = rot90_xy(t)
    s_minus = -s_plus

    best: Optional[Tuple[float, np.ndarray, np.ndarray, Side]] = None
    for side, s in (("plus", s_plus), ("minus", s_minus)):
        w_entry, w_exit = build_gate_waypoints(
            c, t, s, qinv, cfg.bypass_gate_alpha, cfg.bypass_waypoint_margin
        )
        j = score_side(p_ee, w_entry, w_exit, p_rejoin, path, c, qinv, cfg)
        if best is None or j > best[0]:
            best = (j, w_entry, w_exit, side)  # type: ignore[assignment]

    assert best is not None
    _, w_entry, w_exit, side = best
    return w_entry, w_exit, p_rejoin, side


def active_waypoint(session: BypassGateSession) -> Optional[np.ndarray]:
    if session.phase == "entry":
        return session.w_entry
    if session.phase == "exit":
        return session.w_exit
    if session.phase == "rejoin":
        return session.p_rejoin
    return None


def step_bypass_session(
    session: BypassGateSession,
    p_ee: np.ndarray,
    path: np.ndarray,
    c: np.ndarray,
    qinv: np.ndarray,
    cfg: BypassGateConfig,
) -> BypassGateSession:
    """blocked면 경로 생성, ||p-w||<δ 이면 다음 waypoint, rejoin 후 β_t=0."""
    p_ee = _as_vec3(p_ee)


    blocked, _, _ = detect_path_blocked(path, c, qinv, cfg.bypass_eps)

    if session.phase == "normal":
        # if session.blocked_streak >= cfg.bypass_block_steps:
        if blocked:
            planned = plan_bypass_gate(p_ee, path, c, qinv, cfg)
            if planned is not None:
                w_entry, w_exit, p_rejoin, side = planned
                session.w_entry = w_entry
                session.w_exit = w_exit
                session.p_rejoin = p_rejoin
                session.side = side
                session.phase = "entry"
                # session.dwell_left = cfg.bypass_min_dwell
        return session

    wp = active_waypoint(session)
    if wp is None:
        session.phase = "normal"
        return session

    # 식 (10): 도달 반경 δ만 사용.
    reached = float(np.linalg.norm(p_ee - wp)) < cfg.bypass_reach_radius
    if reached:
        if session.phase == "entry":
            session.phase = "exit"

        elif session.phase == "exit":
            session.phase = "rejoin"

        elif session.phase == "rejoin":

            session.phase = "normal"
            session.w_entry = session.w_exit = session.p_rejoin = None
            session.side = None

    return session


# ---------------------------------------------------------------------------
# CBF-QP: main_aegis.py 그대로 + soft waypoint
# ---------------------------------------------------------------------------

# main_aegis.py: u_v_ref = 5 * v_ref, action[:3] = 0.2 * R1 @ u_v, constraint + 10 * h
_AEGIS_VEL_SCALE = 5.0
_AEGIS_ACTION_SCALE = 0.2
_AEGIS_CBF_GAIN = 10.0
_AEGIS_W = np.diag([1.0 / 25.0] * 6 + [1.0, 1.0, 1.0])


def solve_aegis_cbf_qp(
    action: np.ndarray,
    R1: np.ndarray,
    p1: np.ndarray,
    Q1_diag: np.ndarray,
    p2: np.ndarray,
    Q2_diag: np.ndarray,
    R2: np.ndarray,
    z_fixed: np.ndarray,
    dt: float,
    compute_h_coeffs_3d: Callable,
    w_active: Optional[np.ndarray] = None,
    ee_pos: Optional[np.ndarray] = None,
    kp: float = 0.0,
    wp_weight: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """`main_aegis.py` 안전층 QP.

    기본: min (u - u_ref)^T W (u - u_ref)
          s.t. a_u_v u_v + a_u_omega u_omega + a_uz u_z + 10 h >= 0
    u ∈ R^9 = [u_v(3), u_omega(3), u_z(3)]

    w_active가 있을 때만 논문 식 (8)–(9)의 soft term을 목적함수에 추가.
    CBF 제약·W·스케일은 AEGIS와 같게 둔다.
    """
    action = np.asarray(action, dtype=float).reshape(-1)

    # --- main_aegis.py 와 동일 ---
    v_ref = R1.T @ action[:3]
    u_v_ref = _AEGIS_VEL_SCALE * v_ref
    omega_ref = action[3:6]
    u_omega_ref = _AEGIS_VEL_SCALE * omega_ref

    a_v, a_omega, a_uz, h, mu_row = compute_h_coeffs_3d(
        p1, Q1_diag, R1, p2, Q2_diag, R2, z_fixed
    )
    a_u_v = _AEGIS_ACTION_SCALE * a_v
    a_u_omega = _AEGIS_ACTION_SCALE * a_omega
    u_z_nom = _AEGIS_CBF_GAIN * mu_row

    u = cp.Variable(9)
    u_ref_vec = np.hstack([u_v_ref, u_omega_ref, u_z_nom])
    cost = cp.quad_form(u - u_ref_vec, _AEGIS_W)

    # --- Route-Guided 추가분. AEGIS 제약은 건드리지 않음. ---
    if w_active is not None and wp_weight > 0.0 and ee_pos is not None:
        u_route_world = kp * (_as_vec3(w_active) - _as_vec3(ee_pos))
        u_v_wp = _AEGIS_VEL_SCALE * (R1.T @ u_route_world)
        cost = cost + float(wp_weight) * cp.sum_squares(u[:3] - u_v_wp)

    constraints = [
        a_u_v @ u[:3] + a_u_omega @ u[3:6] + a_uz @ u[6:] + _AEGIS_CBF_GAIN * h >= 0
    ]
    prob = cp.Problem(cp.Minimize(cost), constraints)
    try:
        prob.solve(solver=cp.OSQP, warm_start=True)
    except Exception:
        u.value = None

    if u.value is not None:
        u_v = u.value[:3]
        u_omega = u.value[3:6]
        u_z = u.value[6:]
    else:
        # main_aegis는 infeasible 시 raw action을 넣음. 스케일을 맞추려면 ref가 맞다.
        u_v = u_v_ref
        u_omega = u_omega_ref
        u_z = u_z_nom

    ident = np.eye(len(z_fixed))
    dz = (ident - np.outer(z_fixed, z_fixed)) @ u_z
    z_next = z_fixed + dz * dt
    z_next = z_next / np.linalg.norm(z_next)

    action_out = np.zeros(7, dtype=float)
    action_out[:3] = _AEGIS_ACTION_SCALE * R1 @ u_v
    action_out[3:6] = _AEGIS_ACTION_SCALE * u_omega
    action_out[6] = action[6]
    return action_out, z_next


def repair_vla_action(
    action: np.ndarray,
    action_chunk: np.ndarray,
    ee_pos: np.ndarray,
    R1: np.ndarray,
    p1: np.ndarray,
    Q1_diag: np.ndarray,
    p2: np.ndarray,
    Q2_diag: np.ndarray,
    R2: np.ndarray,
    z_fixed: np.ndarray,
    dt: float,
    session: BypassGateSession,
    compute_h_coeffs_3d: Callable,
    config: Optional[BypassGateConfig] = None,
) -> Tuple[np.ndarray, BypassGateSession, np.ndarray]:
    """한 스텝: AEGIS CBF-QP가 기본. blocked일 때만 waypoint soft term."""
    cfg = config or BypassGateConfig()
    qinv = ellipsoid_qinv(R2, Q2_diag)
    path = accumulate_nominal_path(ee_pos, action_chunk)
    session = step_bypass_session(session, ee_pos, path, p2, qinv, cfg)
    wp = active_waypoint(session)
    action_out, z_next = solve_aegis_cbf_qp(
        action=action,
        R1=R1,
        p1=p1,
        Q1_diag=Q1_diag,
        p2=p2,
        Q2_diag=Q2_diag,
        R2=R2,
        z_fixed=z_fixed,
        dt=dt,
        compute_h_coeffs_3d=compute_h_coeffs_3d,
        w_active=wp,
        ee_pos=ee_pos,
        kp=cfg.bypass_kp,
        wp_weight=cfg.bypass_wp_weight,
    )
    return action_out, session, z_next
