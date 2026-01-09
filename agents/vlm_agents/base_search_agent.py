"""SearchAgentV1_5 - v1_4 + Plan (5) explicit "no-foul" candidate library

Implements plan/01_extra_plan.md step 5.

Motivation
----------
Even strong shot-selection agents sometimes give away turns under noise via:
- FOUL_FIRST_HIT (wrong first contact)
- NO_POCKET_NO_RAIL
- NO_HIT

We reduce variance by injecting a small library of explicit "no-foul" shots:
- aim directly at a legal target ball (maximize chance of first contact)
- use medium speed that is likely to send cue/object to a cushion

We then force-keep a handful of these candidates through prefilter so they are
always available as "insurance" when pots are low-confidence.

Env vars
--------
# v1_4 schedule defaults (champion)
BILLIARDS_MAX_SIMULATIONS=180
BILLIARDS_STAGE1_N=90
BILLIARDS_STAGE1_R=1
BILLIARDS_STAGE2_K=12
BILLIARDS_STAGE2_M=3

# no-foul library
BILLIARDS_NOFOUL_COUNT=10            # number to force-keep after prefilter
BILLIARDS_NOFOUL_V0S=2.2,3.0         # comma-separated speeds used
BILLIARDS_NOFOUL_ANGLE_OFFS=0,7,-7   # comma-separated offsets (deg) around cue->target line

Use:
  BILLIARDS_AGENT_BACKEND=search_v1_5 python evaluate.py
"""

import os
import math
import copy
import signal
import random
import numpy as np
from typing import List, Tuple, Optional, Set
import pooltool as pt

from .agent import Agent


# ============ Timeout Protection ============
class SimulationTimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise SimulationTimeoutError("Physics simulation timeout")


def simulate_with_timeout(shot, timeout=3):
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    try:
        pt.simulate(shot, inplace=True)
        signal.alarm(0)
        return True
    except SimulationTimeoutError:
        signal.alarm(0)
        return False
    except Exception:
        signal.alarm(0)
        return False
    finally:
        signal.signal(signal.SIGALRM, old_handler)


# ============ Reward (aligned with BasicAgentPro) ============
def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]

    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]

    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    first_contact_ball_id = None
    foul_first_hit = False
    valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}

    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break

    if first_contact_ball_id is None:
        if len(last_state) > 2 or player_targets != ['8']:
            foul_first_hit = True
    else:
        if first_contact_ball_id not in player_targets:
            foul_first_hit = True

    cue_hit_cushion = False
    target_hit_cushion = False
    foul_no_rail = False

    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if 'cushion' in et:
            if 'cue' in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True

    if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
        foul_no_rail = True

    score = 0
    if cue_pocketed and eight_pocketed:
        score -= 500
    elif cue_pocketed:
        score -= 100
    elif eight_pocketed:
        is_targeting_eight_ball_legally = (len(player_targets) == 1 and player_targets[0] == "8")
        score += 150 if is_targeting_eight_ball_legally else -500

    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30

    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20

    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10

    return score


# ============ Geometry utilities ============
def ball_pos(balls, ball_id: str) -> Optional[np.ndarray]:
    if ball_id not in balls:
        return None
    ball = balls[ball_id]
    if hasattr(ball, 'state') and ball.state.s == 4:
        return None
    return np.array([ball.state.rvw[0][0], ball.state.rvw[0][1]], dtype=float)


def wrap_angle(angle_deg: float) -> float:
    return float(angle_deg % 360)


def segment_distance_point(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def is_line_blocked(a: np.ndarray, b: np.ndarray, balls, ball_radius: float, ignore_ids: Set[str] = frozenset()) -> bool:
    if a is None or b is None:
        return True
    for bid in balls.keys():
        if bid in ignore_ids:
            continue
        pos = ball_pos(balls, bid)
        if pos is None:
            continue
        if segment_distance_point(pos, a, b) < (2.0 * ball_radius * 0.98):
            return True
    return False


def cut_angle_deg(cue_xy: np.ndarray, obj_xy: np.ndarray, pocket_xy: np.ndarray) -> float:
    v1 = obj_xy - cue_xy
    v2 = pocket_xy - obj_xy
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    u1 = v1 / n1
    u2 = v2 / n2
    dot = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def nearest_pocket_distance(pt_xy: np.ndarray, pocket_centers: List[np.ndarray]) -> float:
    if pt_xy is None or not pocket_centers:
        return float('inf')
    return float(min(np.linalg.norm(pt_xy - pc) for pc in pocket_centers))


def ray_to_pocket_risk(cue_xy: np.ndarray, phi_deg: float, pocket_centers: List[np.ndarray]) -> float:
    if cue_xy is None or not pocket_centers:
        return 0.0
    u = np.array([math.cos(math.radians(phi_deg)), math.sin(math.radians(phi_deg))], dtype=float)
    threshold = float(os.environ.get('BILLIARDS_SCRATCH_RAY_THRESH', '0.10'))
    risk = 0.0
    for pc in pocket_centers:
        w = pc - cue_xy
        proj = float(np.dot(w, u))
        if proj <= 0:
            continue
        closest = float(np.linalg.norm(w - proj * u))
        if closest < threshold:
            risk = max(risk, (threshold - closest) / max(1e-6, threshold))
    return float(np.clip(risk, 0.0, 1.0))


def _parse_csv_floats(s: str, default: List[float]) -> List[float]:
    try:
        parts = [p.strip() for p in s.split(',') if p.strip()]
        if not parts:
            return default
        return [float(p) for p in parts]
    except Exception:
        return default


class SearchAgentV1_5(Agent):
    def __init__(self):
        super().__init__()
        self.ball_radius = 0.028575

        self.noise_std = {
            'V0': 0.1,
            'phi': 0.15,
            'theta': 0.1,
            'a': 0.005,
            'b': 0.005,
        }

        # champion schedule defaults
        self.n_simulations = int(os.environ.get('BILLIARDS_MAX_SIMULATIONS', '180'))
        self.risk_lambda = float(os.environ.get('BILLIARDS_RISK_LAMBDA', '0.3'))
        self.debug = os.environ.get('BILLIARDS_DEBUG', '0') == '1'

        # no-foul library parameters
        self.nofoul_keep = int(os.environ.get('BILLIARDS_NOFOUL_COUNT', '10'))
        self.nofoul_v0s = _parse_csv_floats(os.environ.get('BILLIARDS_NOFOUL_V0S', '2.2,3.0'), [2.2, 3.0])
        self.nofoul_angle_offs = _parse_csv_floats(os.environ.get('BILLIARDS_NOFOUL_ANGLE_OFFS', '0,7,-7'), [0.0, 7.0, -7.0])

    def _calc_angle_degrees(self, v):
        return float(math.degrees(math.atan2(v[1], v[0])) % 360)

    def _get_ghost_ball_target(self, cue_pos, obj_pos, pocket_pos):
        vec_obj_to_pocket = np.array(pocket_pos) - np.array(obj_pos)
        dist_obj_to_pocket = np.linalg.norm(vec_obj_to_pocket)
        if dist_obj_to_pocket < 1e-6:
            return 0.0, 0.0
        unit_vec = vec_obj_to_pocket / dist_obj_to_pocket
        ghost_pos = np.array(obj_pos) - unit_vec * (2 * self.ball_radius)
        vec_cue_to_ghost = ghost_pos - np.array(cue_pos)
        dist_cue_to_ghost = np.linalg.norm(vec_cue_to_ghost)
        phi = self._calc_angle_degrees(vec_cue_to_ghost)
        return phi, float(dist_cue_to_ghost)

    def _generate_nofoul_candidates(self, balls, my_targets) -> List[dict]:
        """Explicit 'no-foul' shots: direct-hit target with medium speed.

        These are not optimized for potting; they are insurance actions that tend
        to (a) hit a legal ball first and (b) produce some rail contact.
        """
        cue_xy = ball_pos(balls, 'cue')
        if cue_xy is None:
            return []

        target_ids = [bid for bid in my_targets if bid in balls and balls[bid].state.s != 4]
        if not target_ids:
            target_ids = ['8']

        nofoul: List[dict] = []
        for tid in target_ids:
            obj_xy = ball_pos(balls, tid)
            if obj_xy is None:
                continue
            v = obj_xy - cue_xy
            dist = float(np.linalg.norm(v))
            if dist < 1e-6:
                continue
            phi_direct = self._calc_angle_degrees(v)

            for dphi in self.nofoul_angle_offs:
                phi = wrap_angle(phi_direct + float(dphi))
                for V0 in self.nofoul_v0s:
                    nofoul.append({
                        'V0': float(V0),
                        'phi': float(phi),
                        'theta': 0,
                        'a': 0.0,
                        'b': 0.0,
                        'type': 'nofoul',
                        'target': tid,
                    })

        random.shuffle(nofoul)
        return nofoul

    # ---- Candidate generation copied from v1_4 ----
    def generate_candidates(self, balls, my_targets, table):
        actions: List[dict] = []

        cue_xy = ball_pos(balls, 'cue')
        if cue_xy is None:
            return [self._random_action()]

        target_ids = [bid for bid in my_targets if bid in balls and balls[bid].state.s != 4]
        if not target_ids:
            target_ids = ['8']

        pocket_centers = [np.array([p.center[0], p.center[1]], dtype=float) for p in table.pockets.values()]

        T = float(os.environ.get('BILLIARDS_THICKNESS_T', str(2.5 * self.ball_radius)))
        thickness_ts = np.array([-T, -0.75*T, -0.5*T, -0.25*T, 0.0, 0.25*T, 0.5*T, 0.75*T, T], dtype=float)

        for tid in target_ids:
            obj_xy = ball_pos(balls, tid)
            if obj_xy is None:
                continue

            for pocket_xy in pocket_centers:
                phi_ghost, dist_cue_to_ghost = self._get_ghost_ball_target(cue_xy, obj_xy, pocket_xy)
                if dist_cue_to_ghost < 1e-6:
                    continue

                v_base = float(np.clip(1.5 + dist_cue_to_ghost * 1.5, 1.0, 6.0))
                speeds = [
                    float(np.clip(v_base * 0.7, 1.0, 8.0)),
                    float(np.clip(v_base * 0.9, 1.0, 8.0)),
                    float(np.clip(v_base, 1.0, 8.0)),
                    float(np.clip(v_base * 1.2, 1.0, 8.0)),
                ]

                for t in thickness_ts:
                    delta_phi = float(np.degrees(np.arctan2(t, dist_cue_to_ghost)))
                    phi = wrap_angle(phi_ghost + delta_phi)
                    for V0 in speeds:
                        actions.append({
                            'V0': V0,
                            'phi': phi,
                            'theta': 0,
                            'a': 0.0,
                            'b': 0.0,
                            'type': 'direct_pot',
                            'target': tid,
                            'pocket_xy': pocket_xy,
                        })

        max_banks = int(os.environ.get('BILLIARDS_MAX_BANKS', '40'))
        bank_actions: List[dict] = []

        if pocket_centers:
            xs = [p[0] for p in pocket_centers]
            ys = [p[1] for p in pocket_centers]
            x_min, x_max = float(min(xs)), float(max(xs))
            y_min, y_max = float(min(ys)), float(max(ys))
        else:
            x_min, x_max, y_min, y_max = -1.0, 1.0, -0.5, 0.5

        def reflect_point(p: np.ndarray, which: str) -> np.ndarray:
            if which == 'left':
                return np.array([2*x_min - p[0], p[1]], dtype=float)
            if which == 'right':
                return np.array([2*x_max - p[0], p[1]], dtype=float)
            if which == 'bottom':
                return np.array([p[0], 2*y_min - p[1]], dtype=float)
            if which == 'top':
                return np.array([p[0], 2*y_max - p[1]], dtype=float)
            return p

        for tid in target_ids:
            obj_xy = ball_pos(balls, tid)
            if obj_xy is None:
                continue

            near_cushion = (
                min(abs(obj_xy[0] - x_min), abs(obj_xy[0] - x_max), abs(obj_xy[1] - y_min), abs(obj_xy[1] - y_max))
                < 0.25
            )
            if not near_cushion:
                continue

            for pocket_xy in pocket_centers:
                for which in ('left', 'right', 'top', 'bottom'):
                    pocket_mirror = reflect_point(pocket_xy, which)
                    phi_bank, dist = self._get_ghost_ball_target(cue_xy, obj_xy, pocket_mirror)
                    if dist < 1e-6:
                        continue
                    v_base = float(np.clip(2.0 + dist * 1.2, 1.2, 7.5))
                    for V0 in (v_base * 0.95, v_base, min(7.5, v_base * 1.1)):
                        bank_actions.append({
                            'V0': float(V0),
                            'phi': wrap_angle(phi_bank),
                            'theta': 0,
                            'a': 0.0,
                            'b': 0.0,
                            'type': 'bank_1rail',
                            'target': tid,
                            'pocket_xy': pocket_xy,
                            'bank_cushion': which,
                        })

        if len(bank_actions) > max_banks:
            random.shuffle(bank_actions)
            bank_actions = bank_actions[:max_banks]
        actions.extend(bank_actions)

        for tid in target_ids:
            obj_xy = ball_pos(balls, tid)
            if obj_xy is None:
                continue
            v = obj_xy - cue_xy
            dist = float(np.linalg.norm(v))
            if dist < 1e-6:
                continue
            phi_direct = self._calc_angle_degrees(v)

            for angle_offset in (-50, -35, -20, 20, 35, 50):
                phi = wrap_angle(phi_direct + angle_offset)
                for V0 in (0.8, 1.2, 1.6):
                    actions.append({
                        'V0': float(V0),
                        'phi': phi,
                        'theta': 0,
                        'a': 0.0,
                        'b': 0.0,
                        'type': 'safety_thin',
                        'target': tid,
                    })

            T_safe = float(os.environ.get('BILLIARDS_SAFE_THICKNESS_T', str(2.0 * self.ball_radius)))
            for t in (-T_safe, -0.5*T_safe, 0.0, 0.5*T_safe, T_safe):
                delta_phi = float(np.degrees(np.arctan2(t, max(dist, 1e-6))))
                phi = wrap_angle(phi_direct + delta_phi)
                for V0 in (0.8, 1.1, 1.4):
                    actions.append({
                        'V0': float(V0),
                        'phi': phi,
                        'theta': 0,
                        'a': 0.0,
                        'b': 0.0,
                        'type': 'safety_thickness',
                        'target': tid,
                    })

        # inject no-foul candidates
        actions.extend(self._generate_nofoul_candidates(balls, my_targets))

        for _ in range(12):
            a = self._random_action()
            a['type'] = 'random'
            actions.append(a)

        random.shuffle(actions)
        max_candidates = int(os.environ.get('BILLIARDS_MAX_CANDIDATES', '500'))
        return actions[:max_candidates]

    def _prefilter_candidates(self, balls, candidates: List[dict], pocket_centers: List[np.ndarray]) -> List[dict]:
        """v1_4 prefilter + force keep nofoul shots."""
        cue_xy = ball_pos(balls, 'cue')
        if cue_xy is None:
            return candidates

        keep_total = int(os.environ.get('BILLIARDS_PREFILTER_KEEP', '90'))
        keep_safety = int(os.environ.get('BILLIARDS_PREFILTER_KEEP_SAFETY', '25'))
        keep_bank = int(os.environ.get('BILLIARDS_PREFILTER_KEEP_BANK', '10'))
        keep_pot = max(10, keep_total - keep_safety - keep_bank)

        scored_pots: List[Tuple[float, dict]] = []
        scored_banks: List[Tuple[float, dict]] = []
        scored_safety: List[Tuple[float, dict]] = []
        nofoul: List[dict] = []
        others: List[dict] = []

        w_cut = float(os.environ.get('BILLIARDS_W_CUT', '1.0'))
        w_d_obj_pocket = float(os.environ.get('BILLIARDS_W_OBJ_POCKET', '0.6'))
        w_d_cue_obj = float(os.environ.get('BILLIARDS_W_CUE_OBJ', '0.35'))
        w_block = float(os.environ.get('BILLIARDS_W_BLOCK', '2.0'))
        w_scratch = float(os.environ.get('BILLIARDS_W_SCRATCH', '1.8'))
        w_speed = float(os.environ.get('BILLIARDS_W_SPEED', '0.4'))

        extreme_cut = float(os.environ.get('BILLIARDS_EXTREME_CUT_DEG', '75'))

        for a in candidates:
            atype = a.get('type', '')
            tid = a.get('target', None)
            obj_xy = ball_pos(balls, tid) if tid else None

            if atype == 'nofoul':
                nofoul.append(a)
                continue

            if atype in ('direct_pot', 'bank_1rail') and obj_xy is not None and pocket_centers:
                pocket_xy = a.get('pocket_xy', pocket_centers[0])

                blocked1 = is_line_blocked(cue_xy, obj_xy, balls, self.ball_radius, ignore_ids={'cue', str(tid)})
                blocked2 = is_line_blocked(obj_xy, pocket_xy, balls, self.ball_radius, ignore_ids={str(tid)})

                ca = cut_angle_deg(cue_xy, obj_xy, pocket_xy)
                d_cue_obj = float(np.linalg.norm(obj_xy - cue_xy))
                d_obj_pocket = float(np.linalg.norm(pocket_xy - obj_xy))

                diff_cut = ca / 90.0
                diff_dist = (d_obj_pocket / 2.54)
                diff_cue = (d_cue_obj / 2.54)

                phi = float(a['phi'])
                scratch_r = ray_to_pocket_risk(cue_xy, phi, pocket_centers)

                extreme_pen = 0.0
                if ca > extreme_cut:
                    if blocked1 or blocked2:
                        extreme_pen = 2.0
                    else:
                        extreme_pen = 0.7

                V0 = float(a.get('V0', 1.0))
                v_norm = V0 / 8.0

                score = 0.0
                score += w_block * (1.0 if (not blocked1) else -1.0)
                score += w_block * (1.0 if (not blocked2) else -1.0)
                score -= w_cut * diff_cut
                score -= w_d_obj_pocket * diff_dist
                score -= w_d_cue_obj * diff_cue
                score -= w_scratch * scratch_r
                score -= w_speed * v_norm * scratch_r
                score -= extreme_pen

                if atype == 'direct_pot':
                    scored_pots.append((score, a))
                else:
                    scored_banks.append((score, a))

            elif 'safety' in atype:
                score = 0.0
                if obj_xy is not None:
                    blocked = is_line_blocked(cue_xy, obj_xy, balls, self.ball_radius, ignore_ids={'cue', str(tid)})
                    score += 1.5 if not blocked else -1.5
                score -= 0.2 * float(a.get('V0', 1.0))
                scored_safety.append((score, a))

            else:
                others.append(a)

        scored_pots.sort(key=lambda x: x[0], reverse=True)
        scored_banks.sort(key=lambda x: x[0], reverse=True)
        scored_safety.sort(key=lambda x: x[0], reverse=True)

        kept: List[dict] = []
        kept.extend([a for _, a in scored_pots[:keep_pot]])
        kept.extend([a for _, a in scored_banks[:keep_bank]])
        kept.extend([a for _, a in scored_safety[:keep_safety]])

        # force keep some nofoul candidates
        if self.nofoul_keep > 0 and nofoul:
            random.shuffle(nofoul)
            kept.extend(nofoul[: min(self.nofoul_keep, max(0, keep_total - len(kept)))])

        # fill remaining
        if len(kept) < keep_total:
            random.shuffle(others)
            kept.extend(others[: (keep_total - len(kept))])

        random.shuffle(kept)
        return kept[:keep_total]

    def simulate_action(self, balls, table, action, my_targets, pocket_centers: Optional[List[np.ndarray]] = None):
        try:
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            sim_table = copy.deepcopy(table)
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

            cue = pt.Cue(cue_ball_id="cue")
            shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)

            noisy_V0 = float(np.clip(action['V0'] + np.random.normal(0, self.noise_std['V0']), 0.5, 8.0))
            noisy_phi = wrap_angle(action['phi'] + np.random.normal(0, self.noise_std['phi']))
            noisy_theta = float(np.clip(action.get('theta', 0) + np.random.normal(0, self.noise_std['theta']), 0, 90))
            noisy_a = float(np.clip(action.get('a', 0) + np.random.normal(0, self.noise_std['a']), -0.5, 0.5))
            noisy_b = float(np.clip(action.get('b', 0) + np.random.normal(0, self.noise_std['b']), -0.5, 0.5))

            cue.set_state(V0=noisy_V0, phi=noisy_phi, theta=noisy_theta, a=noisy_a, b=noisy_b)

            success = simulate_with_timeout(shot, timeout=3)
            if not success:
                return None, -500.0, -500.0

            raw = float(analyze_shot_for_reward(shot, last_state_snapshot, my_targets))
            shaped = raw

            pocket_centers = pocket_centers or []

            # scratch-risk shaping: cue ends near a pocket
            try:
                cue_after = ball_pos(shot.balls, 'cue')
                dmin = nearest_pocket_distance(cue_after, pocket_centers)
                if dmin < 2.2 * self.ball_radius:
                    shaped -= 15.0
            except Exception:
                pass

            return shot, raw, shaped

        except Exception:
            return None, -1000.0, -1000.0

    def decision(self, balls=None, my_targets=None, table=None):
        # Same two-stage schedule as v1_4 (champion defaults)
        if balls is None:
            return self._random_action()

        remaining = [bid for bid in my_targets if bid in balls and balls[bid].state.s != 4]
        if len(remaining) == 0:
            my_targets = ["8"]

        pocket_centers = [np.array([p.center[0], p.center[1]], dtype=float) for p in table.pockets.values()]

        candidates = self.generate_candidates(balls, my_targets, table)
        if not candidates:
            return self._random_action()

        candidates = self._prefilter_candidates(balls, candidates, pocket_centers)
        if not candidates:
            return self._random_action()

        total_budget = int(max(1, self.n_simulations))

        req_stage1_n = int(os.environ.get('BILLIARDS_STAGE1_N', '90'))
        req_stage1_r = int(os.environ.get('BILLIARDS_STAGE1_R', '1'))
        req_stage2_k = int(os.environ.get('BILLIARDS_STAGE2_K', '12'))
        req_stage2_m = int(os.environ.get('BILLIARDS_STAGE2_M', '3'))

        req_stage1_n = int(np.clip(req_stage1_n, 1, len(candidates)))
        req_stage1_r = max(1, req_stage1_r)
        req_stage2_k = max(1, req_stage2_k)
        req_stage2_m = max(0, req_stage2_m)

        stage2_k = min(req_stage2_k, len(candidates))
        stage2_k = max(1, min(stage2_k, total_budget))

        max_m = max(0, (total_budget // stage2_k) - 1)
        stage2_m = min(req_stage2_m, max_m)

        stage2_budget = stage2_k * (1 + stage2_m)
        budget_left = max(0, total_budget - stage2_budget)

        stage1_r = min(req_stage1_r, max(1, budget_left)) if budget_left > 0 else 1
        stage1_n = min(req_stage1_n, len(candidates))
        if budget_left > 0:
            stage1_n = max(1, min(stage1_n, budget_left // stage1_r))
        else:
            stage1_n = stage2_k
            stage1_r = 1

        stage1_actions = list(candidates)[:stage1_n]

        def norm(v: float) -> float:
            return float(np.clip((v - (-500.0)) / 650.0, 0.0, 1.0))

        s1_sums = np.zeros(stage1_n, dtype=float)
        s1_sums2 = np.zeros(stage1_n, dtype=float)
        s1_counts = np.zeros(stage1_n, dtype=int)

        for _ in range(stage1_r):
            for i, a in enumerate(stage1_actions):
                _, _, shaped = self.simulate_action(balls, table, a, my_targets, pocket_centers=pocket_centers)
                v = norm(shaped)
                s1_sums[i] += v
                s1_sums2[i] += v * v
                s1_counts[i] += 1

        s1_means = s1_sums / (s1_counts + 1e-9)
        s1_vars = (s1_sums2 / (s1_counts + 1e-9)) - s1_means * s1_means
        s1_stds = np.sqrt(np.maximum(0.0, s1_vars))
        s1_est = s1_means - float(self.risk_lambda) * s1_stds

        stage2_k = min(stage2_k, stage1_n)
        top_idx = np.argsort(s1_est)[-stage2_k:][::-1]
        finalists = [stage1_actions[int(i)] for i in top_idx]

        k = len(finalists)
        sums = np.zeros(k, dtype=float)
        sums2 = np.zeros(k, dtype=float)
        counts = np.zeros(k, dtype=int)

        for j, idx in enumerate(top_idx):
            idx = int(idx)
            sums[j] = float(s1_sums[idx])
            sums2[j] = float(s1_sums2[idx])
            counts[j] = int(s1_counts[idx])

        for _ in range(stage2_m):
            for j, a in enumerate(finalists):
                _, _, shaped = self.simulate_action(balls, table, a, my_targets, pocket_centers=pocket_centers)
                v = norm(shaped)
                sums[j] += v
                sums2[j] += v * v
                counts[j] += 1

        means = sums / (counts + 1e-9)
        vars_ = (sums2 / (counts + 1e-9)) - means * means
        stds = np.sqrt(np.maximum(0.0, vars_))
        estimates = means - float(self.risk_lambda) * stds

        best_idx = int(np.argmax(estimates))
        best_action = finalists[best_idx]

        return {
            'V0': float(best_action['V0']),
            'phi': float(best_action['phi']),
            'theta': float(best_action.get('theta', 0)),
            'a': float(best_action.get('a', 0)),
            'b': float(best_action.get('b', 0)),
        }


SearchAgent = SearchAgentV1_5

