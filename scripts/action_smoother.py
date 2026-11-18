"""Model-agnostic smoothing of chunked policy actions (no retraining). numpy only: used by humanoid_pallet_sim.py
(Isaac python) and by offline checks (env python).

Two layers, for any policy that returns action chunks (ACT, SmolVLA, Diffusion, pi0, ...):
1. Chunk crossfade: chunks are predicted synchronously, so new_chunk[i] and the unexecuted tail of the previous chunk
   old_tail[i] are both targets for the same future step. The first K executed actions of a new chunk are blended
   linearly from the old tail to the new chunk (w = (i+1)/(K+1)), so the target does not jump at a chunk join.
2. Joint rate limiter like the motor's own trajectory limits (moteus position mode with velocity / acceleration
   limits): every joint target moves towards the command at most v_max and changes speed at most a_max per second,
   braking early enough not to overshoot, and following a smoothly moving command without lag (feed-forward of the
   command's own speed). a_max = 6 v_max (default): the expert's recorded actions pass with < 15 mrad deviation and
   none at grasp / release (their segment joins reach 5-26 v_max for one step), while the policies' chunk-join jumps
   are halved and their p99 jerk drops 3.5-4.5x (offline check on dataset observations, 2026-10-08).
3. Optional in-chunk low-pass (filter_window > 0): a Savitzky-Golay filter over the whole predicted chunk, joint
   channels only, before the crossfade. Flow-matching chunks (SmolVLA) wiggle inside the chunk (joint velocity flips
   ~8 times per second, the expert ~0); layers 1-2 only act at joins and on large accelerations. Fitting the window
   over the full chunk (50 steps) puts the executed head away from the far edge, and the near edge uses the same
   polynomial fit as the interior (no lag).
Gripper channels are crossfaded but not rate limited (the gripper has its own 0.1 m/s speed limit in the sim).
"""
import numpy as np


def savgol(x, window, order=2):
    """Savitzky-Golay smoothing of x (T, C) along axis 0, numpy only. Interior: least-squares polynomial of `order`
    over a centred `window`; the first / last window//2 rows: the polynomial fitted to the first / last window."""
    x = np.asarray(x, float)
    T, h = len(x), window // 2
    if window < 3 or T < window:
        return x.copy()
    t = np.arange(-h, h + 1, dtype=float)
    A = np.vander(t, order + 1, increasing=True)  # (window, order+1)
    P = np.linalg.pinv(A)  # coefficients = P @ y
    out = np.empty_like(x)
    win = np.lib.stride_tricks.sliding_window_view(x, window, axis=0)  # (T-window+1, C, window)
    out[h:T - h] = win @ P[0]  # value at the window centre
    for lo, rows in ((0, range(h)), (T - window, range(T - h, T))):
        coef = P @ x[lo:lo + window]  # (order+1, C)
        for r in rows:
            out[r] = np.vander([r - lo - h], order + 1, increasing=True)[0] @ coef
    return out


class ActionSmoother:
    def __init__(self, joint_idx, vmax, dt, accel_scale=6.0, blend_steps=8, filter_window=0, filter_order=2):
        """joint_idx: indices of the joint channels in the action vector; vmax: their velocity limits (rad/s)."""
        self.ji = np.asarray(joint_idx)
        self.vmax = np.asarray(vmax, float)
        self.amax = accel_scale * self.vmax
        self.dt, self.blend_steps = float(dt), int(blend_steps)
        self.filter_window, self.filter_order = int(filter_window), int(filter_order)
        self.tail = None
        self.q = self.v = self.cmd_prev = None

    def reset(self, q0):
        """q0: the current joint targets (len(joint_idx)) at the start of an episode."""
        self.tail = None
        self.q = np.asarray(q0, float).copy()
        self.v = np.zeros_like(self.q)
        self.cmd_prev = None

    def new_chunk(self, chunk, exec_steps):
        """chunk (T, A) -> the (<= exec_steps, A) actions to execute, crossfaded with the previous chunk's tail."""
        chunk = np.asarray(chunk, float).copy()
        if self.filter_window:
            chunk[:, self.ji] = savgol(chunk[:, self.ji], self.filter_window, self.filter_order)
        seq = chunk[:exec_steps].copy()
        if self.tail is not None and self.blend_steps > 0:
            k = min(self.blend_steps, len(self.tail), len(seq))
            w = (np.arange(k) + 1.0) / (k + 1.0)
            seq[:k] = w[:, None] * seq[:k] + (1.0 - w[:, None]) * self.tail[:k]
        self.tail = chunk[exec_steps:].copy() if len(chunk) > exec_steps else None
        return seq

    def step(self, action):
        """One control step: the rate-limited action (joint channels replaced, the rest passed through).
        Desired speed = the speed that reaches the command this step, e/dt. Its part beyond the command's own speed
        (v_ff) is capped at sqrt(2 a_max |e|), so a target that jumped is approached with braking instead of overshoot,
        while a smoothly moving command (e = v_ff dt) is followed exactly; then |v| <= v_max and |dv/dt| <= a_max."""
        out = np.asarray(action, float).copy()
        cmd = out[self.ji]
        v_ff = np.zeros_like(cmd) if self.cmd_prev is None else (cmd - self.cmd_prev) / self.dt
        self.cmd_prev = cmd.copy()
        e = cmd - self.q
        cap = np.sqrt(2.0 * self.amax * np.abs(e))
        v_des = np.clip(v_ff + np.clip(e / self.dt - v_ff, -cap, cap), -self.vmax, self.vmax)
        self.v = self.v + np.clip(v_des - self.v, -self.amax * self.dt, self.amax * self.dt)
        self.q = self.q + self.v * self.dt
        out[self.ji] = self.q
        return out
