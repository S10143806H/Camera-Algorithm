"""
Flicker Detection Algorithm v4.1 - Gain-Normalized Local-Energy Edition

Rationale (vs v3.2, targeting camera-pointed-at-screen false positives
while keeping small-area transient flashes detectable):

  1. Global gain normalization (per-pixel):
     diff = gray_t - gray_{t-1}; gain = median(diff) estimates AE/AWB drift
     and content-wide fades; ndiff = diff - gain keeps only local residuals.
  2. Local flash energy:
     energy_map = boxfilter(|ndiff|, 15x15) on a 320-wide proc frame,
     energy = max(energy_map). Detects flashes as small as ~10px on the
     proc frame regardless of any tile grid alignment (v4.0's 8x4 tile
     means diluted small flashes ~30x and missed them).
  3. Structure-invariance gate:
     real flicker changes brightness but not the edge structure; content
     motion / scene cuts change edges too. Edge-map Dice similarity < lo
     discounts the score heavily.
  4. ABA pulse bonus:
     flash-and-return leaves opposite-signed ndiff at the same location in
     consecutive samples; AE steps are monotonic. Adds confidence, not
     required (long flashes still score via spike alone).
  5. Rolling robust baseline:
     spike score = (energy - rolling_median) / rolling_MAD, so the detector
     adapts to each source's noise floor continuously (no one-shot warmup).
  6. Event decision on RAW confidence:
     transient flicker lasts 1-3 frames; v3.2's EMA + min_trigger + burst
     chain structurally suppresses single pulses. v4 triggers events on
     raw confidence (with cooldown); EMA kept only for display smoothing.

Interface-compatible with v3.2 usage in usb_camera_preview.py:
FlickerDetectorV4.push_frame(frame) -> flicker_detector.FlickerResult.
For offline files use detect_video(..., sample_step=1).
"""

from __future__ import annotations

import cv2
import numpy as np
from collections import deque

from flicker_detector import FlickerResult


class FlickerDetectorV4:
    """Gain-normalized local-energy flicker detector with structure gating."""

    PROC_WIDTH = 320
    LOCAL_WIN = 15            # box filter window on proc frame (px)

    def __init__(self, window_size=120, threshold=0.70, fps=30.0,
                 sample_step=2, min_trigger=1, ema_alpha=0.30,
                 struct_gate_lo=0.50, struct_gate_hi=0.72,
                 spike_lo=5.0, spike_hi=14.0,
                 energy_floor=1.5,
                 strong_single=0.85):
        self.window_size = window_size
        self.threshold = threshold
        self.fps = fps
        if sample_step <= 0:
            self.sample_step = 3 if fps > 25 else 2
        else:
            self.sample_step = max(1, sample_step)
        self.min_trigger = min_trigger        # kept for interface compat
        self.ema_alpha = ema_alpha
        self.struct_gate_lo = struct_gate_lo
        self.struct_gate_hi = struct_gate_hi
        self.spike_lo = spike_lo
        self.spike_hi = spike_hi
        self.energy_floor = energy_floor
        self.strong_single = strong_single

        self._prev_gray = None
        self._prev_edges = None
        self._ndiff_hist = deque(maxlen=3)    # for ABA lookback
        self._gain_hist = deque(maxlen=4)     # for global-flash ABA
        self._energy_hist = deque(maxlen=window_size)
        self.flicker_events = deque(maxlen=window_size)
        self._frame_counter = 0
        self._sampled_count = 0
        self._cooldown = 0
        self._ema_conf = 0.0
        self._warmup_samples = max(12, int((fps / self.sample_step) * 2.0))

        # exposed for annotation
        self.last_flick_tile_box = None       # (x1,y1,x2,y2) in input coords
        self.last_struct_sim = 1.0
        self.last_gain = 0.0

    def reset(self):
        self._prev_gray = None
        self._prev_edges = None
        self._ndiff_hist.clear()
        self._gain_hist.clear()
        self._energy_hist.clear()
        self.flicker_events.clear()
        self._frame_counter = 0
        self._sampled_count = 0
        self._cooldown = 0
        self._ema_conf = 0.0
        self.last_flick_tile_box = None
        self.last_struct_sim = 1.0
        self.last_gain = 0.0

    # ---- helpers -------------------------------------------------------

    @staticmethod
    def _edge_map(gray: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        thr = max(20.0, float(np.percentile(mag, 75)))
        return (mag > thr)

    @staticmethod
    def _dice(a: np.ndarray, b: np.ndarray) -> float:
        sa, sb = int(a.sum()), int(b.sum())
        if sa + sb == 0:
            return 1.0
        return 2.0 * float(np.logical_and(a, b).sum()) / (sa + sb)

    def _window_mean(self, m: np.ndarray, cy: int, cx: int) -> float:
        r = self.LOCAL_WIN // 2
        h, w = m.shape
        y1, y2 = max(0, cy - r), min(h, cy + r + 1)
        x1, x2 = max(0, cx - r), min(w, cx + r + 1)
        return float(m[y1:y2, x1:x2].mean())

    # ---- main ----------------------------------------------------------

    def push_frame(self, frame: np.ndarray) -> FlickerResult:
        self._frame_counter += 1
        if self._frame_counter % self.sample_step != 0:
            return self._passthrough_result()

        h, w = frame.shape[:2]
        scale = self.PROC_WIDTH / float(w)
        proc_h = max(2, int(h * scale))
        gray = cv2.cvtColor(
            cv2.resize(frame, (self.PROC_WIDTH, proc_h), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY).astype(np.float32)
        edges = self._edge_map(gray)
        self._sampled_count += 1

        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_edges = edges
            return FlickerResult(False, 0.0, 0.0, 0.0, {}, 0, 0.0, baseline_ready=False)

        # 1) per-pixel diff with global gain removed
        diff = gray - self._prev_gray
        gain = float(np.median(diff))
        ndiff = diff - gain

        # 2) local flash energy
        energy_map = cv2.blur(np.abs(ndiff), (self.LOCAL_WIN, self.LOCAL_WIN))
        cy, cx = np.unravel_index(int(energy_map.argmax()), energy_map.shape)
        energy = float(energy_map[cy, cx])

        # 3) structure gate
        struct_sim = self._dice(edges, self._prev_edges)
        if struct_sim <= self.struct_gate_lo:
            gate = 0.12
        elif struct_sim >= self.struct_gate_hi:
            gate = 1.0
        else:
            gate = 0.12 + 0.88 * ((struct_sim - self.struct_gate_lo)
                                  / (self.struct_gate_hi - self.struct_gate_lo))

        # 4) ABA pulse bonus: opposite-signed residual at same spot
        cur_local = self._window_mean(ndiff, cy, cx)
        aba = 0.0
        for prev_nd in self._ndiff_hist:
            prev_local = self._window_mean(prev_nd, cy, cx)
            if prev_local * cur_local < 0:
                mag = min(abs(prev_local), abs(cur_local))
                cancel = abs(prev_local + cur_local) / (abs(prev_local) + abs(cur_local) + 1e-6)
                if mag > 1.5 and cancel < 0.65:
                    aba = max(aba, float(np.clip(mag / 10.0, 0.0, 1.0)))
        self._ndiff_hist.append(ndiff)

        # 5) rolling robust baseline -> spike score
        baseline_ready = self._sampled_count > self._warmup_samples
        if len(self._energy_hist) >= 10:
            hist = np.asarray(self._energy_hist)
            med = float(np.median(hist))
            mad = float(np.median(np.abs(hist - med))) + 0.05
            ratio = (energy - med) / mad
        else:
            ratio = 0.0
        self._energy_hist.append(energy)   # after scoring: spike can't mask itself

        if ratio <= self.spike_lo:
            s_spike = max(0.0, ratio / self.spike_lo * 0.10)
        elif ratio >= self.spike_hi:
            s_spike = 1.0
        else:
            s_spike = 0.10 + 0.90 * (ratio - self.spike_lo) / (self.spike_hi - self.spike_lo)
        if energy < self.energy_floor:
            s_spike = 0.0

        # ABA is independent strong evidence for sustained flicker (the
        # oscillation IS the rolling baseline there, so spike goes blind),
        # but only when carried by real energy:
        s_aba = aba * float(np.clip((energy - 3.0) / 6.0, 0.0, 1.0))
        base = max(s_spike, 0.5 * s_spike + 0.78 * s_aba)
        if s_spike > 0.5 and s_aba > 0.5:
            base = min(1.0, base + 0.15)

        # content-appear suppression: a window/panel popping up brings NEW
        # rich edge structure that stays (e.g. AVM view opening); flicker
        # flashes are flat blocks (black/white) or vanishing content.
        # Re-judge the event ROI with black-screen-style metrics: if the new
        # state IS a flat dark/bright block, it stays a flicker; only
        # structured content that appears gets suppressed.
        cur_ed = self._window_mean(edges.astype(np.float32), cy, cx)
        prev_ed = self._window_mean(self._prev_edges.astype(np.float32), cy, cx)
        r = self.LOCAL_WIN // 2
        roi = gray[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1]
        roi_mean = float(roi.mean())
        flat_dark = roi_mean < 75 and float((roi < 60).mean()) > 0.55   # black-screen block
        flat_bright = roi_mean > 180 and float(roi.std()) < 18          # white-flash block
        low_texture = cur_ed <= 0.05
        is_flat_flash = flat_dark or flat_bright or low_texture
        content_appear = (not is_flat_flash) and (cur_ed > prev_ed * 2 + 0.02) and (aba < 0.30)
        if content_appear:
            base *= 0.3

        # ---- global-flash path (large-area flicker) ----
        # When MOST of the frame flashes, median(diff) IS the flash, so the
        # gain subtraction nulls the local path, and the vanished edges also
        # trip the structure gate. Handle it globally, bypassing the gate:
        # a big uniform jump counts as flicker if the new frame is a flat
        # dark/bright screen (black/white flash) or the gain sequence is a
        # spike-and-return (ABA). A one-way jump into structured content is
        # a scene change, not flicker.
        g_flash = 0.0
        if abs(gain) > 6.0:
            frame_mean = float(gray.mean())
            flat_dark_f = frame_mean < 75 and float((gray < 60).mean()) > 0.55
            flat_bright_f = frame_mean > 190 and float(gray.std()) < 20
            g_aba = 0.0
            for pg in self._gain_hist:
                if pg * gain < 0:
                    mag = min(abs(pg), abs(gain))
                    cancel = abs(pg + gain) / (abs(pg) + abs(gain) + 1e-6)
                    if mag > 4.0 and cancel < 0.65:
                        g_aba = max(g_aba, min(1.0, mag / 25.0))
            g_flash = float(np.clip((abs(gain) - 6.0) / 24.0, 0.0, 0.6))
            if flat_dark_f or flat_bright_f:
                g_flash = max(g_flash, 0.75)
            if g_aba > 0:
                g_flash = min(1.0, g_flash + 0.35 + 0.25 * g_aba)
        self._gain_hist.append(gain)

        # structure gate exists to suppress content motion, but a flash that
        # RETURNS (strong ABA) is self-validating — the vanished edges during
        # the flash are the phenomenon itself, not content change. Lift the
        # gate proportionally to ABA strength so large-area flashes (which
        # wipe out half the edge map) are not double-suppressed:
        gate_eff = max(gate, min(1.0, 0.45 + 0.60 * s_aba))

        global_event = g_flash > gate_eff * base
        raw_conf = max(gate_eff * base, g_flash)
        if not baseline_ready:
            raw_conf *= 0.5

        # EMA only for display
        self._ema_conf = self.ema_alpha * raw_conf + (1 - self.ema_alpha) * self._ema_conf

        # 6) event decision on raw confidence
        if self._cooldown > 0:
            self._cooldown -= 1
        is_event = raw_conf >= self.threshold and self._cooldown == 0
        if is_event:
            self._cooldown = max(4, int(self.fps / self.sample_step * 0.3))
            self.flicker_events.append((1, self._frame_counter))
        else:
            self.flicker_events.append((0, self._frame_counter))

        flick_count_evt = sum(e[0] for e in self.flicker_events)
        # report flicker on event frames; strong single events always count
        is_flick = bool(is_event and (raw_conf >= self.strong_single
                                      or aba > 0.30
                                      or flick_count_evt >= 2
                                      or raw_conf >= self.threshold))

        self._prev_gray = gray
        self._prev_edges = edges

        # annotation box in input coords: whole frame for global flashes,
        # else centered on the local argmax window
        if global_event:
            self.last_flick_tile_box = (4, 4, w - 4, h - 4)
        else:
            r = int(self.LOCAL_WIN * 1.2)
            x1 = int(max(0, (cx - r)) / scale)
            y1 = int(max(0, (cy - r)) / scale)
            x2 = int(min(self.PROC_WIDTH, cx + r) / scale)
            y2 = int(min(proc_h, cy + r) / scale)
            self.last_flick_tile_box = (x1, y1, x2, y2)
        self.last_struct_sim = struct_sim
        self.last_gain = gain

        effective_fps = self.fps / self.sample_step
        window_dur = len(self.flicker_events) / max(effective_fps, 1)
        flick_rate = flick_count_evt / max(window_dur, 0.1)

        return FlickerResult(
            is_flickering=is_flick,
            confidence=round(self._ema_conf, 4),
            raw_confidence=round(raw_conf, 4),
            anomaly_ratio=round(min(max(ratio, 0.0) / self.spike_hi, 1.0), 4),
            details={
                'spike': round(s_spike, 4),
                'aba': round(aba, 4),
                'struct_sim': round(struct_sim, 4),
                'gate': round(gate, 4),
                'gain': round(gain, 3),
                'energy': round(energy, 3),
                'ratio': round(ratio, 2),
                'g_flash': round(g_flash, 4),
            },
            flicker_count=flick_count_evt,
            flicker_rate=round(flick_rate, 2),
            baseline_ready=baseline_ready,
        )

    def _passthrough_result(self) -> FlickerResult:
        return FlickerResult(
            is_flickering=False,
            confidence=round(self._ema_conf, 4),
            raw_confidence=round(self._ema_conf, 4),
            anomaly_ratio=0.0,
            details={},
            flicker_count=sum(e[0] for e in self.flicker_events),
            flicker_rate=0.0,
            baseline_ready=self._sampled_count > self._warmup_samples,
        )

    # ---- file batch ----------------------------------------------------

    @staticmethod
    def detect_video(video_path, threshold=0.70, sample_step=1):
        """Offline analysis: sample_step=1 so 1-frame transients are never skipped."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, sample_step)
        det = FlickerDetectorV4(
            window_size=int(fps / step * 4),
            threshold=threshold,
            fps=fps,
            sample_step=step,
        )
        results = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            r = det.push_frame(frame)
            results.append((idx, r, det.last_flick_tile_box))
            idx += 1
        cap.release()
        return results, fps


if __name__ == "__main__":
    import sys, os
    if len(sys.argv) < 2:
        print("Usage: python flicker_detector_v4.py <video_path> [threshold]")
        sys.exit(1)
    path = sys.argv[1]
    thresh = float(sys.argv[2]) if len(sys.argv) > 2 else 0.70
    results, fps = FlickerDetectorV4.detect_video(path, threshold=thresh)
    flick = [(f, r) for f, r, _ in results if r.is_flickering]
    print(f"video={os.path.basename(path)} fps={fps:.1f} frames={len(results)} events={len(flick)}")
    for f, r in flick[:40]:
        d = r.details
        print(f"  #{f:>6d} t={f/fps:6.2f}s raw={r.raw_confidence:.0%} "
              f"spike={d.get('spike',0):.0%} aba={d.get('aba',0):.0%} "
              f"sim={d.get('struct_sim',1):.2f} gain={d.get('gain',0):+.1f} "
              f"E={d.get('energy',0):.1f} ratio={d.get('ratio',0):.1f}")
