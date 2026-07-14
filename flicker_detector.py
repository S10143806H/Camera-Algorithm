"""
Flicker Detection Algorithm v3.3 - Content-Motion Immune Edition
Based on v3.2 + anti-false-positive for normally-moving content (AVM/video playback):

Key changes (vs v3.2):
  1. Tile-normalized jump metric: frame split into 8x6 tiles; the signed
     per-tile delta minus the median tile delta removes global gain shifts
     (AE drift, content-wide brightness change); the max residual tile delta
     replaces the old global |mean diff| — real flicker is local or a step,
     content motion spreads evenly.
  2. Structure gate: Sobel edge-map correlation between consecutive frames.
     Content change = edges present in both frames but decorrelated -> jump
     deltas suppressed (x0.15). Flash flicker = edge energy collapses ->
     kept at full weight (a black/white flash must not be suppressed).
  3. Variance dims gated by the median motion factor of the window, so a
     continuously-moving scene no longer saturates bright_var/sat_var.

v3.2 features kept: warmup baseline, burst filter, cooldown, EMA, debounce.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from collections import deque


@dataclass
class FlickerResult:
    is_flickering: bool
    confidence: float           # 0~1 (EMA smoothed)
    raw_confidence: float       # 0~1 (before smoothing)
    anomaly_ratio: float        # strongest anomaly indicator value
    details: dict               # 4-dim individual scores
    flicker_count: int = 0      # flicker event count in sliding window
    flicker_rate: float = 0.0   # flicker frequency (events/sec)
    baseline_ready: bool = False  # whether warmup baseline is calibrated


class FlickerDetector:
    """
    Temporal sliding window flicker detector v3.2
    Added: warmup baseline calibration + burst filter for live camera
    """

    def __init__(self, window_size=120, threshold=0.70, fps=30.0,
                 sample_step=2, min_trigger=3, ema_alpha=0.25,
                 jump_min_abs_brightness=1.0, jump_min_abs_sat=0.6):
        self.window_size = window_size
        self.threshold = threshold
        self.fps = fps
        # v3.3: absolute significance floors — ratio-based jump scores only
        # count when the (baseline-corrected) delta exceeds these magnitudes,
        # otherwise sub-gray-level sensor noise blows up the MAD ratios
        self.jump_min_abs_brightness = jump_min_abs_brightness
        self.jump_min_abs_sat = jump_min_abs_sat
        # adaptive sample_step: higher fps -> more downsample
        if sample_step <= 0:
            self.sample_step = 3 if fps > 25 else 2
        else:
            self.sample_step = max(1, sample_step)
        self.min_trigger = min_trigger
        self.ema_alpha = ema_alpha

        self.diff_brightness = deque(maxlen=window_size)
        self.diff_saturation = deque(maxlen=window_size)
        self.frame_brightness = deque(maxlen=window_size)
        self.frame_saturation = deque(maxlen=window_size)

        self._prev_mean = None
        self._prev_sat = None
        self._frame_counter = 0
        self.flicker_events = deque(maxlen=window_size)
        self._cooldown = 0
        self._ema_conf = 0.0
        self._above_thresh_count = 0

        # -- v3.3: tile grid + structure gate state --
        self._prev_tile = None      # 6x8 tile mean brightness of prev sampled frame
        self._prev_emag = None      # Sobel edge magnitude map of prev sampled frame
        self._motion_factors = deque(maxlen=window_size)  # 1.0=clean, 0.15=content motion

        # -- Baseline calibration (warmup) --
        self._baseline_samples = deque(maxlen=600)  # up to ~20s @ 1/2 sample
        self._baseline_db_med = None
        self._baseline_db_mad = None
        self._baseline_ds_med = None
        self._baseline_ds_mad = None
        self._baseline_ready = False
        self._warmup_frames = int((fps / self.sample_step) * 2.0)  # 2 seconds
        self._sampled_count = 0

    def reset(self):
        self.diff_brightness.clear()
        self.diff_saturation.clear()
        self.frame_brightness.clear()
        self.frame_saturation.clear()
        self.flicker_events.clear()
        self._baseline_samples.clear()
        self._prev_tile = None
        self._prev_emag = None
        self._motion_factors.clear()
        self._prev_mean = None
        self._prev_sat = None
        self._frame_counter = 0
        self._sampled_count = 0
        self._cooldown = 0
        self._ema_conf = 0.0
        self._above_thresh_count = 0
        self._baseline_db_med = None
        self._baseline_db_mad = None
        self._baseline_ds_med = None
        self._baseline_ds_mad = None
        self._baseline_ready = False

    def push_frame(self, frame: np.ndarray) -> FlickerResult:
        """Push a frame, auto-downsampled internally"""
        self._frame_counter += 1
        if self._frame_counter % self.sample_step != 0:
            return FlickerResult(
                is_flickering=self._ema_conf >= self.threshold and self._above_thresh_count >= self.min_trigger,
                confidence=round(self._ema_conf, 4),
                raw_confidence=round(self._ema_conf, 4),
                anomaly_ratio=0.0,
                details={},
                flicker_count=sum(e[0] for e in self.flicker_events),
                flicker_rate=0.0,
                baseline_ready=self._baseline_ready,
            )

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cur_mean = float(frame.mean())
        cur_sat = float(hsv[:, :, 1].mean())

        # -- v3.3: downscaled gray for tiles + edge structure --
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)
        tile = small.reshape(6, 15, 8, 20).mean(axis=(1, 3))  # 6 rows x 8 cols
        gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
        emag = cv2.magnitude(gx, gy)

        self.frame_brightness.append(cur_mean)
        self.frame_saturation.append(cur_sat)
        self._sampled_count += 1

        if self._prev_mean is None:
            self._prev_mean = cur_mean
            self._prev_sat = cur_sat
            self._prev_tile = tile
            self._prev_emag = emag
            return FlickerResult(False, 0.0, 0.0, 0.0, {}, 0, 0.0,
                                 baseline_ready=self._baseline_ready)

        # -- v3.3 metric 1: tile-normalized local jump --
        # signed per-tile delta; subtracting the median removes any global
        # gain shift (AE drift / content-wide brightness change)
        dtile = tile - self._prev_tile
        global_shift = float(np.median(dtile))
        db_local = float(np.abs(dtile - global_shift).max())

        # -- v3.3 metric 2: structure gate --
        # corr high  -> same structure, any brightness jump is suspicious (flicker)
        # corr low + edge energy comparable -> content changed, suppress
        # edge energy collapsed/appeared    -> flash-like, keep full weight
        a, b = emag.ravel(), self._prev_emag.ravel()
        if a.std() > 1e-3 and b.std() > 1e-3:
            struct_corr = float(np.corrcoef(a, b)[0, 1])
        else:
            struct_corr = 0.0 if (a.std() > 1e-3) != (b.std() > 1e-3) else 1.0
        e_cur, e_prev = float(emag.mean()), float(self._prev_emag.mean())
        e_ratio = min(e_cur, e_prev) / (max(e_cur, e_prev) + 1e-6)

        if e_ratio < 0.45:
            motion_factor = 1.0          # edges collapsed/appeared: flash-like, keep
        elif struct_corr < 0.55:
            motion_factor = 0.15         # structure replaced: content motion
        elif struct_corr < 0.75:
            motion_factor = 0.5          # partial motion
        else:
            motion_factor = 1.0          # structure stable
        self._motion_factors.append(motion_factor)

        db = db_local * motion_factor
        ds = abs(cur_sat - self._prev_sat) * motion_factor

        self.diff_brightness.append(db)
        self.diff_saturation.append(ds)

        self._prev_mean = cur_mean
        self._prev_sat = cur_sat
        self._prev_tile = tile
        self._prev_emag = emag

        if self._cooldown > 0:
            self._cooldown -= 1

        # -- Baseline calibration phase --
        if not self._baseline_ready:
            self._baseline_samples.append((db, ds))
            if self._sampled_count >= self._warmup_frames:
                arr = np.array(self._baseline_samples)
                self._baseline_db_med = float(np.median(arr[:, 0]))
                self._baseline_db_mad = float(np.median(np.abs(arr[:, 0] - self._baseline_db_med))) + 1e-7
                self._baseline_ds_med = float(np.median(arr[:, 1]))
                self._baseline_ds_mad = float(np.median(np.abs(arr[:, 1] - self._baseline_ds_med))) + 1e-7
                self._baseline_ready = True
                # Count outliers in baseline (should be near 0 for normal)
                db_outliers = (arr[:, 0] > self._baseline_db_med + 5 * self._baseline_db_mad).sum() / len(arr)
                ds_outliers = (arr[:, 1] > self._baseline_ds_med + 5 * self._baseline_ds_mad).sum() / len(arr)
                print(f"  [FLCK-BASELINE] db_med={self._baseline_db_med:.2f} db_mad={self._baseline_db_mad:.2f} "
                      f"ds_med={self._baseline_ds_med:.2f} ds_mad={self._baseline_ds_mad:.2f} "
                      f"db_outliers={db_outliers:.1%} ds_outliers={ds_outliers:.1%}")

            min_frames = max(15, self.window_size // 4)
            if len(self.diff_brightness) < min_frames:
                return FlickerResult(False, 0.0, 0.0, 0.0, {}, 0, 0.0,
                                     baseline_ready=self._baseline_ready)

        min_frames = max(15, self.window_size // 4)
        if len(self.diff_brightness) < min_frames:
            return FlickerResult(False, 0.0, 0.0, 0.0, {}, 0, 0.0,
                                 baseline_ready=self._baseline_ready)

        # -- Baseline-corrected deltas --
        db_arr = np.array(self.diff_brightness)
        ds_arr = np.array(self.diff_saturation)

        if self._baseline_ready:
            # Subtract baseline: spike = delta - baseline_median
            db_corrected = np.maximum(db_arr - self._baseline_db_med, 0)
            ds_corrected = np.maximum(ds_arr - self._baseline_ds_med, 0)
        else:
            db_corrected = db_arr
            ds_corrected = ds_arr

        # 4-dim scoring (using corrected deltas)
        scores = {}

        if self._baseline_ready:
            # Use baseline MAD for jump detection (more stable than window MAD)
            db_jump_k = max(5, int(self._baseline_db_mad * 30))
            ds_jump_k = max(5, int(self._baseline_ds_mad * 30))
            scores['bright_jump'] = self._jump_score(db_corrected, outlier_k=db_jump_k,
                                                     min_abs=self.jump_min_abs_brightness)
            scores['sat_jump'] = self._jump_score(ds_corrected, outlier_k=ds_jump_k,
                                                  min_abs=self.jump_min_abs_sat)
        else:
            scores['bright_jump'] = self._jump_score(db_arr, outlier_k=5,
                                                     min_abs=self.jump_min_abs_brightness)
            scores['sat_jump'] = self._jump_score(ds_arr, outlier_k=5,
                                                  min_abs=self.jump_min_abs_sat)

        # v3.3: high-pass the series before variance scoring — a moving
        # average (~0.8s) tracks slow content/AE ramps; only the fast
        # residual oscillation (real flicker frequency band) is scored.
        def _highpass(arr):
            k = max(3, (int(self.fps / self.sample_step * 0.8) // 2) * 2 + 1)
            if len(arr) >= k:
                # edge-pad before convolving: zero-padding ('same' mode) would
                # fake huge residuals at both ends of the rolling window —
                # and the newest frame is always at the boundary
                padded = np.pad(arr, k // 2, mode="edge")
                trend = np.convolve(padded, np.ones(k) / k, mode="valid")
                return arr - trend
            return arr - arr.mean()

        fb_arr = _highpass(np.array(self.frame_brightness))
        scores['bright_var'] = self._variance_score(fb_arr,
            normal_std=0.10, suspect_std=0.20, high_std=0.32)

        fs_arr = _highpass(np.array(self.frame_saturation))
        scores['sat_var'] = self._variance_score(fs_arr,
            normal_std=0.50, suspect_std=1.00, high_std=1.80)

        # -- v3.3: gate variance dims by window motion level --
        # a continuously moving scene inflates global mean/sat std; if the
        # window is dominated by content motion, variance evidence is weak
        if self._motion_factors:
            var_gate = float(np.median(self._motion_factors))
            scores['bright_var'] *= var_gate
            scores['sat_var'] *= var_gate

        # Weighted fusion
        weights = {
            'bright_jump': 0.28,
            'sat_jump':    0.28,
            'bright_var':  0.24,
            'sat_var':     0.20,
        }
        raw_conf = sum(scores[k] * weights[k] for k in weights)
        max_score = max(scores.values())

        # Penetration rules (v3.2: reduced amplitude, higher entry)
        # 1. Strong brightness jump
        if scores['bright_jump'] > 0.75:
            raw_conf = min(raw_conf + 0.10, 1.0)
        elif scores['bright_jump'] > 0.55:
            raw_conf = min(raw_conf + 0.05, 1.0)

        # 2. Strong saturation jump
        if scores['sat_jump'] > 0.70:
            raw_conf = min(raw_conf + 0.08, 1.0)
        elif scores['sat_jump'] > 0.50:
            raw_conf = min(raw_conf + 0.04, 1.0)

        # 3. Any dim extremely strong
        if max_score > 0.80:
            raw_conf = min(raw_conf + 0.06, 1.0)
        elif max_score > 0.60:
            raw_conf = min(raw_conf + 0.03, 1.0)

        # 4. Multi-dim combo
        active_dims = sum(1 for v in scores.values() if v > 0.18)
        if active_dims >= 3:
            raw_conf = min(raw_conf + 0.03, 1.0)
        if active_dims >= 2 and max_score > 0.35:
            raw_conf = min(raw_conf + 0.02, 1.0)

        # 5. Jump + variance dual confirmation
        if scores['bright_var'] > 0.35 and scores['bright_jump'] > 0.25:
            raw_conf = min(raw_conf + 0.02, 1.0)
        if scores['sat_var'] > 0.35 and scores['sat_jump'] > 0.25:
            raw_conf = min(raw_conf + 0.02, 1.0)

        raw_conf = min(raw_conf, 1.0)

        # EMA smoothing (lower alpha for more damping on live camera)
        self._ema_conf = self.ema_alpha * raw_conf + (1 - self.ema_alpha) * self._ema_conf
        smooth_conf = self._ema_conf

        # Debounce: gradual decay
        if smooth_conf >= self.threshold:
            self._above_thresh_count += 1
        else:
            self._above_thresh_count = max(0, self._above_thresh_count - 1)

        is_flick_raw = self._above_thresh_count >= self.min_trigger

        # -- Burst filter: require clustered flicker events --
        # Flicker event counting
        if is_flick_raw and self._cooldown == 0:
            self.flicker_events.append((1, self._frame_counter))
            self._cooldown = max(6, int(self.fps / self.sample_step * 0.4))
        else:
            self.flicker_events.append((0, self._frame_counter))

        # Check burst pattern: >= 2 events within 0.5s window
        flick_count_evt = sum(e[0] for e in self.flicker_events)
        is_flick = False
        if is_flick_raw and flick_count_evt >= 1:
            # Check if there's a burst cluster
            recent_events = [(v, t) for v, t in self.flicker_events if v == 1]
            if len(recent_events) >= 2:
                # Check if at least 2 events within 0.8s
                # (event timestamps are RAW frame counters, so the span must
                #  be in raw-frame units; v3.2 used sampled units — a bug that
                #  made bursts nearly impossible to satisfy. 0.8s because the
                #  cooldown itself enforces >=0.6s between events.)
                frame_span = 0.8 * self.fps
                for i in range(len(recent_events) - 1):
                    t1 = recent_events[i][1]
                    t2 = recent_events[i + 1][1]
                    if t2 - t1 <= frame_span:
                        is_flick = True
                        break
            elif len(recent_events) == 1 and smooth_conf > 0.85:
                # Single very strong event can pass
                is_flick = True
            else:
                is_flick = False  # Isolated spike, likely noise

        effective_fps = self.fps / self.sample_step
        window_dur = len(self.flicker_events) / max(effective_fps, 1)
        flick_rate = flick_count_evt / max(window_dur, 0.1)

        details = {k: round(v, 4) for k, v in scores.items()}
        details['struct_corr'] = round(struct_corr, 4)
        details['motion_factor'] = motion_factor
        details['db_local'] = round(db_local, 3)

        return FlickerResult(
            is_flickering=is_flick,
            confidence=round(smooth_conf, 4),
            raw_confidence=round(raw_conf, 4),
            anomaly_ratio=round(max_score, 4),
            details=details,
            flicker_count=flick_count_evt,
            flicker_rate=round(flick_rate, 2),
            baseline_ready=self._baseline_ready,
        )

    # Core: jump scoring (v3.2: baseline-corrected input)

    @staticmethod
    def _sig_ramp(value, floor):
        """Absolute significance: 0 below floor, 1 at 2*floor, linear ramp between."""
        if floor <= 0:
            return 1.0
        return float(np.clip((value - floor) / floor, 0.0, 1.0))

    @staticmethod
    def _jump_score(diff_arr, outlier_k=5, min_abs=0.0):
        """
        v3.2: expects baseline-corrected input (baseline median subtracted).
        median should be ~0 for corrected data, so outlier detection
        is more sensitive to real spikes.
        v3.3: min_abs adds an absolute magnitude floor — with a near-zero
        median, MAD ratios explode on sub-gray-level sensor noise, so ratio
        evidence only counts once the delta is physically significant.
        """
        if len(diff_arr) < 12:
            return 0.0

        med = np.median(diff_arr)
        mad = np.median(np.abs(diff_arr - med)) + 1e-7

        current = diff_arr[-1]
        ratio = max(0, (current - med) / mad)

        cur_sig = FlickerDetector._sig_ramp(current, min_abs)
        max_sig = FlickerDetector._sig_ramp(float(diff_arr.max()), min_abs)

        outlier_thresh = max(med + outlier_k * mad, min_abs)
        outlier_ratio = float((diff_arr > outlier_thresh).sum() / len(diff_arr))
        max_ratio_inner = float(diff_arr.max() / max(med + 1e-6, 1e-6))

        # s1: current frame MAD deviation ratio
        if ratio > outlier_k + 5:
            s1 = 1.0
        elif ratio > outlier_k + 2:
            s1 = 0.55 + (ratio - outlier_k - 2) / 7.3
        elif ratio > outlier_k:
            s1 = 0.25 + (ratio - outlier_k) / 8
        elif ratio > outlier_k - 1:
            s1 = 0.08 + (ratio - outlier_k + 1) / 10
        else:
            s1 = max(0, ratio / max(outlier_k - 1, 1) * 0.05)

        # s2: outlier frame ratio in window
        if outlier_ratio > 0.05:
            s2 = 1.0
        elif outlier_ratio > 0.02:
            s2 = 0.5 + (outlier_ratio - 0.02) / 0.06
        elif outlier_ratio > 0.008:
            s2 = 0.15 + (outlier_ratio - 0.008) / 0.047
        elif outlier_ratio > 0.002:
            s2 = 0.04 + (outlier_ratio - 0.002) / 0.05
        else:
            s2 = 0.0

        # s3: max frame diff ratio (strongest discriminator)
        if max_ratio_inner > 20:
            s3 = 1.0
        elif max_ratio_inner > 10:
            s3 = 0.6 + (max_ratio_inner - 10) / 25
        elif max_ratio_inner > 5:
            s3 = 0.3 + (max_ratio_inner - 5) / 16.7
        elif max_ratio_inner > 3:
            s3 = 0.1 + (max_ratio_inner - 3) / 20
        else:
            s3 = 0.0

        # v3.3: apply absolute-significance ramps
        s1 *= cur_sig
        s3 *= max_sig

        return min(0.15 * s1 + 0.30 * s2 + 0.55 * s3, 1.0)

    # Core: temporal variance scoring (v3.2: +25% looser than v3.1)

    @staticmethod
    def _variance_score(arr, normal_std=0.10, suspect_std=0.20, high_std=0.32):
        """
        v3.2: ~25% looser than v3.1 (camera baseline variance is higher)
        """
        if len(arr) < 12:
            return 0.0
        std = float(np.std(arr))
        if std > high_std:
            s = 1.0
        elif std > suspect_std:
            s = 0.5 + (std - suspect_std) / (high_std - suspect_std) * 0.5
        elif std > normal_std:
            s = 0.08 + (std - normal_std) / (suspect_std - normal_std) * 0.32
        else:
            s = max(0, std / normal_std * 0.06)
        return min(s, 1.0)

    # Video batch detection

    @staticmethod
    def detect_video(video_path, threshold=0.70, sample_step=2):
        """Batch detection, returns (result_list, fps)"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = sample_step if sample_step > 0 else (3 if fps > 25 else 2)
        effective_fps = fps / step
        det = FlickerDetector(
            window_size=int(effective_fps * 4),
            threshold=threshold,
            fps=fps,
            sample_step=step,
            min_trigger=2,
            ema_alpha=0.25,
        )

        results = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            r = det.push_frame(frame)
            if idx % step == 0:
                results.append((idx, r))
            idx += 1
        cap.release()
        return results, fps


# -- CLI --

if __name__ == "__main__":
    import sys, os

    if len(sys.argv) < 2:
        print("Usage: python flicker_detector.py <video_path> [threshold] [sample_step]")
        print("  threshold default 0.70, sample_step default auto")
        sys.exit(1)

    path = sys.argv[1]
    thresh = float(sys.argv[2]) if len(sys.argv) > 2 else 0.70
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)

    hdr_len = 90
    print("=" * hdr_len)
    print("  Flicker Detection Result (v3.2 - Live Camera Adaptive)")
    print("=" * hdr_len)
    print(f"  Video: {os.path.basename(path)}")
    print(f"  Threshold: {thresh}  Sample: auto  Debounce: 3 frames  EMA: 0.25")

    results, fps = FlickerDetector.detect_video(path, threshold=thresh, sample_step=step)

    total = len(results)
    flick_count = sum(1 for _, r in results if r.is_flickering)
    flick_frames = [(f, r) for f, r in results if r.is_flickering]

    sep = "-" * hdr_len
    hdr = (f"{'Frame':>8s} {'Conf':>7s} {'Raw':>7s} {'Base':>4s} | "
           f"{'BJ':>5s} {'SJ':>5s} {'BV':>5s} {'SV':>5s} | "
           f"{'Cnt':>4s} {'Rate':>7s}")
    print(sep)
    print(hdr)
    print(sep)

    last_printed = -999
    for f, r in flick_frames:
        if f - last_printed < (step or 2) * 3:
            continue
        d = r.details
        bl = "Y" if r.baseline_ready else "N"
        print(f"{f:>8d} {r.confidence:>6.1%} {r.raw_confidence:>6.1%} {bl:>4s} | "
              f"{d.get('bright_jump',0):>4.0%} {d.get('sat_jump',0):>4.0%} "
              f"{d.get('bright_var',0):>4.0%} {d.get('sat_var',0):>4.0%} | "
              f"{r.flicker_count:>4d} {r.flicker_rate:>5.1f}/s")
        last_printed = f

    if flick_frames:
        peak = max(flick_frames, key=lambda x: x[1].confidence)
        print(sep)
        print(f"  [PEAK] Frame #{peak[0]} conf={peak[1].confidence:.1%} raw={peak[1].raw_confidence:.1%}")

    # Print top5 raw confidence frames
    all_by_raw = sorted(results, key=lambda x: x[1].raw_confidence, reverse=True)[:5]
    if all_by_raw and all_by_raw[0][1].raw_confidence > 0.10:
        print(sep)
        print("  [Top5] Highest raw confidence frames (incl. below threshold):")
        for f, r in all_by_raw:
            d = r.details
            flag = " <<<" if r.is_flickering else ""
            bl = "Y" if r.baseline_ready else "N"
            print(f"    #{f:>6d} raw={r.raw_confidence:>5.1%} smooth={r.confidence:>5.1%} base={bl} "
                  f"BJ={d.get('bright_jump',0):.0%} SJ={d.get('sat_jump',0):.0%} "
                  f"BV={d.get('bright_var',0):.0%} SV={d.get('sat_var',0):.0%}{flag}")

    # Print baseline stats if available
    baseline_frames = [r for _, r in results if r.baseline_ready]
    if baseline_frames:
        first_bl = baseline_frames[0]
        print(sep)
        print(f"  [BASELINE] Calibrated after warmup (2s)")

    print(sep)
    verdict = "[WARN] Flicker detected" if flick_count > 0 else "[OK] No flicker detected"
    print(f"  {verdict}: {flick_count}/{total} sampled frames ({flick_count/max(total,1)*100:.1f}%)")
    print("=" * hdr_len)
