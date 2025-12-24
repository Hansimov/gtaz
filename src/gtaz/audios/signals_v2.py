"""V2 音频模板匹配（meta-template + 二阶段验证）。"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf
from numpy.fft import irfft, rfft
from tclogger import TCLogger

from .sounds import AUDIO_DEVICE_NAME, SAMPLE_RATE, WINDOW_MS, SoundRecorder


logger = TCLogger(
    name="TemplateMatcherV2",
    use_prefix=False,
    use_file=True,
    file_path=Path(__file__).parent / "output.log",
    file_mode="w",
)


MODULE_DIR = Path(__file__).parent
CACHE_DIR = MODULE_DIR.parent / "cache"
TEMPLATES_DIR = MODULE_DIR / "wavs"
TEMPLATE_PREFIX = "template_"
TEST_SOUNDS_DIR = CACHE_DIR / "sounds"

MATCH_THRESHOLD = 0.82

META_COARSE_THRESHOLD = 0.46

VERIFY_THRESHOLD = 0.64

REFINE_RADIUS_FRAMES = 18

NMS_WINDOW_S = 1.2
REALTIME_SAMPLE_RATE = 16000


@dataclass
class AudioTemplate:
    name: str
    data: np.ndarray
    sample_rate: int
    file_path: Path | None = None

    @property
    def duration_ms(self) -> float:
        return len(self.data) / self.sample_rate * 1000

    def to_mono(self) -> np.ndarray:
        if self.data.ndim == 1:
            return self.data
        return np.mean(self.data, axis=1)


class TemplateLoader:
    def __init__(self, templates_dir: Path = TEMPLATES_DIR):
        self.templates_dir = Path(templates_dir)
        self._templates: dict[str, AudioTemplate] = {}

    def load_file(
        self, file_path: Path, name: str | None = None
    ) -> AudioTemplate | None:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warn(f"模板文件不存在: {file_path}")
            return None
        try:
            data, sample_rate = sf.read(file_path)
            if name is None:
                name = file_path.stem
            tmpl = AudioTemplate(
                name=name, data=data, sample_rate=sample_rate, file_path=file_path
            )
            self._templates[name] = tmpl
            return tmpl
        except Exception as e:
            logger.warn(f"加载模板文件失败 {file_path}: {e}")
            return None

    def load_directory(
        self,
        dir_path: Path | None = None,
        extensions: list[str] | None = None,
        prefix: str = TEMPLATE_PREFIX,
    ) -> int:
        dir_path = Path(dir_path or self.templates_dir)
        exts = set(extensions or [".wav", ".flac", ".ogg", ".mp3"])
        if not dir_path.exists():
            logger.warn(f"模板目录不存在: {dir_path}")
            return 0

        count = sum(
            1
            for p in dir_path.iterdir()
            if p.is_file()
            and p.suffix.lower() in exts
            and (not prefix or p.name.startswith(prefix))
            and self.load_file(p)
        )
        logger.note(f"从目录 {dir_path} 加载了 {count} 个模板")
        return int(count)

    def get_all_templates(self) -> list[AudioTemplate]:
        return list(self._templates.values())


class FeaturesExtractorV2:
    """Resample + envelopes."""

    def __init__(
        self, target_sample_rate: int = REALTIME_SAMPLE_RATE, hop_length: int = 256
    ):
        self.target_sample_rate = int(target_sample_rate)
        self.hop_length = int(hop_length)

    def preprocess(
        self, audio: np.ndarray, sample_rate: int, to_mono: bool = True
    ) -> np.ndarray:
        x = audio
        if to_mono and x.ndim > 1:
            x = np.mean(x, axis=1)
        x = x.astype(np.float32, copy=False)

        if int(sample_rate) != self.target_sample_rate:
            # resample_poly is stable + fast
            from scipy import signal as scipy_signal

            g = np.gcd(int(sample_rate), self.target_sample_rate)
            up = self.target_sample_rate // g
            down = int(sample_rate) // g
            x = scipy_signal.resample_poly(x, up=up, down=down).astype(
                np.float32, copy=False
            )

        # Remove DC and normalize amplitude
        x = x - float(np.mean(x))
        mx = float(np.max(np.abs(x)))
        if mx > 0:
            x = x / mx
        return x

    def compute_energy_envelope(self, audio_16k: np.ndarray) -> np.ndarray:
        """RMS envelope with hop_length (no overlap needed)."""
        hop = self.hop_length
        n = len(audio_16k)
        if n < hop:
            return np.array([], dtype=np.float32)
        n_frames = n // hop
        x = audio_16k[: n_frames * hop]
        frames = x.reshape(n_frames, hop)
        env = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12).astype(np.float32)
        # z-score normalize (robust against volume)
        m = float(np.mean(env))
        s = float(np.std(env))
        if s > 1e-8:
            env = (env - m) / s
        else:
            env = env - m
        return env

    def compute_event_envelope(
        self,
        audio_16k: np.ndarray,
        band: tuple[float, float] = (700.0, 3200.0),
        smooth_ms: float = 18.0,
    ) -> np.ndarray:
        """Pulse-oriented envelope.

        Bandpass -> abs (rectify) -> lowpass smoothing, then downsample by hop_length.
        This tends to be more discriminative for short transient sounds.
        """
        x = audio_16k.astype(np.float32, copy=False)
        if x.size == 0:
            return np.array([], dtype=np.float32)

        from scipy import signal as scipy_signal

        sr = float(self.target_sample_rate)
        lo, hi = float(band[0]), float(band[1])
        hi = min(hi, 0.49 * sr)
        lo = max(10.0, min(lo, hi * 0.9))

        # 4th-order Butterworth bandpass (stable for our use)
        sos = scipy_signal.butter(
            4, [lo, hi], btype="bandpass", fs=self.target_sample_rate, output="sos"
        )
        y = scipy_signal.sosfilt(sos, x)

        y = np.abs(y).astype(np.float32, copy=False)

        # Smooth by moving average (lowpass on rectified signal)
        win = max(1, int((smooth_ms / 1000.0) * self.target_sample_rate))
        if win > 1:
            kernel = np.ones(win, dtype=np.float32) / float(win)
            y = np.convolve(y, kernel, mode="same").astype(np.float32, copy=False)

        # Downsample to envelope frames
        hop = self.hop_length
        n = int(y.size)
        if n < hop:
            return np.array([], dtype=np.float32)
        n_frames = n // hop
        y = y[: n_frames * hop]
        frames = y.reshape(n_frames, hop)
        env = np.mean(frames, axis=1).astype(np.float32)

        # log-compress to stabilize impulsive peaks
        env = np.log1p(env).astype(np.float32, copy=False)

        # robust normalize (median/MAD)
        med = float(np.median(env))
        mad = float(np.median(np.abs(env - med)))
        scale = mad * 1.4826
        if scale > 1e-8:
            env = (env - med) / scale
        else:
            env = env - med
        return env

    def compute_event_envelopes_multiband(
        self,
        audio_16k: np.ndarray,
        smooth_ms: float = 18.0,
    ) -> list[np.ndarray]:
        """Multi-band transient envelopes for short, broadband impulses."""
        bands: list[tuple[float, float]] = [
            (200.0, 6000.0),
            (200.0, 800.0),
            (800.0, 2400.0),
            (2400.0, 6000.0),
        ]
        envs = [
            self.compute_event_envelope(audio_16k, band=b, smooth_ms=smooth_ms)
            for b in bands
        ]
        return [e for e in envs if e.size > 0]


@dataclass
class MatchResult:
    template_name: str
    score: float
    position: int = 0  # samples @ target SR
    sample_rate: int = REALTIME_SAMPLE_RATE
    template_duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        sr = int(self.sample_rate)
        if sr <= 0:
            return {
                "template_name": self.template_name,
                "score": round(float(self.score), 4),
                "position": int(self.position),
                "start_time": "N/A",
                "end_time": "N/A",
                "duration": "N/A",
            }

        start_seconds = self.position / sr
        duration_seconds = self.template_duration_ms / 1000.0
        return {
            "template_name": self.template_name,
            "score": round(float(self.score), 4),
            "position": int(self.position),
            "start_time": f"{start_seconds:.3f}",
            "end_time": f"{(start_seconds + duration_seconds):.3f}",
            "duration": f"{duration_seconds:.3f}",
        }


@dataclass
class MetaTemplate:
    """Fused meta-template in envelope domain."""

    name: str
    env: np.ndarray  # (T,)
    hop_length: int
    sample_rate: int
    duration_ms: float
    aligned_lags: dict[str, int]


class MetaTemplateMatcher:
    """V2 core: align templates -> fuse into meta-template -> detect via envelope NCC."""

    def __init__(
        self,
        templates: list[AudioTemplate] | None = None,
        threshold: float = MATCH_THRESHOLD,
        target_sample_rate: int = REALTIME_SAMPLE_RATE,
        hop_length: int = 256,
    ):
        self.threshold = float(threshold)
        self.target_sample_rate = int(target_sample_rate)
        self.hop_length = int(hop_length)

        self.features = FeaturesExtractorV2(
            target_sample_rate=self.target_sample_rate, hop_length=self.hop_length
        )

        self._templates: dict[str, AudioTemplate] = {}
        self._template_envs: dict[str, np.ndarray] = {}
        self._template_event_envs: dict[str, np.ndarray] = {}
        self._template_event_envs_mb: dict[str, list[np.ndarray]] = {}

        self.meta: MetaTemplate | None = None
        self.meta_threshold: float | None = None

        if templates:
            for t in templates:
                self.add_template(t)

        # Stage-2 uses original template envelopes for verification
        self.verify_threshold: float = float(VERIFY_THRESHOLD)

    def add_template(self, template: AudioTemplate) -> None:
        x = self.features.preprocess(
            template.to_mono(), template.sample_rate, to_mono=False
        )
        env = self.features.compute_energy_envelope(x)
        ev = self.features.compute_event_envelope(x)
        ev_mb = self.features.compute_event_envelopes_multiband(x)
        self._templates[template.name] = template
        self._template_envs[template.name] = env
        self._template_event_envs[template.name] = ev
        self._template_event_envs_mb[template.name] = ev_mb
        logger.note(f"- 已添加模板: {template.name} (env_frames={len(env)})")

    @staticmethod
    def _aggregate_mb_scores(scores_list: list[np.ndarray]) -> np.ndarray:
        if not scores_list:
            return np.array([], dtype=np.float32)
        m = min((int(s.size) for s in scores_list), default=0)
        if m <= 0:
            return np.array([], dtype=np.float32)
        stacked = np.stack([s[:m] for s in scores_list], axis=0)
        return np.max(stacked, axis=0).astype(np.float32)

    def get_peak_threshold(self, peak_threshold: float | None) -> float | None:
        if peak_threshold is not None:
            return float(peak_threshold)
        if self.meta_threshold is not None:
            return float(self.meta_threshold)
        return None

    def _coarse_threshold(self, peak_threshold: float | None) -> float:
        thr_override = self.get_peak_threshold(peak_threshold)
        return (
            float(thr_override)
            if thr_override is not None
            else float(min(self.threshold, META_COARSE_THRESHOLD))
        )

    def _verify_and_refine(
        self,
        event_env_seq: np.ndarray,
        coarse_peak: int,
        search_radius: int,
        max_templates: int | None = None,
    ) -> tuple[int, str, float]:
        """Returns (best_peak_frame, best_template_name, best_score)."""
        if not self._template_envs:
            return int(coarse_peak), (self.meta.name if self.meta else "meta"), 0.0

        # We need a local window big enough to slide full templates.
        # Use event-envelope template lengths for verification
        max_tlen = max(
            (int(v.size) for v in self._template_event_envs.values()), default=0
        )
        half = max(int(search_radius), int(max_tlen // 2) + 2)

        left = max(0, int(coarse_peak) - half)
        right = min(int(event_env_seq.size), int(coarse_peak) + half + 1)
        if right <= left:
            return int(coarse_peak), (self.meta.name if self.meta else "meta"), 0.0

        # Evaluate each template by NCC over the local region and keep the best.
        best_score = -1.0
        best_tname = ""
        best_pos = int(coarse_peak)

        items = list(self._template_event_envs.items())
        if max_templates is not None:
            items = items[: int(max_templates)]

        for tname, t_ev in items:
            mb = self._template_event_envs_mb.get(tname)
            if mb:
                local_seq = event_env_seq[left:right]
                best_local = -1.0
                best_local_k = 0
                for b_ev in mb:
                    if b_ev.size == 0:
                        continue
                    if local_seq.size < int(b_ev.size):
                        continue
                    scores = self._fft_ncc_1d(local_seq, b_ev)
                    if scores.size == 0:
                        continue
                    k = int(np.argmax(scores))
                    s = float(scores[k])
                    if s > best_local:
                        best_local = s
                        best_local_k = k
                if best_local > best_score:
                    best_score = best_local
                    best_tname = tname
                    best_pos = int(left + best_local_k)
                continue

            if t_ev.size == 0:
                continue

            local_seq = event_env_seq[left:right]
            # Need enough samples for valid NCC
            if local_seq.size < int(t_ev.size):
                continue
            scores = self._fft_ncc_1d(local_seq, t_ev)
            if scores.size == 0:
                continue
            k = int(np.argmax(scores))
            s = float(scores[k])
            if s > best_score:
                best_score = s
                best_tname = tname
                best_pos = int(left + k)

        return best_pos, best_tname, float(best_score)

    @staticmethod
    def _nms_by_time(hits: list[MatchResult], window_s: float) -> list[MatchResult]:
        if not hits:
            return []
        hits_sorted = sorted(hits, key=lambda r: r.score, reverse=True)
        keep: list[MatchResult] = []
        for h in hits_sorted:
            ts = h.position / max(1, h.sample_rate)
            if all(
                abs(ts - (k.position / max(1, k.sample_rate))) >= window_s for k in keep
            ):
                keep.append(h)
        return sorted(keep, key=lambda r: r.position)

    @property
    def template_count(self) -> int:
        return len(self._templates)

    @staticmethod
    def _fft_ncc_1d(
        sequence: np.ndarray, template: np.ndarray, min_energy_ratio: float = 0.10
    ) -> np.ndarray:
        """Compute NCC scores for all valid positions (sequence vs template)."""
        if sequence.size == 0 or template.size == 0:
            return np.array([], dtype=np.float32)
        if sequence.size < template.size:
            return np.array([], dtype=np.float32)

        seq = sequence.astype(np.float32, copy=False)
        tmp = template.astype(np.float32, copy=False)
        n = seq.size
        m = tmp.size

        tmp_energy = float(np.sum(tmp * tmp))
        if tmp_energy <= 0:
            return np.zeros(n - m + 1, dtype=np.float32)

        fft_len = 1
        while fft_len < n + m - 1:
            fft_len *= 2

        tmp_fft = rfft(tmp[::-1], fft_len)
        seq_fft = rfft(seq, fft_len)
        corr = irfft(seq_fft * tmp_fft, fft_len)
        corr = corr[m - 1 : m - 1 + (n - m + 1)]

        seq_sq = seq * seq
        cumsum_sq = np.concatenate(([0.0], np.cumsum(seq_sq)))
        win_sq = cumsum_sq[m:] - cumsum_sq[:-m]

        cumsum_val = np.concatenate(([0.0], np.cumsum(seq)))
        win_sum = cumsum_val[m:] - cumsum_val[:-m]
        win_mean = win_sum / m
        win_var_energy = win_sq - m * (win_mean * win_mean)
        win_var_energy = np.maximum(win_var_energy, 0.0)

        denom = np.sqrt(win_var_energy * tmp_energy)
        valid_mask = win_var_energy >= (min_energy_ratio * tmp_energy)
        safe = np.where(denom > 0, denom, 1.0)
        scores = np.where(valid_mask, corr / safe, -np.inf)
        scores = np.where(np.isfinite(scores), np.clip(scores, 0.0, 1.0), 0.0).astype(
            np.float32
        )
        return scores

    @staticmethod
    def _pick_peaks(
        scores: np.ndarray, min_height: float | None, min_distance: int
    ) -> np.ndarray:
        if scores.size < 3:
            return np.array([], dtype=np.int64)
        mid = scores[1:-1]
        if min_height is None:
            mask = (scores[:-2] < mid) & (mid >= scores[2:])
        else:
            mask = (
                (scores[:-2] < mid) & (mid >= scores[2:]) & (mid >= float(min_height))
            )
        idx = np.nonzero(mask)[0] + 1
        if idx.size == 0:
            return idx.astype(np.int64)
        if min_distance <= 1:
            return np.sort(idx).astype(np.int64)

        order = np.argsort(scores[idx])[::-1]
        chosen: list[int] = []
        for k in order:
            p = int(idx[k])
            if all(abs(p - c) >= min_distance for c in chosen):
                chosen.append(p)
        return np.array(sorted(chosen), dtype=np.int64)

    def build_meta_template(self, name: str = "meta_template") -> MetaTemplate:
        """Align event envelopes to an anchor and fuse via median."""
        if not self._template_event_envs:
            raise ValueError("No templates loaded")

        def z(x: np.ndarray) -> np.ndarray:
            x = x.astype(np.float32, copy=False)
            m = float(np.mean(x))
            s = float(np.std(x))
            if s > 1e-8:
                return (x - m) / s
            return x - m

        def best_lag_full(a: np.ndarray, b: np.ndarray) -> int:
            """lag > 0 means b starts after a."""
            aa = z(a)
            bb = z(b)
            # full correlation via FFT: corr[k] corresponds to lag = k-(len(b)-1)
            n = aa.size
            m = bb.size
            if n == 0 or m == 0:
                return 0
            fft_len = 1
            while fft_len < n + m - 1:
                fft_len *= 2
            A = rfft(aa, fft_len)
            B = rfft(bb, fft_len)
            corr = irfft(A * np.conj(B), fft_len)[: n + m - 1]
            k = int(np.argmax(corr))
            return int(k - (m - 1))

        # Choose anchor by median length (stable)
        names = list(self._template_event_envs.keys())
        lengths = [(n, int(self._template_event_envs[n].size)) for n in names]
        lengths.sort(key=lambda x: x[1])
        anchor_name = lengths[len(lengths) // 2][0]
        anchor = z(self._template_event_envs[anchor_name])

        # Collect lags relative to anchor, then fuse in anchor coordinates.
        lags: dict[str, int] = {}
        lags[anchor_name] = 0

        for n in names:
            if n == anchor_name:
                continue
            ev = self._template_event_envs[n]
            lags[n] = best_lag_full(anchor, ev)

        aligned: list[np.ndarray] = []
        for n in names:
            env = z(self._template_event_envs[n])
            lag = int(lags[n])
            start_in_env = max(0, -lag)
            start_in_anchor = max(0, lag)
            take = min(int(anchor.size) - start_in_anchor, int(env.size) - start_in_env)
            buf = np.zeros(int(anchor.size), dtype=np.float32)
            if take > 0:
                buf[start_in_anchor : start_in_anchor + take] = env[
                    start_in_env : start_in_env + take
                ]
            aligned.append(buf)

        # Stack and fuse via median (robust)
        stack = np.stack(aligned, axis=0).astype(np.float32)
        fused = np.median(stack, axis=0).astype(np.float32)

        # Normalize fused again
        m = float(np.mean(fused))
        s = float(np.std(fused))
        if s > 1e-8:
            fused = (fused - m) / s
        else:
            fused = fused - m

        duration_ms = (fused.size * self.hop_length) / self.target_sample_rate * 1000.0

        meta = MetaTemplate(
            name=name,
            env=fused,
            hop_length=self.hop_length,
            sample_rate=self.target_sample_rate,
            duration_ms=duration_ms,
            aligned_lags=lags,
        )
        self.meta = meta
        return meta

    def detect_with_meta(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        peak_threshold: float | None = None,
        peak_min_distance_ratio: float = 0.85,
        max_peaks: int = 20,
        debug: bool = False,
    ) -> list[MatchResult]:
        if self.meta is None:
            self.build_meta_template()
        assert self.meta is not None

        x = self.features.preprocess(audio_data, sample_rate, to_mono=True)
        event_envs = self.features.compute_event_envelopes_multiband(x)
        if not event_envs:
            return []
        event_env = event_envs[0]
        if event_env.size < self.meta.env.size:
            return []

        # Coarse: aggregate max NCC across bands (broadband + sub-bands).
        scores = self._aggregate_mb_scores(
            [self._fft_ncc_1d(e, self.meta.env) for e in event_envs]
        )
        coarse_max = float(np.max(scores)) if scores.size else float("-inf")

        min_dist = max(1, int(self.meta.env.size * peak_min_distance_ratio))

        # Coarse candidates: rank-based (top-K local maxima) instead of fixed threshold.
        # NCC absolute scale can drift; ranking is stable.
        peaks_all = self._pick_peaks(scores, min_height=None, min_distance=min_dist)
        if peaks_all.size == 0:
            if debug:
                logger.note(f"  coarse: max={coarse_max:.3f} peaks=0")
            return []

        order = np.argsort(scores[peaks_all])[::-1]
        peaks_k = peaks_all[order[:max_peaks]]

        # Relative prominence gate: keep peaks that stand out from the rest.
        top_scores = scores[peaks_k]
        med = float(np.median(scores))
        mad = float(np.median(np.abs(scores - med))) * 1.4826
        rel_thr = med + 4.0 * (mad if mad > 1e-8 else 0.0)
        peaks_k = peaks_k[top_scores >= rel_thr]
        peaks = np.sort(peaks_k)

        if debug:
            logger.note(
                f"  coarse: max={coarse_max:.3f} cand={int(peaks_all.size)} kept={int(peaks.size)}"
            )

        if peaks.size == 0:
            return []

        # Stage-2 verification & refinement
        verified: list[MatchResult] = []
        best_verify_seen = float("-inf")
        for p in peaks:
            best_pos, best_tname, best_s = self._verify_and_refine(
                event_env,
                coarse_peak=int(p),
                search_radius=int(REFINE_RADIUS_FRAMES),
            )
            best_verify_seen = max(best_verify_seen, float(best_s))
            if best_s < float(self.verify_threshold):
                continue

            pos_samples = int(best_pos) * self.hop_length
            verified.append(
                MatchResult(
                    template_name=best_tname,
                    score=float(best_s),
                    position=pos_samples,
                    sample_rate=self.target_sample_rate,
                    template_duration_ms=(
                        float(self._templates[best_tname].duration_ms)
                        if best_tname in self._templates
                        else self.meta.duration_ms
                    ),
                )
            )

        # NMS to reduce duplicates / false positives
        verified = self._nms_by_time(verified, window_s=float(NMS_WINDOW_S))
        if debug:
            logger.note(
                f"  verify: best={best_verify_seen:.3f} ver_th={float(self.verify_threshold):.3f} kept={len(verified)}"
            )
        return verified


def compute_iou(
    pred_start: float, pred_end: float, ref_start: float, ref_end: float
) -> float:
    """Relaxed IoU': intersection(pred, ref) / duration(pred)."""
    intersection_start = max(pred_start, ref_start)
    intersection_end = min(pred_end, ref_end)
    intersection = max(0.0, intersection_end - intersection_start)
    pred_duration = max(1e-9, pred_end - pred_start)
    return float(intersection / pred_duration)


class TemplateMatchTester:
    """refs + 模板 + 匹配 + 写 results.json。"""

    def __init__(
        self,
        templates_dir: Path | None = None,
        sounds_dir: Path | None = None,
        threshold: float = MATCH_THRESHOLD,
        prefix: str = TEMPLATE_PREFIX,
    ):
        self.templates_dir = Path(templates_dir) if templates_dir else TEMPLATES_DIR
        self.sounds_dir = Path(sounds_dir) if sounds_dir else TEST_SOUNDS_DIR
        self.threshold = float(threshold)
        self.prefix = prefix

        self.refs: dict[str, dict[str, Any]] = {}
        self.matcher: MetaTemplateMatcher | None = None

        # Optional overrides for thresholds
        self.meta_threshold: float | None = None
        self.verify_threshold: float | None = None

    def load_refs(self) -> None:
        refs_path = self.sounds_dir / "refs.json"
        if refs_path.exists():
            self.refs = json.loads(refs_path.read_text(encoding="utf-8"))
            logger.note(f"已加载参考时间数据: {refs_path}")
        else:
            logger.warn(f"未找到参考时间文件: {refs_path}")
            self.refs = {}

    def load_templates(self) -> int:
        logger.note(f"正在从 {self.templates_dir} 加载模板...")
        loader = TemplateLoader(self.templates_dir)
        count = loader.load_directory(prefix=self.prefix)
        if count == 0:
            return 0
        self.matcher = MetaTemplateMatcher(
            templates=loader.get_all_templates(), threshold=self.threshold
        )
        if self.meta_threshold is not None:
            self.matcher.meta_threshold = float(self.meta_threshold)
        if self.verify_threshold is not None:
            self.matcher.verify_threshold = float(self.verify_threshold)
        # build meta immediately so we can log lags
        meta = self.matcher.build_meta_template()
        logger.note(f"已构建融合模板: {meta.name} (len={len(meta.env)} frames)")
        return count

    def iter_test_files(self) -> list[Path]:
        exts = {".wav", ".flac", ".ogg", ".mp3"}
        if not self.sounds_dir.exists():
            return []
        out: list[Path] = []
        for item in sorted(self.sounds_dir.iterdir()):
            if item.is_dir():
                out.extend(
                    f
                    for f in sorted(item.iterdir())
                    if f.is_file() and f.suffix.lower() in exts
                )
            elif item.is_file() and item.suffix.lower() in exts:
                out.append(item)
        return out

    def run_file(self, test_file: Path, skip_seconds: float = 5.0) -> dict[str, Any]:
        assert self.matcher is not None
        data, sr = sf.read(test_file)

        skip_samples = int(skip_seconds * sr)
        if len(data) <= skip_samples:
            return {"file": test_file.name, "skipped": True, "reason": "too short"}
        data2 = data[skip_samples:]

        start_time = time.perf_counter()
        segs = self.matcher.detect_with_meta(
            data2,
            sample_rate=sr,
            peak_threshold=self.matcher.get_peak_threshold(None),
            debug=True,
        )
        match_time_ms = (time.perf_counter() - start_time) * 1000.0

        for r in segs:
            r.position += int(skip_seconds * r.sample_rate)

        segs_sorted = sorted(segs, key=lambda r: r.score, reverse=True)
        best = segs_sorted[0] if segs_sorted else None

        ref = self.refs.get(test_file.name)
        has_ref = isinstance(ref, dict) and "start_time" in ref and "end_time" in ref
        ref_start = float(ref["start_time"]) if has_ref else -1.0
        ref_end = float(ref["end_time"]) if has_ref else -1.0

        best_iou = 0.0
        if has_ref and segs_sorted:
            cand_iou = lambda c: compute_iou(
                float((d := c.to_dict())["start_time"]),
                float(d["end_time"]),
                ref_start,
                ref_end,
            )
            best = max(segs_sorted, key=cand_iou)
            best_iou = float(cand_iou(best))

        return {
            "file": test_file.name,
            "match_time_ms": round(match_time_ms, 2),
            "best_match": best.to_dict() if best else None,
            "all_matches": [r.to_dict() for r in segs_sorted],
            "reference": (
                {"start_time": ref_start, "end_time": ref_end} if has_ref else None
            ),
            "metrics": {"best_iou": round(best_iou, 4)} if has_ref else None,
        }

    def run(self) -> dict[str, Any]:
        self.load_refs()
        template_count = self.load_templates()
        if template_count == 0:
            logger.warn("没有找到任何模板文件")
            return {}

        test_files = self.iter_test_files()
        if not test_files:
            logger.warn(f"在 {self.sounds_dir} 中没有找到测试音频文件")
            return {}

        logger.note(f"找到 {len(test_files)} 个测试音频文件")

        all_results: dict[str, Any] = {}
        total_ms = 0.0

        for f in test_files:
            rel = f.relative_to(self.sounds_dir)
            try:
                res = self.run_file(f)
            except Exception as e:
                logger.warn(f"处理失败 {rel}: {e}")
                continue

            total_ms += float(res.get("match_time_ms", 0.0))
            all_results[f.name] = res

            segs = res.get("all_matches") or []
            ref = res.get("reference")
            best = res.get("best_match")
            if not segs:
                logger.okay(f"{rel}: ✓ 负例" if ref is None else f"{rel}: ⚠ 未命中")
                continue

            line = f"{rel}: n={len(segs)}"
            if best is not None:
                line += (
                    f" best={best['template_name']} {best['score']:.3f}"
                    f" [{best['start_time']}~{best['end_time']}]"
                )
            if ref is not None and best is not None:
                biou = float((res.get("metrics") or {}).get("best_iou", 0.0))
                line += f" IoU={biou:.3f} off={float(best['start_time'])-float(ref['start_time']):+.3f}s"
            logger.note(line)

        results_json_path = self.sounds_dir / "results.json"
        results_json_path.write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.okay(f"\n所有结果已合并保存到: {results_json_path}")

        info = {
            "模板数量": template_count,
            "测试文件数": len(all_results),
            "总匹配耗时": f"{total_ms:.1f}ms",
            "平均耗时": f"{(total_ms / max(1, len(all_results))):.1f}ms",
        }
        logger.note("测试统计: " + ", ".join([f"{k}={v}" for k, v in info.items()]))

        return all_results


class TemplateMatcherArgParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="GTAV 音频模板匹配器 (V2)")
        self._add_arguments()

    def _add_arguments(self):
        self.parser.add_argument(
            "--test",
            action="store_true",
            help="测试模式：使用缓存的测试音频文件测试模板匹配效果",
        )
        self.parser.add_argument(
            "-t",
            "--templates-dir",
            type=str,
            default=None,
            help="模板目录",
        )
        self.parser.add_argument(
            "-T",
            "--threshold",
            type=float,
            default=MATCH_THRESHOLD,
            help="匹配阈值",
        )

        self.parser.add_argument(
            "--meta-threshold",
            type=float,
            default=META_COARSE_THRESHOLD,
            help="meta 粗检阈值（召回）",
        )
        self.parser.add_argument(
            "--verify-threshold",
            type=float,
            default=VERIFY_THRESHOLD,
            help="二阶段精检阈值（精度）",
        )
        self.parser.add_argument(
            "-s",
            "--sounds-dir",
            type=str,
            default=None,
            help="测试音频目录",
        )
        self.parser.add_argument(
            "-p",
            "--prefix",
            type=str,
            default=TEMPLATE_PREFIX,
            help="模板文件前缀过滤",
        )

        self.parser.add_argument(
            "-d",
            "--duration",
            type=float,
            default=None,
            help="运行时长（秒），不指定则持续运行",
        )
        self.parser.add_argument(
            "-n",
            "--device-name",
            type=str,
            default=AUDIO_DEVICE_NAME,
            help="音频设备名称",
        )
        self.parser.add_argument(
            "-r",
            "--sample-rate",
            type=int,
            default=SAMPLE_RATE,
            help="采样率",
        )
        self.parser.add_argument(
            "-w",
            "--window-ms",
            type=int,
            default=WINDOW_MS,
            help="窗口时长（毫秒）",
        )
        self.parser.add_argument(
            "-i",
            "--interval-ms",
            type=float,
            default=100,
            help="匹配间隔（毫秒）",
        )

    def parse(self) -> argparse.Namespace:
        return self.parser.parse_args()


class RealtimeMatcher:
    """保持原有架构：SoundRecorder + Matcher。V2 先使用 meta-detector 触发。"""

    def __init__(
        self,
        recorder: SoundRecorder | None = None,
        matcher: MetaTemplateMatcher | None = None,
        match_interval_ms: float = 100,
    ):
        self.recorder = recorder or SoundRecorder()
        self.matcher = matcher
        self._sleep_s = float(match_interval_ms) / 1000.0

        self._is_running = False
        self._thread = None

    def load_templates(
        self, dir_path: Path | None = None, prefix: str = TEMPLATE_PREFIX
    ) -> int:
        loader = TemplateLoader(dir_path or TEMPLATES_DIR)
        count = loader.load_directory(prefix=prefix)

        if self.matcher is None:
            self.matcher = MetaTemplateMatcher(threshold=MATCH_THRESHOLD)

        for t in loader.get_all_templates():
            self.matcher.add_template(t)
        self.matcher.build_meta_template()
        return count

    def _loop(self):
        while self._is_running:
            window_data = self.recorder.buffer.get_window_data()
            if (
                window_data is not None
                and len(window_data) > 0
                and self.matcher is not None
            ):
                segs = self.matcher.detect_with_meta(
                    window_data,
                    sample_rate=self.recorder.sample_rate,
                    peak_threshold=self.matcher.get_peak_threshold(None),
                    max_peaks=1,
                )
                if segs:
                    r = segs[0]
                    logger.okay(
                        f"检测到音频触发: {r.template_name} score={r.score:.3f}"
                    )
            time.sleep(self._sleep_s)

    def start(self):
        if self._is_running:
            return
        if not self.recorder.is_streaming:
            if not self.recorder.start_stream():
                logger.warn("无法启动音频流")
                return
        self._is_running = True
        import threading

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.okay("实时匹配器已启动")

    def stop(self):
        if not self._is_running:
            return
        self._is_running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.recorder.stop_stream()

    def run(self, duration: float | None = None):
        self.start()
        try:
            deadline = None if duration is None else (time.time() + float(duration))
            logger.note(
                "持续模式：Ctrl+C 退出" if deadline is None else f"运行 {duration} 秒"
            )
            while deadline is None or time.time() < deadline:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.note("\n检测到 Ctrl+C，正在退出...")
        finally:
            self.stop()


def main():
    args = TemplateMatcherArgParser().parse()

    if args.test:
        tester = TemplateMatchTester(
            templates_dir=Path(args.templates_dir) if args.templates_dir else None,
            sounds_dir=Path(args.sounds_dir) if args.sounds_dir else None,
            threshold=args.threshold,
            prefix=args.prefix,
        )
        tester.meta_threshold = float(args.meta_threshold)
        tester.verify_threshold = float(args.verify_threshold)
        tester.run()
        return

    recorder = SoundRecorder(
        device_name=args.device_name,
        sample_rate=args.sample_rate,
        window_ms=args.window_ms,
    )
    matcher = MetaTemplateMatcher(
        threshold=args.threshold, target_sample_rate=REALTIME_SAMPLE_RATE
    )
    matcher.meta_threshold = float(args.meta_threshold)
    matcher.verify_threshold = float(args.verify_threshold)

    realtime = RealtimeMatcher(
        recorder=recorder, matcher=matcher, match_interval_ms=args.interval_ms
    )
    realtime.load_templates(
        Path(args.templates_dir) if args.templates_dir else TEMPLATES_DIR,
        prefix=args.prefix,
    )
    realtime.run(duration=args.duration)


if __name__ == "__main__":
    main()
