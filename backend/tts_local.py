"""Yerel Türkçe TTS: Coqui XTTS (GPU/CPU) + Piper yedek. Bulut yok."""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import threading
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger("uvicorn.error")

# Coqui XTTS CPML: yerel/non-commercial kullanım onayı (stdin prompt'u yok).
os.environ.setdefault("COQUI_TOS_AGREED", "1")

TTS_LANGUAGE = "tr"
XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
PIPER_VOICE = os.getenv("TTS_PIPER_VOICE", "tr_TR-dfki-medium")
PREFERRED_SPEAKERS = (
    "Tammie Ema",
    "Ana Florence",
    "Sofia Hellen",
    "Viktor Eka",
    "Damien Black",
)
XTTS_CHUNK = 220
PAUSE_SEC = 0.28

BACKEND_DIR = Path(__file__).resolve().parent
MODELS_DIR = BACKEND_DIR / "models" / "tts"
CACHE_DIR = BACKEND_DIR / "cache" / "tts"

_lock = threading.RLock()
_engine: "LocalTTS | None" = None


@dataclass
class SpeakResult:
    wav_bytes: bytes
    engine: str
    language: str
    sample_rate: int
    cached: bool


def _device() -> str:
    forced = os.getenv("TTS_DEVICE", "").strip().lower()
    if forced in {"cuda", "cpu"}:
        return forced
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _float_to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    samples = np.asarray(audio, dtype=np.float32).flatten()
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm.tobytes())
    return buf.getvalue()


def _concat_wavs(chunks: list[np.ndarray], sample_rate: int, pause_sec: float) -> np.ndarray:
    if not chunks:
        return np.zeros(int(sample_rate * 0.2), dtype=np.float32)
    gap = np.zeros(int(sample_rate * pause_sec), dtype=np.float32)
    parts: list[np.ndarray] = []
    for index, chunk in enumerate(chunks):
        parts.append(np.asarray(chunk, dtype=np.float32).flatten())
        if index < len(chunks) - 1:
            parts.append(gap)
    return np.concatenate(parts)


def _split_sentences(text: str, max_len: int = XTTS_CHUNK) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    pieces = re.split(r"(?<=[.!?…])\s+", cleaned)
    sentences: list[str] = []
    buf = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        candidate = f"{buf} {piece}".strip() if buf else piece
        if len(candidate) <= max_len:
            buf = candidate
            continue
        if buf:
            sentences.append(buf)
        if len(piece) <= max_len:
            buf = piece
            continue
        for i in range(0, len(piece), max_len):
            sentences.append(piece[i : i + max_len].strip())
        buf = ""
    if buf:
        sentences.append(buf)
    return sentences


_TTS_ABBREVIATIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bU\.S\.A\.?\b", re.IGNORECASE), "yu es ey"),
    (re.compile(r"\bUSA\b"), "yu es ey"),
    (re.compile(r"\bU\.S\.?\b"), "yu es"),
    (re.compile(r"\bABD\b"), "ey bi di"),
    (re.compile(r"\bEU\b"), "i yu"),
    (re.compile(r"\bUK\b"), "yu key"),
    (re.compile(r"\bUN\b"), "yu en"),
    (re.compile(r"\bIMF\b"), "i em ef"),
    (re.compile(r"\bWTO\b"), "dabılyu ti o"),
    (re.compile(r"\bOECD\b"), "o i si di"),
    (re.compile(r"\bUSD\b"), "dolar"),
    (re.compile(r"\bEUR\b"), "avro"),
    (re.compile(r"\bGBP\b"), "sterlin"),
    (re.compile(r"\bFOB\b"), "ef o be"),
    (re.compile(r"\bCIF\b"), "si i ef"),
    (re.compile(r"\bHS\b"), "ha es"),
)


def _expand_abbreviations(text: str) -> str:
    """Yabancı kısaltmaları TTS'in Türkçe'de doğru okuyacağı fonetiğe çevirir.

    Ekran metni değişmez; yalnızca seslendirme girdisi dönüşür.
    """
    spoken = text
    for pattern, spoken_form in _TTS_ABBREVIATIONS:
        spoken = pattern.sub(spoken_form, spoken)
    return spoken


def _normalize_text(text: str) -> str:
    cleaned = text.replace("*", " ").replace("#", " ").replace("`", "")
    cleaned = _expand_abbreviations(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _cache_key(text: str, engine: str) -> Path:
    digest = hashlib.sha256(f"{engine}|{TTS_LANGUAGE}|{text}".encode("utf-8")).hexdigest()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{digest}.wav"


class LocalTTS:
    def __init__(self) -> None:
        self.language = TTS_LANGUAGE
        self.device = _device()
        self.engine_name = "none"
        self.ready = False
        self.error: str | None = None
        self._xtts = None
        self._xtts_speaker: str | None = None
        self._piper = None
        self._piper_config = None
        self._sample_rate = 24000
        preferred = os.getenv("TTS_ENGINE", "auto").strip().lower()
        self.preferred = preferred if preferred in {"auto", "xtts", "piper"} else "auto"

    def status(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "engine": self.engine_name,
            "language": self.language,
            "device": self.device,
            "error": self.error,
            "cloud": False,
        }

    def ensure_loaded(self) -> None:
        with _lock:
            if self.ready:
                return
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            if self.preferred == "xtts":
                self._load_xtts()
                self.engine_name = "xtts"
                self.ready = True
                self.error = None
                return
            self._load_piper()
            self.engine_name = "piper"
            self.ready = True
            self.error = None
            logger.info("Piper Türkçe ses yüklendi (%s)", PIPER_VOICE)

    def try_xtts(self) -> None:
        if self.preferred == "piper" or self._xtts is not None:
            return
        try:
            logger.info("XTTS yükleniyor (dil=%s, cihaz=%s)", TTS_LANGUAGE, self.device)
            self._load_xtts()
            with _lock:
                self.engine_name = "xtts"
                self.ready = True
                self.error = None
            logger.info("XTTS yüklendi (%s, dil=%s, konuşmacı=%s)", self.device, TTS_LANGUAGE, self._xtts_speaker)
        except Exception as exc:  # noqa: BLE001
            logger.warning("XTTS yüklenemedi, Piper ile devam: %s", exc)
            if not self.ready:
                self.error = str(exc)

    def _load_xtts(self) -> None:
        from TTS.api import TTS

        gpu = self.device == "cuda"
        tts = TTS(XTTS_MODEL, progress_bar=False, gpu=gpu)
        if gpu:
            tts.to("cuda")
        speakers = list(tts.speakers or [])
        speaker = os.getenv("TTS_SPEAKER", "").strip() or next(
            (name for name in PREFERRED_SPEAKERS if name in speakers),
            speakers[0] if speakers else None,
        )
        self._xtts = tts
        self._xtts_speaker = speaker
        self._sample_rate = 24000

    def _load_piper(self) -> None:
        from piper import PiperVoice, SynthesisConfig
        from piper.download_voices import download_voice

        model_path = MODELS_DIR / f"{PIPER_VOICE}.onnx"
        if not model_path.is_file():
            download_voice(PIPER_VOICE, MODELS_DIR)
        voice = PiperVoice.load(model_path, use_cuda=False)
        self._piper = voice
        self._piper_config = SynthesisConfig(length_scale=1.05, volume=1.0)
        self._sample_rate = int(voice.config.sample_rate)

    def synthesize(self, text: str) -> SpeakResult:
        cleaned = _normalize_text(text)
        if not cleaned:
            raise ValueError("Okunacak metin boş.")
        with _lock:
            return self._synthesize_locked(cleaned)

    def _synthesize_locked(self, cleaned: str) -> SpeakResult:
        self.ensure_loaded()
        cache = _cache_key(cleaned, self.engine_name)
        if cache.is_file() and cache.stat().st_size > 44:
            return SpeakResult(
                wav_bytes=cache.read_bytes(),
                engine=self.engine_name,
                language=TTS_LANGUAGE,
                sample_rate=self._sample_rate,
                cached=True,
            )
        if self.engine_name == "xtts":
            wav = self._synth_xtts(cleaned)
        else:
            wav = self._synth_piper(cleaned)
        cache.write_bytes(wav)
        return SpeakResult(
            wav_bytes=wav,
            engine=self.engine_name,
            language=TTS_LANGUAGE,
            sample_rate=self._sample_rate,
            cached=False,
        )

    def _synth_xtts(self, text: str) -> bytes:
        assert self._xtts is not None
        chunks: list[np.ndarray] = []
        rate = 24000
        for sentence in _split_sentences(text):
            audio = self._xtts.tts(
                text=sentence,
                speaker=self._xtts_speaker,
                language=TTS_LANGUAGE,
                split_sentences=False,
            )
            array = np.asarray(audio, dtype=np.float32)
            chunks.append(array)
            rate = int(getattr(self._xtts.synthesizer, "output_sample_rate", rate) or rate)
        merged = _concat_wavs(chunks, rate, PAUSE_SEC)
        self._sample_rate = rate
        return _float_to_wav(merged, rate)

    def _synth_piper(self, text: str) -> bytes:
        assert self._piper is not None
        chunks: list[np.ndarray] = []
        rate = int(self._piper.config.sample_rate)
        for sentence in _split_sentences(text, max_len=400):
            for audio in self._piper.synthesize(sentence, syn_config=self._piper_config):
                chunks.append(np.asarray(audio.audio_float_array, dtype=np.float32))
                rate = audio.sample_rate
        merged = _concat_wavs(chunks, rate, PAUSE_SEC)
        self._sample_rate = rate
        return _float_to_wav(merged, rate)


def get_tts() -> LocalTTS:
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = LocalTTS()
    return _engine


def warmup() -> dict[str, object]:
    engine = get_tts()
    logger.info("TTS warmup preferred=%s device=%s", engine.preferred, engine.device)
    try:
        engine.ensure_loaded()
        logger.info("TTS piper ready, trying XTTS")
        engine.try_xtts()
    except Exception as exc:  # noqa: BLE001
        logger.exception("TTS warmup failed")
        engine.error = str(exc)
        engine.ready = False
    return engine.status()


def synthesize_wav(text: str) -> SpeakResult:
    return get_tts().synthesize(text)
