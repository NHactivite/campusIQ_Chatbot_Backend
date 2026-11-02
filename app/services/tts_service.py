"""Azure Text-to-Speech. Assamese and English voices, chosen from the text."""
import asyncio
import os
import re
from collections import OrderedDict

import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

load_dotenv()

KEY = os.getenv("AZURE_SPEECH_KEY")
REGION = os.getenv("AZURE_SPEECH_REGION", "centralindia")
VOICES = {
    "as": ("as-IN", os.getenv("AZURE_TTS_VOICE_AS", "as-IN-YashicaNeural")),
    "en": ("en-IN", os.getenv("AZURE_TTS_VOICE_EN", "en-IN-NeerjaNeural")),
}
MAX_CHARS = 1500

_ASSAMESE = re.compile(r"[\u0980-\u09FF]")
_cache: "OrderedDict[tuple, bytes]" = OrderedDict()


def clean_for_speech(text: str) -> str:
    out = []
    for line in text.splitlines():
        s = line.strip()
        if re.match(r"^(source|sources|উৎস)\s*[:：]", s, re.IGNORECASE):
            continue  # citations are shown on screen, not read aloud
        s = re.sub(r"^[*\-•]\s+", "", s)   # bullets
        s = s.replace("**", "").replace("`", "")
        if s:
            out.append(s)
    return " ".join(out)[:MAX_CHARS]


def pick_lang(text: str, hint: str | None) -> str:
    if hint in ("as", "en"):
        return hint
    return "as" if _ASSAMESE.search(text) else "en"


def _escape(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _synthesize_sync(text: str, lang: str) -> bytes:
    if not KEY:
        raise RuntimeError("AZURE_SPEECH_KEY is not set")
    locale, voice = VOICES[lang]

    cfg = speechsdk.SpeechConfig(subscription=KEY, region=REGION)
    cfg.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3)
    # audio_config=None keeps the audio in memory (no temp file)
    synth = speechsdk.SpeechSynthesizer(speech_config=cfg, audio_config=None)

    ssml = (
        f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{locale}">'
        f'<voice name="{voice}">{_escape(text)}</voice></speak>'
    )
    result = synth.speak_ssml_async(ssml).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return bytes(result.audio_data)
    if result.reason == speechsdk.ResultReason.Canceled:
        d = result.cancellation_details
        raise RuntimeError(f"Azure TTS canceled: {d.reason} {d.error_details}")
    raise RuntimeError(f"Azure TTS failed: {result.reason}")


async def synthesize_to_mp3(text: str, lang_hint: str | None = None) -> bytes | None:
    """MP3 bytes, or None if speech is unavailable. Never raises."""
    spoken = clean_for_speech(text)
    if not spoken:
        return None
    lang = pick_lang(spoken, lang_hint)
    key = (lang, spoken)
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    try:
        audio = await asyncio.to_thread(_synthesize_sync, spoken, lang)
    except Exception as e:
        print(f"[tts] unavailable: {type(e).__name__}: {e}")
        return None
    _cache[key] = audio
    if len(_cache) > 50:
        _cache.popitem(last=False)
    return audio