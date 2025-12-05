import librosa
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_silence
import re
from collections import Counter
import json
import sys
import warnings
from transcribe import transcribe_audio
warnings.filterwarnings("ignore")

# external modules you already built
from agents_feedback import run_shark_analysis
from transcribe import transcribe_audio
def to_python_types(obj):
    """
    Recursively convert numpy types to native Python types
    so they can be JSON serialized.
    """
    if isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
def analyze_speech(audio_path, transcript,api_key):
    """
    Full speech analysis pipeline.
    Inputs:
        audio_path (str): path to audio file
        transcript (str or None): if None → auto-transcribe
    Output:
        dictionary with all metrics + agent feedback
    """

    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    # ------------------------------
    # 2. Get transcript if needed
    # ------------------------------
    if transcript is None:
        print("⏳ Transcribing using Whisper…")
        transcript = transcribe_audio(audio_path)

    # ------------------------------
    # 3. Pace (WPM)
    # ------------------------------
    if transcript:
        words = transcript.split()
        word_count = len(words)
        minutes = duration / 60
        wpm = round(word_count / minutes, 2) if minutes > 0 else 0

        if wpm < 110:
            pace_type = "Slow"
        elif wpm < 150:
            pace_type = "Normal"
        elif wpm < 180:
            pace_type = "Fast"
        else:
            pace_type = "Very Fast"

        pace_data = {
            "wpm": wpm,
            "word_count": word_count,
            "duration_minutes": round(minutes, 2),
            "pace_type": pace_type
        }
    else:
        pace_data = {
            "wpm": None,
            "word_count": None,
            "duration_minutes": round(duration / 60, 2),
            "pace_type": "Unknown (no transcript)"
        }

    # ------------------------------
    # 4. Pitch Analysis
    # ------------------------------
    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7")
    )
    f0_clean = f0[~np.isnan(f0)]

    if len(f0_clean) > 0:
        mean_pitch = np.mean(f0_clean)
        std_pitch = np.std(f0_clean)
        pitch_range = np.max(f0_clean) - np.min(f0_clean)
        cv = (std_pitch / mean_pitch) * 100 if mean_pitch > 0 else 0

        if cv < 10:
            tone_type = "Very Monotone"
        elif cv < 20:
            tone_type = "Somewhat Monotone"
        elif cv < 30:
            tone_type = "Normal Variation"
        else:
            tone_type = "Very Dynamic"

        pitch_data = {
            "mean_pitch_hz": round(mean_pitch, 2),
            "std_pitch_hz": round(std_pitch, 2),
            "pitch_range_hz": round(pitch_range, 2),
            "coefficient_variation": round(cv, 2),
            "tone_type": tone_type
        }
    else:
        pitch_data = {"status": "Could not detect pitch"}

    # ------------------------------
    # 5. Volume / Energy
    # ------------------------------
    rms = librosa.feature.rms(y=y)[0]
    mean_energy = np.mean(rms)
    std_energy = np.std(rms)
    energy_range = np.max(rms) - np.min(rms)
    cv_energy = (std_energy / mean_energy) * 100 if mean_energy > 0 else 0

    if cv_energy < 15:
        volume_type = "Flat / Monotone"
    elif cv_energy < 30:
        volume_type = "Moderate Variation"
    else:
        volume_type = "Very Expressive"

    volume_data = {
        "mean_energy": float(round(mean_energy, 4)),
        "std_energy": float(round(std_energy, 4)),
        "energy_range": float(round(energy_range, 4)),
        "coefficient_variation": round(cv_energy, 2),
        "volume_type": volume_type
    }

    # ------------------------------
    # 6. Pauses / Hesitation
    # ------------------------------
    try:
        audio = AudioSegment.from_file(audio_path)
        silences = detect_silence(audio, min_silence_len=300, silence_thresh=-40)
        pauses = [(s / 1000, e / 1000) for s, e in silences]
        durations = [(e - s) for s, e in pauses]

        total_pause = sum(durations)
        avg_pause = np.mean(durations) if durations else 0
        hesitation_index = len(pauses) / (duration / 60) if duration > 0 else 0

        if hesitation_index < 5:
            hesitation_level = "Low (Fluent)"
        elif hesitation_index < 15:
            hesitation_level = "Moderate"
        else:
            hesitation_level = "High (Hesitant)"

        pause_data = {
            "pause_count": len(pauses),
            "total_pause_time_sec": round(total_pause, 2),
            "avg_pause_duration_sec": round(avg_pause, 2),
            "hesitation_index": round(hesitation_index, 2),
            "hesitation_level": hesitation_level,
            "pause_percentage": round((total_pause / duration) * 100, 2)
        }
    except Exception as e:
        pause_data = {"status": f"Pause analysis error: {e}"}

    # ------------------------------
    # 7. Filler Words
    # ------------------------------
    if transcript:
        transcript_l = transcript.lower()
        patterns = [
            r"\buh+\b", r"\bum+\b", r"\buhm+\b", r"\bah+\b",
            r"\ber+\b", r"\blike\b", r"\byou know\b", r"\bactually\b",
            r"\bbasically\b", r"\bliterally\b", r"\bkind of\b", r"\bsort of\b"
        ]

        matches = []
        for p in patterns:
            matches.extend(re.findall(p, transcript_l))

        count = len(matches)
        words = transcript.split()
        word_count = len(words)

        filler_rate = (count / word_count * 100) if word_count else 0

        if filler_rate < 2:
            level = "Low (clean)"
        elif filler_rate < 5:
            level = "Moderate"
        else:
            level = "High (frequent)"

        filler_data = {
            "filler_count": count,
            "filler_rate_percent": round(filler_rate, 2),
            "filler_level": level,
            "filler_breakdown": dict(Counter(matches)),
            "total_words": word_count
        }
    else:
        filler_data = {"status": "Transcript required"}

    # ------------------------------
    # 8. Emotion / Confidence
    # ------------------------------
    confidence_score = 0
    indicators = []

    # pitch
    cv = pitch_data.get("coefficient_variation", 0)
    if cv > 20:
        confidence_score += 2
        indicators.append("dynamic_pitch")
    elif cv < 10:
        confidence_score -= 1
        indicators.append("monotone")

    # volume
    if cv_energy > 25:
        confidence_score += 2
        indicators.append("expressive")
    elif cv_energy < 15:
        confidence_score -= 1

    # pace
    wpm = pace_data.get("wpm", 0)
    if isinstance(wpm, (int, float)) and 120 <= wpm <= 160:
        confidence_score += 1
    elif wpm > 180:
        indicators.append("rushed")
    elif wpm < 100:
        indicators.append("hesitant")
        confidence_score -= 1

    # pauses
    h_idx = pause_data.get("hesitation_index", 0)
    if h_idx < 5:
        confidence_score += 2
    elif h_idx > 15:
        confidence_score -= 2
        indicators.append("nervous")

    if confidence_score >= 4:
        mood = "Confident & Expressive"
    elif confidence_score >= 1:
        mood = "Moderately Confident"
    elif confidence_score >= -1:
        mood = "Neutral"
    else:
        mood = "Nervous / Hesitant"

    emotion_data = {
        "confidence_score": confidence_score,
        "classification": mood,
        "indicators": indicators
    }

    # ------------------------------
    # 9. Shark Tank Agents (DeepSeek)
    # ------------------------------
    agent_feedback = run_shark_analysis(transcript,api_key)

    # ------------------------------
    # 📦 FINAL JSON DICTIONARY
    # ------------------------------
    return {
        "audio_path": audio_path,
        "transcript": transcript,
        "analysis": {
            "pace": pace_data,
            "pitch": pitch_data,
            "volume": volume_data,
            "pauses": pause_data,
            "filler_words": filler_data,
            "emotion_confidence": emotion_data,
        },
        "agent_feedback": agent_feedback
    }


if __name__ == "__main__":
    # Expect exactly 2 user inputs: audio.wav and API_KEY
    if len(sys.argv) < 3:
        print("Usage: python app.py <audio.wav> <API_KEY>")
        sys.exit(1)

    audio_path = sys.argv[1]
    api_key = sys.argv[2]

    # STEP 1 → Transcribe Audio
    transcript = transcribe_audio(audio_path)

    # STEP 2 → Run Full Speech Analysis + Shark Agent Feedback
    results = analyze_speech(audio_path, transcript, api_key)

    # STEP 3 → Print JSON output
    clean_results = to_python_types(results)
    print(json.dumps(clean_results, indent=4, ensure_ascii=False))