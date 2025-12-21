# 🎙️ AI-powered speech analysis system

A comprehensive AI-powered speech analysis system that evaluates pitch presentations using advanced audio processing and multi-agent AI feedback. Athena analyzes speech patterns, delivery metrics, and provides feedback from four different "Shark Tank" perspectives.

##  What is Athena?

Athena is an intelligent speech coach that helps entrepreneurs, presenters, and speakers improve their pitch delivery by analyzing multiple aspects of their speech and providing multi-perspective AI feedback.

##  Features

- **Audio Transcription**: Automatic speech-to-text using Faster Whisper
- **Speech Metrics Analysis**:
  - Speaking pace (Words Per Minute)
  - Pitch variation and tone analysis
  - Volume and energy patterns
  - Pause detection and hesitation analysis
  - Filler word identification
  - Confidence and emotion scoring
- **Multi-Agent AI Feedback**: Get perspectives from four specialized "sharks":
  - Visionary Shark (market potential & innovation)
  - Finance Shark (revenue model & unit economics)
  - Skeptic Shark (risks & weaknesses)
  - Customer Advocate Shark (problem-solution fit)

##  Project Structure
```
ATHENA PROJECT/
├── __pycache__/
│   ├── agents_feedback.cpython-312.pyc
│   └── transcribe.cpython-312.pyc
├── codes/
│   ├── agents_feedback.py    # Multi-agent feedback system
│   ├── app.py                # Main analysis pipeline
│   └── transcribe.py         # Audio transcription module
├── requirements.txt          # Python dependencies
└── sample_audio.wav         # Sample audio file
```

##  Installation

### Prerequisites

- Python 3.8 or higher
- FFmpeg (required for audio processing)

### Install FFmpeg

**Windows:**
```bash
# Using Chocolatey
choco install ffmpeg

# Or download from https://ffmpeg.org/download.html
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

### Install Python Dependencies
```bash
# Clone the repository
git clone https://github.com/yourusername/athena-project.git
cd athena-project

# Install required packages
pip install -r requirements.txt
```

##  Usage

### Basic Usage
```bash
python codes/app.py <audio_file.wav> <DEEPSEEK_API_KEY>
```

### Example
```bash
python codes/app.py sample_audio.wav sk-xxxxxxxxxxxxxxxx
```

### Output

The system outputs a comprehensive JSON report including:
```json
{
  "audio_path": "sample_audio.wav",
  "transcript": "...",
  "analysis": {
    "pace": {
      "wpm": 145.67,
      "word_count": 243,
      "duration_minutes": 1.67,
      "pace_type": "Normal"
    },
    "pitch": {
      "mean_pitch_hz": 185.23,
      "coefficient_variation": 22.45,
      "tone_type": "Normal Variation"
    },
    "volume": { ... },
    "pauses": { ... },
    "filler_words": { ... },
    "emotion_confidence": { ... }
  },
  "agent_feedback": {
    "Visionary Shark": { ... },
    "Finance Shark": { ... },
    "Skeptic Shark": { ... },
    "Customer Advocate Shark": { ... }
  }
}
```

##  Configuration

### API Key

You need a DeepSeek API key to use the multi-agent feedback system. Get yours at [https://platform.deepseek.com](https://platform.deepseek.com)

### Whisper Model

The transcription uses the `base` model by default. You can modify this in `transcribe.py`:
```python
model = WhisperModel("base")  # Options: tiny, base, small, medium, large
```

##  Analysis Metrics

### Pace Analysis
- **WPM Range**: 
  - Slow: < 110 WPM
  - Normal: 110-150 WPM
  - Fast: 150-180 WPM
  - Very Fast: > 180 WPM

### Pitch Analysis
- Mean pitch frequency
- Standard deviation
- Coefficient of variation
- Tone classification (Monotone to Very Dynamic)

### Volume Analysis
- Energy levels and variation
- Volume expressiveness rating

### Pause Analysis
- Pause count and duration
- Hesitation index
- Fluency rating

### Filler Words Detection
Detects common filler words: "uh", "um", "like", "you know", "actually", "basically", etc.

### Confidence Scoring
Composite score based on pitch variation, volume expressiveness, speaking pace, and pause patterns.

##  AI Agent System

Each agent evaluates the pitch from their unique perspective and provides:
- Narrative feedback (5-8 lines)
- Strengths (3 key points)
- Weaknesses (3 key points)
- Investment verdict (Invest / Not Invest / Need More Info)

##  Testing with Sample Audio
```bash
# Test with the included sample
python codes/app.py sample_audio.wav YOUR_API_KEY
```

##  Dependencies

Key libraries:
- `faster-whisper`: Audio transcription
- `librosa`: Audio analysis
- `pydub`: Audio processing
- `numpy/scipy`: Numerical operations
- `requests`: API calls to DeepSeek

Full list available in `requirements.txt`

**Note**: This system requires an active internet connection for the DeepSeek API calls and audio transcription.
