from faster_whisper import WhisperModel
import sys
import os

def transcribe_audio(path):
    if not os.path.exists(path):
        print("File not found:", path)
        return
    
    print("Loading STT model…")
    model = WhisperModel("base")

    segments, info = model.transcribe(path)
    text = ""
    for seg in segments:
        text+=seg.text
    return text

if __name__ == "__main__":
    transcribe_audio(sys.argv[1])
