import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import numpy as np

class AudioToTextConverter:
    def __init__(self, filename, model_size="base"):
        self.filename = filename
        self.model = whisper.load_model(model_size)

    def transcribe(self):
        result = self.model.transcribe(self.filename)
        print("\nTranscription:")
        print(result["text"])
        return result["text"]

def record_audio(filename, fs=44100):#we don't need this one when we have already a file just for testing 
    recording = []

    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        recording.append(indata.copy())

    print("Recording... Press Enter to stop.")
    with sd.InputStream(samplerate=fs, channels=1, dtype='int16', callback=audio_callback):
        input()
    audio = np.concatenate(recording, axis=0)
    write(filename, fs, audio)
    print(f"Recording saved to {filename}")

#if __name__ == "__main__":
#    filename = "recording.wav"
#    record_audio(filename) we don't need this one when we have already a file just for testing 
#    transcriber = AudioToTextConverter(filename)
#    transcriber.transcribe()

# Example usage:
# converter = AudioToTextConverter(language='en')
# text = converter.convert('sample.wav')
# print("Transcribed Text:", text)
