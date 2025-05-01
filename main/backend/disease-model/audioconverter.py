import whisper

class AudioToTextConverter:
    def __init__(self, language='en'):
        self.language = language
        self.model = whisper.load_model("base")  # You can use "small", "medium", or "large" models based on your needs

    def convert(self, wav_file_path):
        try:
            # Transcribe the audio using Whisper
            result = self.model.transcribe(wav_file_path, language=self.language)
            return result['text']
        except Exception as e:
            return f"[ERROR] An error occurred: {e}"

# Example usage:
# converter = AudioToTextConverter(language='en')
# text = converter.convert('sample.wav')
# print("Transcribed Text:", text)
