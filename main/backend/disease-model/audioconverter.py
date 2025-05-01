import speech_recognition as sr

class AudioToTextConverter:
    def __init__(self, language='en-US'):
        self.recognizer = sr.Recognizer()
        self.language = language

    def convert(self, wav_file_path):
        try:
            with sr.AudioFile(wav_file_path) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data, language=self.language)
                return text
        except sr.UnknownValueError:
            return "[ERROR] Speech was unintelligible."
        except sr.RequestError as e:
            return f"[ERROR] Could not request results from API: {e}"
        except FileNotFoundError:
            return "[ERROR] File not found."


# how to use         
# converter = AudioToTextConverter(language='en-US')
# text = converter.convert('sample.wav')
# print("Transcribed Text:", text)

