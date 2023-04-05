import speech_recognition as sr


class SpeechRecognition:
    azure_key = "82177f62988643fbab2d6583090e7e09"  # Azure Speech Recognition Key
    stop_listening = None
    recognizer = None
    microphone = None

    def __init__(self, azure_key):
        self.azure_key = azure_key
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

    def recognize_speech(self):
        speech_to_text = ""

        with sr.Microphone() as source:
            print("Say something!")
            #self.recognizer.non_speaking_duration = 0.4
            #self.recognizer.pause_threshold = 0.4
            audio = self.recognizer.listen(source, timeout=None)
        try:
            speech_to_text = self.recognizer.recognize_google(audio, language='en-in')
            print('You Said: ' + speech_to_text)
        except sr.UnknownValueError:
            print("Azure Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(("Could not request results from Azure Speech Voice Recognition service; {0}".format(e)))

        return speech_to_text




