#! /usr/bin/env python

import os
import pyglet


window = pyglet.window.Window(width=900, height=900)


from gtts import gTTS
import tensorflow as tf
from face_recognizer import FaceRecognizer
from topic_extraction import TopicExtraction
from emotion_recognition import EmotionRecognition
from user_interface import UserInterface
from context_recognition import ContextRecognition
from speech_to_text import SpeechRecognition


def text_to_speech(text):
    if text:
        tts = gTTS(text=text, lang='en')
        tts.save("speech.mp3")
        os.system("mpg321 speech.mp3")


# To recollect the chat history of a particular user
def history_recollection():
    history = FaceRecognizer(user_interface)
    history.build_imagecsv()
    user_number = history.RecognizeFace()
    user_name = history.names[user_number]

    if user_name is None:
        chatbot_response = "I don't think we've met before, what's your name?"
        user_interface.update_sprites(chatbot_response, " ".join(("Emotion: ", meeting_emotion)), " ".join(("User: ", "Unknown")), "Primary Topics: ")
        user_interface.render()
        text_to_speech(chatbot_response)
        #user_name = speech.recognize_speech()
        user_name=input("Enter yout username")
        user_interface.update_sprites(chatbot_response, " ".join(("Emotion: ", meeting_emotion)), " ".join(("User: ", user_name)), "Primary Topics: ")
        user_interface.render()
        #history.retrain(user_name)

    else:
        chatbot_response = "It's good to see you again, " + user_name
        user_interface.update_sprites(chatbot_response, " ".join(("Emotion: ", meeting_emotion)), " ".join(("User: ", user_name)), "Primary Topics: ")
        user_interface.render()
        text_to_speech(chatbot_response)

    #history.exit()
    return user_name

if __name__ == "__main__":
    print(tf.executing_eagerly())
    input_sentence = ""
    current_emotion = ""
    meeting_emotion = ""
    top_words = []

    # Enter a Azure Speech token into the SpeechRecognition constructor
    speech = SpeechRecognition("82177f62988643fbab2d6583090e7e09")

    user_interface = UserInterface(window)

    context = ContextRecognition()
    context.load_corpus("corpus/")
    context.load_model()

    emotion = EmotionRecognition(user_interface)
    emotion.start()

    # The chatbot will start listening to the user after they say "Hi Bot"
    # The bot will then read your emotion via webcam
    while input_sentence != "Hi Bot":
        user_interface.update_sprites("Listening...","Emotion: Angry", "User: Arivan", "Primary Topics: ")
        user_interface.render()

        emotion.run()

        # input_sentence = speech.recognize_speech()
        # print(("You said: ", input_sentence))

        meeting_emotion = emotion.get_emotion()
        print(("Emotion read: ", meeting_emotion))
        meeting_emotion="Neutral"
        break

    # Make a call to see whether or not the user is recognized
    user_name = history_recollection()

    topic_extract = TopicExtraction()
    topic_extract.load_history(user_name)
    user_name="Arian"
    # Based on the emotion the chatbot will respond differently
    if meeting_emotion == "Happy" or meeting_emotion == "Surprise":
        chatbot_response = "You seem like you're in a good mood today. How can I help you?"

    elif meeting_emotion == "Neutral":
        chatbot_response = "How may I help you today?"

    else:
        chatbot_response = "I sense that you may be bothered right now. How can I help?"

    user_interface.update_sprites(chatbot_response, " ".join(("Emotion: ", meeting_emotion)), " ".join(("User: ", user_name)), "Primary Topics: ")
    user_interface.render()
    text_to_speech(chatbot_response)
    user_interface.remove_webcam_label()

    # Run until the user says Goodbye computer
    chatbot_response = ""
    input_sentence = speech.recognize_speech()
    while input_sentence != "goodbye computer":
        #print "User said: ", input_sentence

        if input_sentence != "":
            print(("Original input sentence: ", input_sentence))

            top_words = topic_extract.get_top_topics(input_sentence)
            input_sentence = " ".join((input_sentence, top_words[0]))
            print(input_sentence)

            response, correlation = context.compute_document_similarity(input_sentence)

            if correlation > 0:
                chatbot_response = response

            elif correlation == 0 and input_sentence:
                chatbot_response = "I'm sorry, but I couldn't find an appropriate response to your query."

        if len(top_words) > 0:
            user_interface.update_sprites(chatbot_response, " ".join(("Emotion: ", "N/A")), " ".join(("User: ", user_name)), " ".join(("Primary Topics: ", top_words[0], ", ", top_words[1])))
        else:
            user_interface.update_sprites(chatbot_response, " ".join(("Emotion: ", "N/A")), " ".join(("User: ", user_name)), "Primary Topics: ")
        user_interface.render()
        text_to_speech(chatbot_response)

        chatbot_response = ""
        top_words = []
        input_sentence = speech.recognize_speech()

    # Run the emotion check again to see how the user feels after talking
    emotion.reset()
    emotion.run()
    leaving_emotion = emotion.get_emotion()
    print(("Leaving emotion: ", leaving_emotion))
    #print "Name: ", user_name

    # If the user came in happy and left sad, apologize to them
    if (meeting_emotion == "Happy" or meeting_emotion == "Surprise") and (leaving_emotion == "Angry" or leaving_emotion ==
        "Sad" or leaving_emotion == "Disgust" or leaving_emotion == "Fear"):

        chatbot_response = "I apologize, it seems that I left you in a worse mood than before we talked."

    # Else if the user came in angry and left happy, respond happy
    elif (meeting_emotion == "Angry" or meeting_emotion == "Sad" or meeting_emotion == "Disgust" or meeting_emotion == "Fear") and (leaving_emotion == "Happy" or leaving_emotion == "Surprise"):
        chatbot_response = "I'm happy, it seems that you're in a better mood than before our talk."

    # Otherwise, respond with a standard goodbye
    else:
        if user_name == None:
            chatbot_response = "Goodbye. It was nice meeting you."
        else:
            chatbot_response = "Bye, " + user_name + ". It was nice talking to you."

    if len(top_words) > 0:
        user_interface.update_sprites(chatbot_response, " ".join(("Emotion: ", leaving_emotion)), " ".join(("User: ", user_name)), " ".join(("Primary Topics: ", top_words[0], ", ", top_words[1])))
    else:
        user_interface.update_sprites(chatbot_response, " ".join(("Emotion: ", leaving_emotion)), " ".join(("User: ", user_name)), "Primary Topics: ")

    user_interface.render()
    text_to_speech(chatbot_response)
    #topic_extract.write_history(user_name)
















