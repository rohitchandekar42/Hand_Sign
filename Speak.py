import pyttsx3
import warnings
import streamlit as st

# Suppress all warnings
warnings.filterwarnings("ignore")

def SpeakWindow(Text,box):
    if Text != '':
        engine = pyttsx3.init('sapi5')
        voices = engine.getProperty('voices')
        engine.setProperty('voice',voices[0].id)
        engine.setProperty('rate',190)
        engine.say(Text)
        print(f"You  : {Text}.\n")
        box.write(f"You  : {Text}.\n")
        engine.runAndWait()
        box.empty()