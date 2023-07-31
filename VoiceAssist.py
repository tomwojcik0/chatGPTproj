# 'voice assistant' with chatGPT  (tiff in tech, "automating my life with python & chatgpt: coding my own..."
# note: may need to install some packages/libraries before this will run - see tiff's video)
import datetime
import speech_recognition as sr
import pyttsx3

# Initialize the recognizer
r = sr.Recognizer()

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to convert text to speech
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to listen to user's voice command
def listen_command():
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)

    try:
        print("Recognizing...")
        command = r.recognize_google(audio, language='en-in')
        print(f"User said: {command}\n")

    except Exception as e:
        print("Sorry, I didn't get that. Can you say it again?")
        return "None"
    return command

# Voice assistant
def voice_assistant():
    speak("Hello, I am your voice assistant. How can I help you today?")
    while True:
        command = listen_command().lower()

        # setting reminder
        if "set reminder" in command:
            speak("What should I remind you about?")
            reminder = listen_command()
            speak("When should I remind you?")
            time = listen_command()
            speak(f"Reminder set for {time} to {reminder}")

        # creating to-do list
        elif "create to-do list" in command:
            todo_list = []
            while True:
                speak("What task do you want to add to your to-do list?")
                task = listen_command()
                todo_list.append(task)
                speak(f"Task {task} added to your to-do list. Do you want to add another task?")
                answer = listen_command()
                if 'no' in answer:
                    break
            speak("Your to-do list is ready.")
            for task in todo_list:
                print(task)

        elif 'exit' in command:
            break

        else:
            speak("Sorry, I didn't get that. Can you say it again?")

if __name__ == "__main__":
    voice_assistant()
