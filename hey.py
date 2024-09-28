import pyttsx3
import subprocess

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def face_recognition():
    # Run the face recognition script
    speak("Starting face recognition.")
    subprocess.run(['python', 'face_object/face_final/final.py'])  # Adjusted path

def object_recognition():
    # Run the object recognition script
    speak("Starting object recognition.")
    subprocess.run(['python', 'face_object/object/yolorun.py'])  # Adjusted path

def text_detection():
    # Run the text detection script
    speak("Starting text detection.")
    subprocess.run(['python', 'face_object/OCR_laptop/text.py'])  # Adjusted path

def main():
    speak("Hello, select from the following options.")
    speak("1. Face recognition")
    speak("2. Object recognition")
    speak("3. Text detection")

    choice = input("Enter your choice: ")  # You can replace this with voice input if needed
    if choice == '1':
        face_recognition()
    elif choice == '2':
        object_recognition()
    elif choice == '3':
        text_detection()
    else:
        speak("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
