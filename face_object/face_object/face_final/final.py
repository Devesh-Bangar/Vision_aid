

import cv2
import face_recognition
import imutils
import pickle
import pyttsx3
from imutils.video import VideoStream

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load known faces and encodings
data = pickle.loads(open("face_object/face_final/encodings.pickle", "rb").read())
pickle
# Initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=1).start()

# Pre-trained face detector from OpenCV
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def facial_recognition():
    while True:
        frame = vs.read()
        if frame is None:
            print("No frame captured from the video stream.")
            continue

        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces in the grayscale frame
        rects = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []

        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"
            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)
            names.append(name)

        for ((top, right, bottom, left), name) in zip(boxes, names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            if names:
                print(name)
                engine.say(name)
            else:
                print("No face detected")
                engine.say("No face detected")

        engine.runAndWait()
        cv2.imshow("Frame", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    facial_recognition()
