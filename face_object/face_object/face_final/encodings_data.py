import face_recognition
import pickle
import cv2
import os

# Path to the directory containing the images
image_dir = "face_object/face_final/dataset"

known_encodings = []
known_names = []

# Loop over the image paths
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("jpg") or file.endswith("png"):
            image_path = os.path.join(root, file)
            image = cv2.imread(image_path)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect the (x, y)-coordinates of the bounding boxes
            boxes = face_recognition.face_locations(rgb, model="hog")
            # Compute the facial embeddings for each face
            encodings = face_recognition.face_encodings(rgb, boxes)

            # Loop over the encodings
            for encoding in encodings:
                known_encodings.append(encoding)
                name = os.path.basename(root)
                known_names.append(name)

# Save the known encodings and names to a pickle file
data = {"encodings": known_encodings, "names": known_names}
with open("encodings.pickle", "wb") as f:
    f.write(pickle.dumps(data))

print("[INFO] Encodings have been saved to encodings.pickle")
