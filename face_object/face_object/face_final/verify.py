
import pickle

# Load the known faces and embeddings
data = pickle.loads(open("face_object/face_final/encodings.pickle", "rb").read())

# Print the names and encodings
print(f"Names: {data['names']}")
print(f"Encodings: {data['encodings']}")
