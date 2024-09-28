import cv2
import os

def capture_images(person_name, save_dir='face_object/face_final/dataset'):
    # Create the person's directory if it doesn't exist
    person_dir = os.path.join(save_dir, person_name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)

    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print(f"[INFO] Starting to capture images for {person_name}. Press 'q' to stop.")
    count = 0
    while count < 5:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture image.")
            break

        # Display the frame
        cv2.imshow("Frame", frame)

        # Save the frame to the person's directory
        img_path = os.path.join(person_dir, f"{person_name}_{count + 1}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"[INFO] Captured {img_path}")
        count += 1

        # Wait for a key press and check if it's 'q' to quit early
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Release the webcam and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Finished capturing images for {person_name}.")

if __name__ == "__main__":
    person_name = input("Enter the person's name: ")
    capture_images(person_name)
