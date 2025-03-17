import cv2
import mediapipe as mp

# Import the tasks API for gesture recognition
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
from mediapipe.tasks.python import BaseOptions
from mediapipe.framework.formats import landmark_pb2

import webbrowser
import time
import pyautogui

# Path to the gesture recognition model
model_path = "gesture_recognizer.task"  # Update this to the correct path where the model is saved

# Initialize the Gesture Recognizer
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=1
)
gesture_recognizer = GestureRecognizer.create_from_options(options)

def main():
    # webbrowser.open('https://supermarioplay.com/', new=2)

    # Initialize video capture
    cap = cv2.VideoCapture(0)  # 0 is the default webcam

    held_keys = {}

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        # and convert the BGR image to RGB.
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the image to a Mediapipe Image object for the gesture recognizer
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Perform gesture recognition on the image
        result = gesture_recognizer.recognize(mp_image)

        # Draw the gesture recognition results on the image
        if result.gestures:
            recognized_gesture = result.gestures[0][0].category_name
            confidence = result.gestures[0][0].score
            
            key_mappings = {
                "Open_Palm": "right",
                "Closed_Fist": "left",
                "Pointing_Up": ["right", "up"],
                "Victory": ["left", "up"],
                "Thumb_Down": "down",
                "Thumb_Up": "up",
            }

            if recognized_gesture in key_mappings:
                keys = key_mappings[recognized_gesture]
                if isinstance(keys, str): # convert string into list for easy iteration
                    keys = [keys]

                for key in keys:
                    # press keys down
                    if key not in held_keys: 
                        pyautogui.keyDown(key)
                        held_keys[key] = True
            else:
            # release keys
                for key in list(held_keys.keys()): # if keys were held down (bool), release the key up 
                    pyautogui.keyUp(key)
                    del held_keys[key]

            # code for first attempt, mario moved very little because registering key presses 
            # if recognized_gesture == "Open_Palm":
            #     # pyautogui.press('right')     # press the left arrow key
            #     pyautogui.press('right')  # hold down the shift key
            
            # if recognized_gesture == "Closed_Fist":
            #     pyautogui.press('left')

            # if recognized_gesture == "Pointing_Up":
            #     pyautogui.press('right')
            #     pyautogui.press('up')

            # if recognized_gesture == "Thumb_Down":
            #     pyautogui.press('down')

            # if recognized_gesture == "Thumb_Up":
            #     pyautogui.press('up')

            # Display recognized gesture and confidence
            cv2.putText(image, f"Gesture: {recognized_gesture} ({confidence:.2f})", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting image
        cv2.imshow('Gesture Recognition', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
