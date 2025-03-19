import cv2
import mediapipe as mp
import math

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

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def recognize_gun(hand_landmarks):
    # Extract necessary landmarks
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]

    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # check if thumb is pointing up
    is_thumb_up = thumb_tip.y < thumb_mcp.y

    # check if index is pointing straight right
    is_index_horizontal = abs(index_tip.y - index_pip.y) < 0.02
    is_index_pointing_right = index_tip.x > index_pip.x and index_pip.x > index_mcp.x
    # is_index_pointing_left = index_tip.x < index_pip.x and index_pip.x < index_mcp.x

    # make sure mid, ring, pinky are curled
    is_middle_curled = middle_tip.y > middle_pip.y
    is_ring_curled = ring_tip.y > index_mcp.y
    is_pinky_curled = pinky_tip.y > index_mcp.y

    if is_thumb_up and is_index_horizontal and is_index_pointing_right and is_middle_curled and is_ring_curled and is_pinky_curled:
        return "Gun_Right"
    
    # if is_thumb_up and is_index_horizontal and is_index_pointing_left and is_middle_curled and is_ring_curled and is_pinky_curled:
    #     return "Gun_Left"

    return "Unknown"

def main():
    # webbrowser.open('https://supermarioplay.com/', new=2)

    # Initialize video capture
    cap = cv2.VideoCapture(0)  # 0 is the default webcam

    held_keys = {}

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

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
                    "Victory": ["left", "up"],
                    "Thumb_Down": "down",
                    "Thumb_Up": "up",
                    # "Pointing_Up": ["right", "up"],
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

                # Display recognized gesture and confidence
                cv2.putText(image, f"Gesture: {recognized_gesture} ({confidence:.2f})", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # To improve performance, optionally mark the image as not writeable to pass by reference.
            # image_rgb.flags.writeable = True
            results = hands.process(image_rgb)

            # Draw the hand annotations on the image.
            # image_rgb.flags.writeable = True
            # image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Recognize gesture
                    gun = recognize_gun(hand_landmarks)

                    key_mappings = {
                        "Gun_Right": ["right", "up"],
                        # "Gun_Left": ["left", "up"],
                    }

                    if gun in key_mappings:
                        keys = key_mappings[gun]
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

                    # Display gesture near hand location
                    cv2.putText(image, gun, 
                        (int(hand_landmarks.landmark[0].x * image.shape[1]), 
                        int(hand_landmarks.landmark[0].y * image.shape[0]) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            # Display the resulting image
            cv2.imshow('Gesture Recognition', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
