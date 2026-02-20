import cv2
import numpy as np
import os
import mediapipe as mp

# --- CONFIGURATION ---
DATA_PATH = os.path.join('MP_Data_10/Test') 
SEQUENCE_LENGTH = 30 

# Setup MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    # 1. Pose Landmarks (Body)
    if results.pose_landmarks:
        # Includes x, y, z, and visibility
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33*4)
        
    # 2. Left Hand
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21*3)
        
    # 3. Right Hand
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21*3)
        
    return np.concatenate([pose, lh, rh])

def draw_styled_landmarks(image, results):
    # Draw Pose (Body)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )
    
    # Draw Left Hand
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)) 
    
    # Draw Right Hand
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) 

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def record_gesture(action_name):
    action_path = os.path.join(DATA_PATH, action_name)
    if not os.path.exists(action_path):
        os.makedirs(action_path)
    
    # Check existing files to handle resuming
    existing_sequences = [int(f) for f in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, f))]
    sequence_count = max(existing_sequences) + 1 if existing_sequences else 0
    
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            
            # UI Overlay
            cv2.putText(image, f"Recording: '{action_name}'", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Takes Recorded: {sequence_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "Press 'SPACE' to Record | 'Q' to Quit", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('OctoSign Recorder', image)
            
            key = cv2.waitKey(10) & 0xFF

            if key == 32: # Spacebar pressed
                save_path = os.path.join(action_path, str(sequence_count))
                os.makedirs(save_path, exist_ok=True)
                
                print(f"🎬 Recording Take {sequence_count}...")
                
                for frame_num in range(SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    if not ret: break
                    
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)
                    
                    # 'Recording' Red Dot
                    cv2.circle(image, (30, 30), 20, (0, 0, 255), -1) 
                    cv2.putText(image, "REC", (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('OctoSign Recorder', image)
                    cv2.waitKey(1) 
                    
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(save_path, str(frame_num))
                    np.save(npy_path, keypoints)
                
                sequence_count += 1
                print(f"✅ Take {sequence_count-1} Saved!")
                
            elif key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# --- ENTRY POINT ---
if __name__ == "__main__":
    while True:
        word = input("Enter the word you want to record (or 'exit'): ").strip().lower()
        if word == 'exit':
            break
        if word == '':
            continue
            
        print(f"Opening camera for '{word}'... Press SPACE to record takes.")
        record_gesture(word)