import cv2
import json
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import layers, models

# --- UPDATED MEDIAPIPE IMPORTS (v0.10.14+) ---
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing

# --- CONFIGURATION ---
DATA_FILE = "asl_dataset.json"
INPUT_SHAPE = (21, 3, 1)  
CLASSES = ["B", "C", "D", "E", "F", "I", "K", "L", "O", "U", "V", "W", "Y"]  

# Initialize Hands detector
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# --- 1. DATA COLLECTION HELPER ---
def collect_data():
    """Captures hand landmarks from the webcam to build a dataset."""
    cap = cv2.VideoCapture(0)
    dataset = {label: [] for label in CLASSES}
   
    print("--- ASL Data Collection ---")
    print("Press the corresponding letter key on your keyboard to save a frame.")
    print("Press 'q' to quit data collection.")
   
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
       
        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
               
                # Normalization relative to wrist landmark (index 0)
                wrist_x = hand_landmarks.landmark[0].x
                wrist_y = hand_landmarks.landmark[0].y
                wrist_z = hand_landmarks.landmark[0].z

                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])

        # On-screen frame counters
        y_pos = 30
        for class_name, items in dataset.items():
            cv2.putText(frame, f"{class_name}: {len(items)}", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_pos += 20
               
        cv2.imshow("Data Collection", frame)
       
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        elif key != 255:
            char = chr(key).upper()
            if char in CLASSES:
                if results.multi_hand_landmarks:
                    dataset[char].append(landmarks)
                    print(f"Saved frame for {char} (Total: {len(dataset[char])})")
                else:
                    print("No hand detected! Position your hand in frame.")

    cap.release()
    cv2.destroyAllWindows()
   
    with open(DATA_FILE, 'w') as f:
        json.dump(dataset, f)
    print(f"Dataset successfully saved to {DATA_FILE}")

# --- 2. MODEL BUILDING & TRAINING ---
def train_model():
    """Loads dataset, builds a model, and trains it."""
    if not os.path.exists(DATA_FILE):
        print("Dataset file not found. Please run data collection first.")
        return None
       
    with open(DATA_FILE, 'r') as f:
        dataset = json.load(f)
       
    X, y = [], []
    label_map = {label: idx for idx, label in enumerate(CLASSES)}
   
    for label, items in dataset.items():
        for item in items:
            X.append(item)
            y.append(label_map[label])
           
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
   
    if len(X) == 0:
        print("Dataset is empty. Collect data first.")
        return None
       
    X = X.reshape(-1, 21, 3, 1)
   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
   
    # Modern Keras 3 architecture using Input layer
    model = models.Sequential([
        layers.Input(shape=INPUT_SHAPE),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
       
        layers.Conv2D(64, (3, 1), activation='relu', padding='same'),
        layers.Flatten(),
       
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(len(CLASSES), activation='softmax')
    ])
   
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
   
    print("\nTraining CNN Model...")
    model.fit(
        X_train, y_train,
        epochs=30,
        validation_data=(X_test, y_test),
        batch_size=8
    )
   
    return model

# --- 3. REAL-TIME TRANSLATION ---
def run_inference(model):
    """Uses the trained model to translate signs in real-time."""
    cap = cv2.VideoCapture(0)
    print("\n--- Real-Time ASL Translation Started ---")
    print("Press 'q' to exit.")
   
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
           
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
       
        prediction_text = "No hand detected"
       
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
               
                # Normalization matching data collection
                wrist_x = hand_landmarks.landmark[0].x
                wrist_y = hand_landmarks.landmark[0].y
                wrist_z = hand_landmarks.landmark[0].z

                landmarks = [
                    [lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z]
                    for lm in hand_landmarks.landmark
                ]
                
                input_data = np.array(landmarks, dtype=np.float32).reshape(1, 21, 3, 1)
               
                predictions = model.predict(input_data, verbose=0)
                class_idx = np.argmax(predictions)
                confidence = predictions[0][class_idx]
               
                if confidence > 0.75:
                    prediction_text = f"Letter: {CLASSES[class_idx]} ({confidence*100:.1f}%)"
                else:
                    prediction_text = "Uncertain..."
                   
        cv2.putText(
            frame, prediction_text, (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
        )
        cv2.imshow("ASL to English Translator", frame)
       
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- EXECUTION FLOW ---
if __name__ == "__main__":
    collect_data()
    trained_model = train_model()
    if trained_model:
        run_inference(trained_model)