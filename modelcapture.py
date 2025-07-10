import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import joblib

try:
    movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    model = movenet.signatures["serving_default"]
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

def load_and_train_classifier(csv_file="yoga_detection_dataset.csv", model_file="pose_classifier.joblib"):
    try:
        df = pd.read_csv(csv_file)
        keypoint_columns = [f"kp_{i}" for i in range(34)]
        X = df[keypoint_columns].values
        y = df["label"].values
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
        classifiers = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "SVM": SVC(kernel='rbf', random_state=42),
            "Neural Network": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        best_classifier = None
        best_accuracy = 0
        best_name = ""
        print("Comparing classifiers...")
        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name} accuracy: {accuracy:.2f}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_classifier = clf
                best_name = name
        print(f"\nBest classifier: {best_name} with accuracy {best_accuracy:.2f}")
        best_classifier.fit(X_train, y_train)
        joblib.dump(best_classifier, model_file)
        joblib.dump(label_encoder, "label_encoder.joblib")
        joblib.dump(scaler, "scaler.joblib")
        reference_keypoints = df.groupby("label")[keypoint_columns].mean().to_dict('index')
        print("Available poses:", list(reference_keypoints.keys()))
        return best_classifier, label_encoder, reference_keypoints, scaler
    except Exception as e:
        print(f"Error training classifier: {e}")
        return None, None, None, None

def extract_keypoints_from_frame(frame, model):
    try:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = np.array(image, dtype=np.uint8)
        image_tensor = tf.convert_to_tensor(image, dtype=tf.uint8)
        image_tensor = tf.expand_dims(image_tensor, axis=0)
        input_image = tf.image.resize_with_pad(image_tensor, 192, 192)
        input_image = tf.cast(input_image, dtype=tf.int32)
        outputs = model(input_image)
        raw_keypoints = outputs["output_0"].numpy()
        if len(raw_keypoints.shape) == 4:
            keypoints = raw_keypoints[0, 0]
        elif len(raw_keypoints.shape) == 3:
            keypoints = raw_keypoints[0]
        else:
            print(f"Unexpected raw keypoints shape: {raw_keypoints.shape}")
            return None, None
        if keypoints.shape != (17, 3):
            print(f"Unexpected keypoints shape: {keypoints.shape}")
            return None, None
        flattened_keypoints = keypoints[:, :2].flatten()
        if len(flattened_keypoints) != 34:
            print(f"Unexpected number of keypoints: {len(flattened_keypoints)}")
            return None, None
        return flattened_keypoints, keypoints
    except Exception as e:
        print(f"Error extracting keypoints: {e}")
        return None, None

def compare_poses(user_keypoints, reference_keypoints, pose_name, threshold=0.1):
    if user_keypoints is None or pose_name not in reference_keypoints:
        return None, "Invalid pose or keypoints"
    ref_keypoints = np.array([reference_keypoints[pose_name][f"kp_{i}"] for i in range(34)])
    differences = user_keypoints - ref_keypoints
    distances = np.sqrt(np.sum(differences**2, axis=0))
    feedback = []
    keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    for i in range(0, 34, 2):
        keypoint_idx = i // 2
        keypoint_name = keypoint_names[keypoint_idx]
        x_diff = differences[i]
        y_diff = differences[i + 1]
        if abs(x_diff) > threshold:
            if x_diff > 0:
                feedback.append(f"Move your {keypoint_name} to the left.")
            else:
                feedback.append(f"Move your {keypoint_name} to the right.")
        if abs(y_diff) > threshold:
            if y_diff > 0:
                feedback.append(f"Lower your {keypoint_name}.")
            else:
                feedback.append(f"Raise your {keypoint_name}.")
    if not feedback:
        feedback.append("Your pose looks good!")
    return distances, feedback

def draw_keypoints(frame, keypoints):
    if keypoints is None:
        return frame
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (0, 5), (0, 6),
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (5, 6),
        (5, 11), (6, 12),
        (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16)
    ]
    height, width = frame.shape[:2]
    scaled_keypoints = keypoints[:, :2] * [width, height]
    for i, (x, y) in enumerate(scaled_keypoints):
        if keypoints[i, 2] > 0.3:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
    for start, end in connections:
        if keypoints[start, 2] > 0.3 and keypoints[end, 2] > 0.3:
            start_point = (int(scaled_keypoints[start, 0]), int(scaled_keypoints[start, 1]))
            end_point = (int(scaled_keypoints[end, 0]), int(scaled_keypoints[end, 1]))
            cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
    return frame
def monitor_pose(classifier, label_encoder, reference_keypoints, scaler, model, target_pose):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    if target_pose not in reference_keypoints:
        print(f"Error: Target pose '{target_pose}' not found in the dataset.")
        print("Available poses:", list(reference_keypoints.keys()))
        return
    print(f"Monitoring for target pose: {target_pose}")
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        user_keypoints, keypoints_with_confidence = extract_keypoints_from_frame(frame, model)
        detected_pose = "Unknown"
        feedback = []
        if user_keypoints is not None:
            user_keypoints_scaled = scaler.transform(user_keypoints.reshape(1, -1))
            predicted_label = classifier.predict(user_keypoints_scaled)[0]
            detected_pose = label_encoder.inverse_transform([predicted_label])[0]
            if detected_pose == target_pose:
                distance, feedback = compare_poses(user_keypoints, reference_keypoints, target_pose)
            else:
                feedback = [f"Please perform the {target_pose} pose. Detected pose: {detected_pose}"]
            for i, msg in enumerate(feedback):
                cv2.putText(frame, msg, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if keypoints_with_confidence is not None:
            frame = draw_keypoints(frame, keypoints_with_confidence)
        cv2.putText(frame, f"Target Pose: {target_pose}", (10, frame.shape[0] - 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Detected Pose: {detected_pose}", (10, frame.shape[0] - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Yoga Pose Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    classifier, label_encoder, reference_keypoints, scaler = load_and_train_classifier()
    if classifier is None or label_encoder is None or reference_keypoints is None or scaler is None:
        print("Failed to train classifier or load reference keypoints. Exiting.")
        exit()
    target_pose = input("Enter the target pose to monitor (e.g., Navasana): ")
    monitor_pose(classifier, label_encoder, reference_keypoints, scaler, model, target_pose)