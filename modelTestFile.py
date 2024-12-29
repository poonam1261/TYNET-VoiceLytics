import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load the trained SVM model and scaler
svm_model = joblib.load("svm_model.pkl")  # Replace with your model path
scaler = joblib.load("scaler.pkl")  # Replace with your scaler path

# Initialize Mediapipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# Function to calculate angles
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])  # x, y coordinates of point A
    b = np.array([b.x, b.y])  # x, y coordinates of point B
    c = np.array([c.x, c.y])  # x, y coordinates of point C

    # Calculate the angle using the cosine rule
    ab = a - b
    bc = c - b
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


# Define function to extract features (angles) from landmarks
def extract_features(landmarks):
    # Calculate biomechanical angles
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    left_foot_index = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]

    # Calculate angles for the left side
    shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
    elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    hip_angle = calculate_angle(left_knee, left_hip, left_shoulder)
    knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    ankle_angle = calculate_angle(left_knee, left_ankle, left_foot_index)

    # Calculate relative angles to the ground
    ground_vector = np.array([1, 0])  # Assumes ground is horizontal
    shoulder_ground_angle = calculate_angle_relative_to_ground(left_shoulder, ground_vector)
    elbow_ground_angle = calculate_angle_relative_to_ground(left_elbow, ground_vector)
    hip_ground_angle = calculate_angle_relative_to_ground(left_hip, ground_vector)
    knee_ground_angle = calculate_angle_relative_to_ground(left_knee, ground_vector)
    ankle_ground_angle = calculate_angle_relative_to_ground(left_ankle, ground_vector)

    # Return features in the same order as training data
    return [
        shoulder_angle, elbow_angle, hip_angle, knee_angle, ankle_angle,
        shoulder_ground_angle, elbow_ground_angle, hip_ground_angle,
        knee_ground_angle, ankle_ground_angle
    ]

# Helper function to calculate angle relative to ground
def calculate_angle_relative_to_ground(landmark, ground_vector):
    point = np.array([landmark.x, landmark.y])
    cosine_angle = np.dot(point, ground_vector) / (np.linalg.norm(point) * np.linalg.norm(ground_vector))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)



def result():
    # Start video capture
    cap = cv2.VideoCapture(0)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Process the frame to detect pose landmarks
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                # Draw the landmarks
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Extract pose landmarks
                landmarks = results.pose_landmarks.landmark

                # Extract features (angles) for the model
                try:
                    features = extract_features(landmarks)

                    # Standardize features
                    features = np.array(features).reshape(1, -1)
                    features = scaler.transform(features)

                    # Predict the exercise type
                    prediction = svm_model.predict(features)
                    predicted_label = prediction[0]  # Get the predicted label

                    # Display the predicted exercise
                    cv2.putText(image, f"Exercise: {predicted_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)
                except Exception as e:
                    cv2.putText(image, "Error in prediction!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display the video feed
            cv2.imshow('Exercise Detection', image)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
