import os
import json
import cv2
import joblib
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "taekwondo_rf_final.pkl"
)

FEATURES_PATH = os.path.join(
    BASE_DIR,
    "selected_features.npy"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "class_names.json"
)

POSE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "pose_landmarker_full.task"
)

NUM_FRAMES = 30
NUM_LANDMARKS = 33


LANDMARKS = {
    "right_hip": 24,
    "left_hip": 23,
    "right_knee": 26,
    "left_knee": 25,
    "right_ankle": 28,
    "left_ankle": 27,
    "right_elbow": 14,
    "left_elbow": 13,
    "right_wrist": 16,
    "left_wrist": 15
}


def extract_pose(video_path):

    base_options = python.BaseOptions(
        model_asset_path=POSE_MODEL_PATH
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )

    landmarker = vision.PoseLandmarker.create_from_options(
        options
    )

    cap = cv2.VideoCapture(video_path)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    frame_indices = np.linspace(
        0,
        max(total_frames - 1, 0),
        NUM_FRAMES
    ).astype(int)

    poses = []

    for index in frame_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(index)
        )

        ret, frame = cap.read()

        if not ret:
            continue

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        timestamp_ms = int(
            index * 1000 / fps
        )

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        if (
            result.pose_landmarks
            and len(result.pose_landmarks) > 0
        ):

            landmarks = result.pose_landmarks[0]

            values = []

            for landmark in landmarks:

                values.extend(
                    [
                        landmark.x,
                        landmark.y,
                        landmark.z,
                        landmark.visibility
                    ]
                )

            poses.append(values)

    cap.release()
    landmarker.close()

    if len(poses) == 0:
        raise RuntimeError(
            "No pose detected in video."
        )

    poses = np.array(
        poses,
        dtype=np.float32
    )

    if len(poses) < NUM_FRAMES:

        indices = np.linspace(
            0,
            len(poses) - 1,
            NUM_FRAMES
        ).astype(int)

        poses = poses[indices]

    elif len(poses) > NUM_FRAMES:

        indices = np.linspace(
            0,
            len(poses) - 1,
            NUM_FRAMES
        ).astype(int)

        poses = poses[indices]

    print(
        f"Detected frames: {len(poses)}"
    )

    return poses


def normalize_pose(pose_data):

    data = pose_data.copy()

    positions = data[:, :99].reshape(
        NUM_FRAMES,
        NUM_LANDMARKS,
        3
    )

    visibility = data[:, 99:].reshape(
        NUM_FRAMES,
        NUM_LANDMARKS
    )

    left_hip = positions[:, 23]

    right_hip = positions[:, 24]

    center = (
        left_hip +
        right_hip
    ) / 2

    positions = (
        positions -
        center[:, None, :]
    )

    left_shoulder = positions[:, 11]

    right_shoulder = positions[:, 12]

    shoulder_distance = np.linalg.norm(
        left_shoulder -
        right_shoulder,
        axis=1
    )

    shoulder_distance[
        shoulder_distance < 1e-6
    ] = 1.0

    positions = (
        positions /
        shoulder_distance[:, None, None]
    )

    normalized = np.concatenate(
        [
            positions.reshape(
                NUM_FRAMES,
                99
            ),
            visibility
        ],
        axis=1
    )

    return normalized.astype(
        np.float32
    )


def calculate_angle(a, b, c):

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(
        ba,
        axis=-1
    )

    norm_bc = np.linalg.norm(
        bc,
        axis=-1
    )

    denominator = (
        norm_ba *
        norm_bc
    )

    denominator[
        denominator < 1e-8
    ] = 1e-8

    cosine = np.sum(
        ba * bc,
        axis=-1
    ) / denominator

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    return np.degrees(
        np.arccos(cosine)
    )


def create_temporal_features(normalized):

    positions = normalized[
        :, :99
    ].reshape(
        NUM_FRAMES,
        NUM_LANDMARKS,
        3
    )

    feature_names = []

    features = []

    for landmark_name, landmark_index in LANDMARKS.items():

        for axis_index, axis_name in enumerate(
            ["x", "y", "z"]
        ):

            values = positions[
                :,
                landmark_index,
                axis_index
            ]

            velocity = np.diff(
                values
            )

            acceleration = np.diff(
                velocity
            )

            if len(velocity) == 0:
                velocity = np.array([0.0])

            if len(acceleration) == 0:
                acceleration = np.array([0.0])

            direction_changes = np.sum(
                np.diff(
                    np.sign(
                        velocity
                    )
                ) != 0
            )

            current_features = [
                np.mean(values),
                np.std(values),
                np.min(values),
                np.max(values),
                np.max(values) - np.min(values),
                np.argmin(values),
                np.argmax(values),
                np.mean(
                    np.abs(velocity)
                ),
                np.max(
                    np.abs(velocity)
                ),
                np.mean(
                    np.abs(acceleration)
                ),
                np.max(
                    np.abs(acceleration)
                ),
                direction_changes
            ]

            features.extend(
                current_features
            )

            feature_names.extend(
                [
                    f"mean_{landmark_name}_{axis_name}",
                    f"std_{landmark_name}_{axis_name}",
                    f"min_{landmark_name}_{axis_name}",
                    f"max_{landmark_name}_{axis_name}",
                    f"range_{landmark_name}_{axis_name}",
                    f"min_frame_{landmark_name}_{axis_name}",
                    f"max_frame_{landmark_name}_{axis_name}",
                    f"mean_velocity_{landmark_name}_{axis_name}",
                    f"max_velocity_{landmark_name}_{axis_name}",
                    f"mean_acceleration_{landmark_name}_{axis_name}",
                    f"max_acceleration_{landmark_name}_{axis_name}",
                    f"direction_changes_{landmark_name}_{axis_name}"
                ]
            )

    return np.array(
        features,
        dtype=np.float32
    )


def analyze_technique(normalized):

    positions = normalized[
        :, :99
    ].reshape(
        NUM_FRAMES,
        NUM_LANDMARKS,
        3
    )

    right_hip = positions[:, 24]
    left_hip = positions[:, 23]

    right_knee = positions[:, 26]
    left_knee = positions[:, 25]

    right_ankle = positions[:, 28]
    left_ankle = positions[:, 27]

    right_shoulder = positions[:, 12]
    left_shoulder = positions[:, 11]

    right_elbow = positions[:, 14]
    left_elbow = positions[:, 13]

    right_wrist = positions[:, 16]
    left_wrist = positions[:, 15]

    hip_center = (
        right_hip +
        left_hip
    ) / 2

    shoulder_center = (
        right_shoulder +
        left_shoulder
    ) / 2

    ankle_distance = np.linalg.norm(
        right_ankle -
        left_ankle,
        axis=1
    )

    hip_width = np.linalg.norm(
        right_hip -
        left_hip,
        axis=1
    )

    shoulder_width = np.linalg.norm(
        right_shoulder -
        left_shoulder,
        axis=1
    )

    hand_distance = np.linalg.norm(
        right_wrist -
        left_wrist,
        axis=1
    )

    right_knee_angle = calculate_angle(
        right_hip,
        right_knee,
        right_ankle
    )

    left_knee_angle = calculate_angle(
        left_hip,
        left_knee,
        left_ankle
    )

    right_elbow_angle = calculate_angle(
        right_shoulder,
        right_elbow,
        right_wrist
    )

    left_elbow_angle = calculate_angle(
        left_shoulder,
        left_elbow,
        left_wrist
    )

    body_center_movement = np.linalg.norm(
        np.diff(
            hip_center,
            axis=0
        ),
        axis=1
    )

    shoulder_movement = np.linalg.norm(
        np.diff(
            shoulder_center,
            axis=0
        ),
        axis=1
    )

    hip_rotation = np.abs(
        np.degrees(
            np.arctan2(
                right_hip[:, 2] -
                left_hip[:, 2],
                right_hip[:, 0] -
                left_hip[:, 0]
            )
        )
    )

    shoulder_rotation = np.abs(
        np.degrees(
            np.arctan2(
                right_shoulder[:, 2] -
                left_shoulder[:, 2],
                right_shoulder[:, 0] -
                left_shoulder[:, 0]
            )
        )
    )

    balance_variation = np.std(
        hip_center,
        axis=0
    )

    balance_value = np.linalg.norm(
        balance_variation
    )

    balance_score = np.clip(
        100 -
        balance_value * 80,
        0,
        100
    )

    ankle_variation = np.std(
        ankle_distance
    )

    leg_score = np.clip(
        100 -
        ankle_variation * 80,
        0,
        100
    )

    max_knee_angle = max(
        np.max(right_knee_angle),
        np.max(left_knee_angle)
    )

    knee_score = np.clip(
        (
            max_knee_angle -
            100
        ) / 80 * 100,
        0,
        100
    )

    rotation_value = np.mean(
        np.abs(
            hip_rotation -
            shoulder_rotation
        )
    )

    hip_score = np.clip(
        rotation_value * 2.5,
        0,
        100
    )

    upper_movement = np.mean(
        shoulder_movement
    )

    upper_score = np.clip(
        100 -
        upper_movement * 100,
        0,
        100
    )

    scores = {
        "Body Balance": float(
            balance_score
        ),
        "Leg Position": float(
            leg_score
        ),
        "Hip Rotation": float(
            hip_score
        ),
        "Knee Extension": float(
            knee_score
        ),
        "Upper Body": float(
            upper_score
        )
    }

    result = {}

    for name, score in scores.items():

        if score >= 75:
            status = "Good"

        elif score >= 50:
            status = "Moderate"

        else:
            status = "Needs Improvement"

        result[name] = {
            "score": score,
            "status": status
        }

    return result


def predict(video_path):

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(
            f"Selected features not found: {FEATURES_PATH}"
        )

    if not os.path.exists(CLASS_NAMES_PATH):
        raise FileNotFoundError(
            f"Class names not found: {CLASS_NAMES_PATH}"
        )

    if not os.path.exists(POSE_MODEL_PATH):
        raise FileNotFoundError(
            f"Pose model not found: {POSE_MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    selected_features = np.load(FEATURES_PATH)

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        class_names = json.load(file)

    print("\nClass names:")
    print(class_names)

    pose_data = extract_pose(video_path)

    normalized = normalize_pose(pose_data)

    temporal_features = create_temporal_features(normalized)

    print(
        f"Feature shape: {temporal_features.shape}"
    )

    selected = temporal_features[selected_features]

    selected = selected.reshape(1, -1)

    print(
        f"Selected feature shape: {selected.shape}"
    )

    prediction = model.predict(selected)[0]

    probabilities = model.predict_proba(selected)[0]

    top_indices = np.argsort(probabilities)[::-1][:5]

    print("\nTop 5 predictions:")

    for probability_index in top_indices:

        class_index = model.classes_[probability_index]

        if isinstance(class_names, dict):
            class_name = class_names[str(int(class_index))]
        else:
            class_name = class_names[int(class_index)]

        probability = probabilities[probability_index] * 100

        print(
            f"{class_name}: {probability:.2f}%"
        )

    if isinstance(class_names, dict):
        predicted_class = class_names[str(int(prediction))]
    else:
        predicted_class = class_names[int(prediction)]

    confidence = float(np.max(probabilities))

    print(f"\nPrediction: {predicted_class}")
    print(f"Confidence: {confidence * 100:.2f}%")

    technique = analyze_technique(normalized)

    print("\nTechnique Analysis:")

    for name, data in technique.items():

        print(
            f"{name}: "
            f"{data['status']} "
            f"({data['score']:.1f})"
        )

    return {
        "class": predicted_class,
        "confidence": confidence * 100,
        "technique": technique
    }


if __name__ == "__main__":

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()

    video_path = filedialog.askopenfilename(
        title="Select Taekwondo Video",
        filetypes=[
            ("MP4 Videos", "*.mp4"),
            ("All Files", "*.*")
        ]
    )

    root.destroy()

    if not video_path:
        print("No video selected.")

    else:
        print(
            f"\nSelected video: {video_path}"
        )

        result = predict(
            video_path
        )

        print(
            "\nFinal Result:"
        )

        print(
            f"Action: {result['class']}"
        )

        print(
            f"Confidence: "
            f"{result['confidence']:.2f}%"
        )

        print(
            "\nTechnique Analysis:"
        )

        for name, data in result["technique"].items():

            print(
                f"{name}: "
                f"{data['status']} "
                f"({data['score']:.1f})"
            )