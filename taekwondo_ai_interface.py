import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import cv2
import json
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = Path(__file__).resolve().parent
POSE_MODEL_PATH = BASE_DIR / "models" / "pose_landmarker_full.task"

class TaekwondoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Taekwondo AI Coach")
        self.root.geometry("1380x820")
        self.root.minsize(1120, 720)
        self.root.configure(bg="#213971")

        self.cap = None
        self.camera = None
        self.camera_running = False
        self.playing = False
        self.show_skeleton = True
        self.photo = None

        self.video_path = None
        self.total_frames = 0
        self.fps = 30.0
        self.duration = 0.0
        self.current_frame = 0
        self.updating_timeline = False

        self.pose = None
        self.model_ready = False

        self.build_ui()
        self.init_pose()
        self.load_ai_model()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(15, self.update_stream)

    def build_ui(self):
        header = tk.Frame(self.root, bg="#0b1220", height=74)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="TAEKWONDO AI COACH",
            bg="#0b1220",
            fg="#f8fafc",
            font=("Segoe UI", 21, "bold")
        ).pack(side="left", padx=28, pady=15)

        tk.Label(
            header,
            text="Pose Analysis  •  Live Camera  •  Random Forest",
            bg="#0b1220",
            fg="#94a3b8",
            font=("Segoe UI", 10)
        ).pack(side="left", padx=4, pady=24)

        self.header_status = tk.Label(
            header,
            text="● Interface Ready",
            bg="#0b1220",
            fg="#22c55e",
            font=("Segoe UI", 9, "bold")
        )
        self.header_status.pack(side="right", padx=28)

        main = tk.Frame(self.root, bg="#0f172a")
        main.pack(fill="both", expand=True, padx=20, pady=18)

        left = tk.Frame(main, bg="#0f172a")
        left.pack(side="left", fill="both", expand=True, padx=(0, 14))

        right = tk.Frame(main, bg="#0f172a", width=390)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        card = tk.Frame(left, bg="#1e293b")
        card.pack(fill="both", expand=True)

        tk.Label(
            card,
            text="LIVE / VIDEO ANALYSIS",
            bg="#1e293b",
            fg="#cbd5e1",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", padx=18, pady=(14, 8))

        self.video_label = tk.Label(
            card,
            text="No camera or video selected\n\nPress Start Camera for live pose detection.",
            bg="#080f1c",
            fg="#64748b",
            font=("Segoe UI", 14),
            justify="center"
        )
        self.video_label.pack(fill="both", expand=True, padx=18, pady=(0, 12))

        timeline = tk.Frame(card, bg="#1e293b")
        timeline.pack(fill="x", padx=18)

        self.time_var = tk.StringVar(value="00:00 / 00:00")

        tk.Label(
            timeline,
            textvariable=self.time_var,
            bg="#1e293b",
            fg="#94a3b8",
            font=("Segoe UI", 9)
        ).pack(side="left")

        self.timeline = ttk.Scale(
            timeline,
            from_=0,
            to=100,
            orient="horizontal",
            command=self.on_timeline_move
        )
        self.timeline.pack(side="left", fill="x", expand=True, padx=12)

        self.frame_var = tk.StringVar(value="Frame: —")

        tk.Label(
            timeline,
            textvariable=self.frame_var,
            bg="#1e293b",
            fg="#64748b",
            font=("Segoe UI", 8)
        ).pack(side="right")

        controls = tk.Frame(card, bg="#1e293b")
        controls.pack(fill="x", padx=18, pady=14)

        self.camera_btn = self.button(
            controls,
            "📷 Start Camera",
            self.toggle_camera
        )
        self.camera_btn.pack(side="left", padx=(0, 7))

        self.button(
            controls,
            "Select Video",
            self.select_video
        ).pack(side="left", padx=7)

        self.play_btn = self.button(
            controls,
            "▶ Play",
            self.toggle_play
        )
        self.play_btn.pack(side="left", padx=7)

        self.button(
            controls,
            "Reset",
            self.reset_video
        ).pack(side="left", padx=7)

        self.skeleton_btn = self.button(
            controls,
            "🦴 Skeleton: ON",
            self.toggle_skeleton
        )
        self.skeleton_btn.pack(side="left", padx=7)

        self.analysis_btn = self.button(
            controls,
            "Run AI Analysis",
            self.run_analysis
        )
        self.analysis_btn.pack(side="right")

        self.status_var = tk.StringVar(
            value="Ready — waiting for camera or video"
        )

        tk.Label(
            card,
            textvariable=self.status_var,
            bg="#1e293b",
            fg="#94a3b8",
            font=("Segoe UI", 8)
        ).pack(anchor="w", padx=18, pady=(0, 14))

        self.build_result(right)
        self.build_technique(right)
        self.build_info(right)

    def button(self, parent, text, command):
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg="#2563eb",
            fg="white",
            activebackground="#1d4ed8",
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=13,
            pady=8,
            cursor="hand2",
            font=("Segoe UI", 8, "bold")
        )

    def build_result(self, parent):
        card = tk.Frame(parent, bg="#1e293b")
        card.pack(fill="x", pady=(0, 12))

        tk.Label(
            card,
            text="AI ANALYSIS RESULT",
            bg="#1e293b",
            fg="#cbd5e1",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", padx=18, pady=(15, 12))

        self.action_var = tk.StringVar(value="—")
        self.score_var = tk.StringVar(value="—")
        self.result_var = tk.StringVar(value="Waiting")

        for title, var in [
            ("Detected Action", self.action_var),
            ("AI Score", self.score_var),
            ("Technique Status", self.result_var)
        ]:
            row = tk.Frame(card, bg="#1e293b")
            row.pack(fill="x", padx=18, pady=5)

            tk.Label(
                row,
                text=title,
                bg="#1e293b",
                fg="#94a3b8",
                font=("Segoe UI", 8)
            ).pack(side="left")

            tk.Label(
                row,
                textvariable=var,
                bg="#1e293b",
                fg="#f8fafc",
                font=("Segoe UI", 9, "bold")
            ).pack(side="right")

        self.model_status = tk.Label(
            card,
            text="● Checking AI model...",
            bg="#1e293b",
            fg="#f59e0b",
            font=("Segoe UI", 8, "bold")
        )
        self.model_status.pack(anchor="w", padx=18, pady=16)

    def build_technique(self, parent):
        card = tk.Frame(parent, bg="#1e293b")
        card.pack(fill="x", pady=(0, 12))

        tk.Label(
            card,
            text="TECHNIQUE ANALYSIS",
            bg="#1e293b",
            fg="#cbd5e1",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", padx=18, pady=(15, 10))

        self.technique_vars = {}

        for name in [
            "Body Balance",
            "Leg Position",
            "Hip Rotation",
            "Knee Extension",
            "Upper Body"
        ]:
            row = tk.Frame(card, bg="#1e293b")
            row.pack(fill="x", padx=18, pady=4)

            tk.Label(
                row,
                text=name,
                bg="#1e293b",
                fg="#94a3b8",
                font=("Segoe UI", 8)
            ).pack(side="left")

            var = tk.StringVar(value="Not evaluated")
            self.technique_vars[name] = var

            tk.Label(
                row,
                textvariable=var,
                bg="#1e293b",
                fg="#64748b",
                font=("Segoe UI", 8, "bold")
            ).pack(side="right")

        tk.Label(
            card,
            text="Movement classification is provided by the trained AI model.",
            bg="#1e293b",
            fg="#64748b",
            font=("Segoe UI", 7)
        ).pack(anchor="w", padx=18, pady=(8, 15))

    def build_info(self, parent):
        card = tk.Frame(parent, bg="#1e293b")
        card.pack(fill="both", expand=True)

        tk.Label(
            card,
            text="SESSION INFORMATION",
            bg="#1e293b",
            fg="#cbd5e1",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", padx=18, pady=(15, 10))

        self.info_tree = ttk.Treeview(
            card,
            columns=("value",),
            show="headings",
            height=6
        )

        self.info_tree.heading("value", text="Value")
        self.info_tree.column("value", anchor="w", width=340)
        self.info_tree.pack(fill="x", padx=18)

        tk.Label(
            card,
            text="POSE DETECTION",
            bg="#1e293b",
            fg="#cbd5e1",
            font=("Segoe UI", 9, "bold")
        ).pack(anchor="w", padx=18, pady=(15, 7))

        self.pose_status = tk.Label(
            card,
            text="● Checking MediaPipe...",
            bg="#1e293b",
            fg="#f59e0b",
            font=("Segoe UI", 8, "bold")
        )
        self.pose_status.pack(anchor="w", padx=18)

        self.landmark_var = tk.StringVar(value="—")

        tk.Label(
            card,
            text="Current landmarks:",
            bg="#1e293b",
            fg="#64748b",
            font=("Segoe UI", 8)
        ).pack(anchor="w", padx=18, pady=(10, 3))

        tk.Label(
            card,
            textvariable=self.landmark_var,
            bg="#1e293b",
            fg="#e2e8f0",
            font=("Segoe UI", 9, "bold")
        ).pack(anchor="w", padx=18)

        self.camera_status = tk.Label(
            card,
            text="● Camera: OFF",
            bg="#1e293b",
            fg="#64748b",
            font=("Segoe UI", 8, "bold")
        )
        self.camera_status.pack(anchor="w", padx=18, pady=15)

    def init_pose(self):
        try:
            if not POSE_MODEL_PATH.exists():
                self.pose_status.config(
                    text="● Pose model not found",
                    fg="#ef4444"
                )
                return

            base_options = python.BaseOptions(
                model_asset_path=str(POSE_MODEL_PATH)
            )

            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )

            self.pose = vision.PoseLandmarker.create_from_options(options)

            self.pose_status.config(
                text="● MediaPipe Pose ready",
                fg="#22c55e"
            )

        except Exception as e:
            self.pose = None
            self.pose_status.config(
                text="● MediaPipe initialization failed",
                fg="#ef4444"
            )
            self.status_var.set(
                f"MediaPipe error: {type(e).__name__}"
            )

    def load_ai_model(self):
        try:
            model_path = BASE_DIR / "taekwondo_rf_final.pkl"
            features_path = BASE_DIR / "selected_features.npy"
            classes_path = BASE_DIR / "class_names.json"

            if not model_path.exists():
                raise FileNotFoundError("taekwondo_rf_final.pkl")

            if not features_path.exists():
                raise FileNotFoundError("selected_features.npy")

            if not classes_path.exists():
                raise FileNotFoundError("class_names.json")

            with open(classes_path, "r", encoding="utf-8") as f:
                json.load(f)

            np.load(features_path)

            self.model_ready = True

            self.model_status.config(
                text="● Random Forest model connected",
                fg="#22c55e"
            )

        except Exception as e:
            self.model_ready = False

            self.model_status.config(
                text="● AI model not connected",
                fg="#ef4444"
            )

            self.status_var.set(
                f"Model error: {type(e).__name__}"
            )

    def toggle_camera(self):
        if self.camera_running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        if self.pose is None:
            messagebox.showerror(
                "MediaPipe",
                "MediaPipe Pose is not available."
            )
            return

        self.playing = False
        self.play_btn.config(text="▶ Play")

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not self.camera.isOpened():
            self.camera.release()
            self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():
            self.camera = None

            messagebox.showerror(
                "Camera Error",
                "Could not open the laptop camera."
            )
            return

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.camera_running = True

        self.camera_btn.config(text="⏹ Stop Camera")
        self.camera_status.config(
            text="● Camera: ON",
            fg="#22c55e"
        )

        self.header_status.config(
            text="● Live Camera",
            fg="#22c55e"
        )

        self.status_var.set(
            "Live camera active — MediaPipe is tracking"
        )

        self.action_var.set("Live mode")
        self.score_var.set("—")
        self.result_var.set("Pose tracking")

        self.set_session_info()

    def stop_camera(self):
        self.camera_running = False

        if self.camera is not None:
            self.camera.release()
            self.camera = None

        self.camera_btn.config(text="📷 Start Camera")

        self.camera_status.config(
            text="● Camera: OFF",
            fg="#64748b"
        )

        self.header_status.config(
            text="● Interface Ready",
            fg="#22c55e"
        )

        self.status_var.set("Camera stopped")
        self.landmark_var.set("—")

        self.set_session_info()

        self.video_label.config(
            image="",
            text="Camera stopped\n\nPress Start Camera to begin live pose detection."
        )

    def update_stream(self):
        if self.camera_running and self.camera is not None:
            ok, frame = self.camera.read()

            if ok:
                frame = cv2.flip(frame, 1)

                processed, count = self.process_pose(frame)

                self.display_frame(processed)

                self.landmark_var.set(
                    str(count) if count is not None else "0"
                )

                self.time_var.set("LIVE")
                self.frame_var.set("Live Camera")

                self.set_session_info()

        elif self.playing:
            self.play_video_frame()

        self.root.after(15, self.update_stream)

    def process_pose(self, frame):
        if not self.show_skeleton or self.pose is None:
            return frame, None

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = self.pose.detect(mp_image)

            if not result.pose_landmarks:
                return frame, 0

            landmarks = result.pose_landmarks[0]

            connections = [
                (0, 1), (1, 2), (2, 3), (3, 7),
                (0, 4), (4, 5), (5, 6), (6, 8),
                (9, 10),
                (11, 12),
                (11, 13), (13, 15),
                (12, 14), (14, 16),
                (11, 23), (12, 24),
                (23, 24),
                (23, 25), (25, 27),
                (27, 29), (29, 31),
                (24, 26), (26, 28),
                (28, 30), (30, 32),
                (15, 17), (15, 19), (15, 21),
                (16, 18), (16, 20), (16, 22)
            ]

            h, w = frame.shape[:2]

            for start, end in connections:
                p1 = landmarks[start]
                p2 = landmarks[end]

                x1 = int(p1.x * w)
                y1 = int(p1.y * h)
                x2 = int(p2.x * w)
                y2 = int(p2.y * h)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (50, 220, 120),
                    2
                )

            for landmark in landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    3,
                    (70, 140, 255),
                    -1
                )

            return frame, len(landmarks)

        except Exception:
            return frame, 0

    def toggle_skeleton(self):
        self.show_skeleton = not self.show_skeleton

        self.skeleton_btn.config(
            text="🦴 Skeleton: ON"
            if self.show_skeleton
            else "🦴 Skeleton: OFF"
        )

    def select_video(self):
        path = filedialog.askopenfilename(
            title="Select Taekwondo Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        self.stop_camera()
        self.release_video()

        self.cap = cv2.VideoCapture(path)

        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None

            messagebox.showerror(
                "Video Error",
                "Could not open the selected video."
            )
            return

        self.video_path = path

        self.total_frames = int(
            self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        self.fps = self.cap.get(
            cv2.CAP_PROP_FPS
        ) or 30.0

        self.duration = (
            self.total_frames / self.fps
            if self.fps > 0
            else 0
        )

        self.current_frame = 0

        self.timeline.config(
            from_=0,
            to=max(0, self.total_frames - 1)
        )

        self.timeline.set(0)

        self.set_session_info()
        self.reset_results()

        self.header_status.config(
            text="● Video Loaded",
            fg="#22c55e"
        )

        self.status_var.set(
            f"Loaded: {Path(path).name}"
        )

        self.show_frame(0)

    def show_frame(self, frame_number):
        if not self.cap or self.camera_running:
            return

        frame_number = int(
            max(
                0,
                min(
                    frame_number,
                    max(0, self.total_frames - 1)
                )
            )
        )

        self.cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number
        )

        ok, frame = self.cap.read()

        if not ok:
            return

        self.current_frame = frame_number

        processed, count = self.process_pose(frame)

        self.display_frame(processed)

        self.updating_timeline = True
        self.timeline.set(frame_number)
        self.updating_timeline = False

        self.update_time_labels()

        self.landmark_var.set(
            str(count)
            if count is not None
            else "Unavailable"
        )

        self.set_session_info()

    def toggle_play(self):
        if not self.cap:
            messagebox.showinfo(
                "Video",
                "Please select a video first."
            )
            return

        if self.camera_running:
            messagebox.showinfo(
                "Camera Active",
                "Stop the camera before playing a video."
            )
            return

        self.playing = not self.playing

        self.play_btn.config(
            text="⏸ Pause"
            if self.playing
            else "▶ Play"
        )

        self.status_var.set(
            "Playing video..."
            if self.playing
            else "Paused"
        )

    def play_video_frame(self):
        if not self.playing or not self.cap:
            return

        ok, frame = self.cap.read()

        if not ok:
            self.playing = False

            self.play_btn.config(
                text="▶ Play"
            )

            self.status_var.set(
                "Video finished"
            )

            self.cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                0
            )

            return

        self.current_frame = max(
            0,
            int(
                self.cap.get(
                    cv2.CAP_PROP_POS_FRAMES
                )
            ) - 1
        )

        processed, count = self.process_pose(frame)

        self.display_frame(processed)

        self.landmark_var.set(
            str(count)
            if count is not None
            else "Unavailable"
        )

        self.updating_timeline = True
        self.timeline.set(self.current_frame)
        self.updating_timeline = False

        self.update_time_labels()

    def on_timeline_move(self, value):
        if (
            self.updating_timeline
            or not self.cap
            or self.camera_running
        ):
            return

        try:
            self.show_frame(
                int(float(value))
            )
        except ValueError:
            pass

    def update_time_labels(self):
        current = (
            self.current_frame / self.fps
            if self.fps > 0
            else 0
        )

        self.time_var.set(
            f"{self.format_time(current)} / "
            f"{self.format_time(self.duration)}"
        )

        self.frame_var.set(
            f"Frame: {self.current_frame}/"
            f"{max(0, self.total_frames - 1)}"
        )

    def reset_video(self):
        if not self.cap or self.camera_running:
            return

        self.playing = False

        self.play_btn.config(
            text="▶ Play"
        )

        self.show_frame(0)
        self.reset_results()

        self.status_var.set(
            "Video reset"
        )

    def run_analysis(self):
        if self.camera_running:
            messagebox.showinfo(
                "AI Analysis",
                "Video classification is currently available for selected video files."
            )
            return

        if not self.video_path:
            messagebox.showinfo(
                "Analysis",
                "Please select a video first."
            )
            return

        if not self.model_ready:
            messagebox.showerror(
                "AI Model",
                "The trained AI model could not be loaded."
            )
            return

        self.analysis_btn.config(
            state="disabled",
            text="Analyzing..."
        )

        self.status_var.set(
            "AI is analyzing the video..."
        )

        self.header_status.config(
            text="● AI Analysis",
            fg="#f59e0b"
        )

        self.root.update_idletasks()

        try:
            from predict_video import predict

            result = predict(
                str(self.video_path)
            )

            if not isinstance(result, dict):
                raise ValueError(
                    "Prediction result must be a dictionary."
                )

            action = result.get(
                "class",
                "Unknown"
            )

            confidence = float(
                result.get(
                    "confidence",
                    0.0
                )
            )

            self.action_var.set(
                action.replace("_", " ").title()
            )

            self.score_var.set(
                f"{confidence:.2f}%"
            )

            if confidence >= 70:
                status = "High confidence"
            elif confidence >= 50:
                status = "Moderate confidence"
            else:
                status = "Prediction completed"

            self.result_var.set(status)

            technique = result.get("technique", {})

            for name, var in self.technique_vars.items():
                data = technique.get(name)

                if isinstance(data, dict):
                    status = data.get("status", "Not evaluated")
                    score = data.get("score")

                    if score is not None:
                        var.set(f"{status} ({float(score):.1f})")
                    else:
                        var.set(status)
                else:
                    var.set("Not evaluated")

            self.model_status.config(
                text="● Random Forest model connected",
                fg="#22c55e"
            )

            self.status_var.set(
                f"Detected: {action.replace('_', ' ').title()}"
            )

            self.header_status.config(
                text="● Analysis Complete",
                fg="#22c55e"
            )

        except Exception as e:
            self.action_var.set("Error")
            self.score_var.set("—")
            self.result_var.set("Failed")

            self.status_var.set(
                f"Analysis error: {type(e).__name__}"
            )

            self.header_status.config(
                text="● Analysis Error",
                fg="#ef4444"
            )

            messagebox.showerror(
                "AI Analysis Error",
                f"{type(e).__name__}: {e}"
            )

        finally:
            self.analysis_btn.config(
                state="normal",
                text="Run AI Analysis"
            )

    def reset_results(self):
        self.action_var.set("—")
        self.score_var.set("—")
        self.result_var.set("Waiting")

        if self.model_ready:
            self.model_status.config(
                text="● Random Forest model connected",
                fg="#22c55e"
            )
        else:
            self.model_status.config(
                text="● AI model not connected",
                fg="#ef4444"
            )

        for var in self.technique_vars.values():
            var.set("Not evaluated")

        self.landmark_var.set("—")

    def display_frame(self, frame):
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        width = max(
            self.video_label.winfo_width(),
            600
        )

        height = max(
            self.video_label.winfo_height(),
            430
        )

        h, w = rgb.shape[:2]

        scale = min(
            width / w,
            height / h
        )

        resized = cv2.resize(
            rgb,
            (
                max(1, int(w * scale)),
                max(1, int(h * scale))
            ),
            interpolation=cv2.INTER_AREA
        )

        self.photo = ImageTk.PhotoImage(
            Image.fromarray(resized)
        )

        self.video_label.config(
            image=self.photo,
            text=""
        )

    def set_session_info(self):
        if not hasattr(
            self,
            "info_tree"
        ):
            return

        for item in self.info_tree.get_children():
            self.info_tree.delete(item)

        if self.camera_running:
            rows = [
                "Mode: Live Camera",
                "Camera: Webcam 0",
                "FPS: Live",
                "Frames: Live",
                "Current frame: Live"
            ]

        elif not self.cap or not self.video_path:
            rows = [
                "Mode: None",
                "Video: Not selected",
                "Frames: —",
                "FPS: —",
                "Duration: —"
            ]

        else:
            rows = [
                "Mode: Video File",
                f"Video: {Path(self.video_path).name}",
                f"Frames: {self.total_frames}",
                f"FPS: {self.fps:.1f}",
                f"Duration: {self.format_time(self.duration)}"
            ]

        for row in rows:
            self.info_tree.insert(
                "",
                "end",
                values=(row,)
            )

    def format_time(self, seconds):
        seconds = max(
            0,
            int(seconds)
        )

        m, s = divmod(
            seconds,
            60
        )

        h, m = divmod(
            m,
            60
        )

        if h:
            return f"{h:02d}:{m:02d}:{s:02d}"

        return f"{m:02d}:{s:02d}"

    def release_video(self):
        self.playing = False

        self.play_btn.config(
            text="▶ Play"
        )

        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def on_close(self):
        self.camera_running = False

        if self.camera is not None:
            self.camera.release()
            self.camera = None

        self.release_video()

        if self.pose is not None:
            try:
                self.pose.close()
            except Exception:
                pass

        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = TaekwondoApp(root)
    root.mainloop()