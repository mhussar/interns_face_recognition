import cv2
import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
import numpy as np
import sqlite3
import json
import os # Imported to help with path management

# --- Function to calculate a simple hash/feature for face comparison (NOT Dlib) ---
def get_simple_face_feature(face_image_rgb):
    """Generates a simple feature vector from a face image."""
    # Resize to a standard size and convert to grayscale
    face_image_resized = cv2.resize(face_image_rgb, (64, 64), interpolation=cv2.INTER_AREA)
    face_image_gray = cv2.cvtColor(face_image_resized, cv2.COLOR_RGB2GRAY)
    # Flatten the 2D image array into a 1D vector
    return face_image_gray.flatten()

# --- Function to compare two simple face features ---
def compare_simple_face_features(feature1, feature2, tolerance=6000):
    """
    Compares two face features using Euclidean distance.
    Returns True if the distance is within the tolerance.
    """
    distance = np.linalg.norm(feature1 - feature2)
    return distance < tolerance, distance

# --- Main Application Class ---
class FaceRecognitionApp:
    def __init__(self, window, window_title="OpenCV DNN & Simple Face Recognition App"):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("800x600")
        self.window.resizable(True, True)

        # --- Get the directory of the script to locate model/db files reliably ---
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # --- Camera Setup ---
        try:
            self.vid = cv2.VideoCapture(0) # 0 for default webcam
            if not self.vid.isOpened():
                messagebox.showerror("Camera Error", "Could not open video stream. Please ensure your webcam is connected and not in use by another application.")
                self.window.destroy()
                return
        except Exception as e:
            messagebox.showerror("Camera Init Error", f"Failed to initialize camera: {e}")
            self.window.destroy()
            return

        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # --- OpenCV DNN Face Detector Model Setup ---
        self.prototxt_path = os.path.join(self.script_dir, "deploy.prototxt")
        self.model_path = os.path.join(self.script_dir, "res10_300x300_ssd_iter_140000.caffemodel")

        try:
            self.net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.model_path)
        except cv2.error as e:
            messagebox.showerror("Model Load Error", f"Could not load DNN model: {e}\nPlease ensure 'deploy.prototxt' and 'res10_300x300_ssd_iter_140000.caffemodel' are in the same directory as your script.")
            self.window.destroy()
            return
        
        self.confidence_threshold = 0.5

        # --- Database Setup ---
        # **NOTE**: If the app fails to start silently, your DB file might be corrupt.
        # Delete 'face_data.db' and restart the app to create a fresh one.
        self.db_path = os.path.join(self.script_dir, "face_data.db")
        try:
            self._init_db()
            self._load_known_faces()
        except Exception as e:
            messagebox.showerror("DB Error", f"Failed to initialize or load from database: {e}\nConsider deleting the 'face_data.db' file and restarting.")
            self.window.destroy()
            return

        # --- Variables for registering new faces ---
        self.unknown_face_data_to_register = None 

        # --- UI Elements ---
        self.canvas = tk.Canvas(window, width=self.width, height=self.height, bg="black", bd=0, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)

        self.status_label = tk.Label(window, text="Initializing...", font=("Inter", 12), fg="gray")
        self.status_label.pack(pady=5)

        self.control_frame = tk.Frame(window)
        self.control_frame.pack(pady=10)

        # Register button is created but not shown until an unknown face appears.
        self.register_button = tk.Button(self.control_frame, text="Register New Face", command=self._prompt_for_name,
                                         bg="#4CAF50", fg="white", font=("Inter", 10, "bold"),
                                         relief="raised", bd=2, activebackground="#45a049")
        
        self.btn_quit = tk.Button(self.control_frame, text="Quit", command=self.quit_app,
                                   bg="#ef4444", fg="white", font=("Inter", 10, "bold"),
                                   relief="raised", bd=2, activebackground="#dc2626")
        self.btn_quit.pack(side=tk.RIGHT, padx=5)

        # --- Main Loop ---
        self.delay = 15
        self.update_frame_id = None
        self.update_frame() # Start the video processing loop

        self.window.protocol("WM_DELETE_WINDOW", self.quit_app)

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS known_faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                feature TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

    def _load_known_faces(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name, feature FROM known_faces")
        rows = cursor.fetchall()
        conn.close()

        self.known_face_features = []
        self.known_face_names = []

        for name, feature_str in rows:
            try:
                feature_list = json.loads(feature_str)
                feature_np = np.array(feature_list, dtype=np.uint8)
                self.known_face_features.append(feature_np)
                self.known_face_names.append(name)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Warning: Could not load face data for '{name}'. Skipping. Error: {e}")
        
        print(f"Loaded {len(self.known_face_names)} known faces from database.")

    def update_frame(self):
        ret, frame = self.vid.read()

        if not ret or frame is None:
            self.status_label.config(text="Warning: No frame received from camera.")
            self.update_frame_id = self.window.after(self.delay, self.update_frame) 
            return

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

        self.net.setInput(blob)
        detections = self.net.forward()

        # Reset variables for the current frame
        recognized_names = set()
        faces_detected_count = 0
        unknown_found_in_frame = False
        self.unknown_face_data_to_register = None

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.confidence_threshold:
                faces_detected_count += 1
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Ensure box coordinates are within frame bounds
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w - 1, endX), min(h - 1, endY)

                if startX >= endX or startY >= endY:
                    continue

                face_roi_bgr = frame[startY:endY, startX:endX]
                
                # --- Recognition Logic ---
                face_roi_rgb = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2RGB)
                current_face_feature = get_simple_face_feature(face_roi_rgb)
                
                name = "Unknown"
                best_match_distance = float('inf')

                if self.known_face_features:
                    for idx, known_feature in enumerate(self.known_face_features):
                        is_match, distance = compare_simple_face_features(current_face_feature, known_feature)
                        if is_match and distance < best_match_distance:
                            best_match_distance = distance
                            name = self.known_face_names[idx]
                
                # --- FIX: Correctly handle "Unknown" faces ---
                if name == "Unknown":
                    unknown_found_in_frame = True
                    # Store the feature of the first detected unknown face for registration
                    if self.unknown_face_data_to_register is None:
                        self.unknown_face_data_to_register = current_face_feature
                else:
                    recognized_names.add(name)

                # --- Drawing Logic ---
                # FIX: Use Green for known faces, Orange for unknown
                box_color = (0, 255, 0) if name != "Unknown" else (0, 165, 255)
                text_color = (255, 255, 255)
                
                # Draw bounding box
                cv2.rectangle(frame, (startX, startY), (endX, endY), box_color, 2)
                
                # Draw label with a background
                text = f"{name} ({int(confidence * 100)}%)" if name == "Unknown" else name
                cv2.rectangle(frame, (startX, endY - 25), (endX, endY), box_color, cv2.FILLED)
                cv2.putText(frame, text, (startX + 6, endY - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, text_color, 1)

        # --- Update UI based on what was found in the frame ---
        if unknown_found_in_frame:
            self.register_button.pack(side=tk.LEFT, padx=5)
            self.status_label.config(text=f"Faces detected: {faces_detected_count}. An unknown face is present!")
        else:
            self.register_button.pack_forget()
            if faces_detected_count > 0:
                names_str = ', '.join(sorted(list(recognized_names))) if recognized_names else "None recognized"
                self.status_label.config(text=f"Faces detected: {faces_detected_count}. Recognized: {names_str}")
            else:
                self.status_label.config(text="No faces detected.")

        # Display the frame in the Tkinter canvas
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # Schedule the next frame update
        self.update_frame_id = self.window.after(self.delay, self.update_frame)

    def _prompt_for_name(self):
        if self.unknown_face_data_to_register is None:
            messagebox.showwarning("Registration Error", "No unknown face is currently selected for registration.")
            return

        # Pause the video feed while the dialog is open
        if self.update_frame_id:
            self.window.after_cancel(self.update_frame_id)

        name = simpledialog.askstring("Register Face", "Enter the name for this person:", parent=self.window)
        
        if name and name.strip():
            self._save_face(name.strip(), self.unknown_face_data_to_register)
        else:
            if name is not None: # User entered empty string
                messagebox.showwarning("Input Error", "Name cannot be empty.")
            # If name is None, user clicked "Cancel", so we do nothing.
        
        # Resume the video feed
        self.update_frame()

    def _save_face(self, name, feature):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        feature_str = json.dumps(feature.tolist())

        try:
            cursor.execute("INSERT INTO known_faces (name, feature) VALUES (?, ?)", (name, feature_str))
            conn.commit()
            messagebox.showinfo("Success", f"Face for '{name}' has been registered!")
            self._load_known_faces() # Reload known faces to include the new one
        except sqlite3.IntegrityError:
            messagebox.showerror("Error", f"A face with the name '{name}' already exists.")
        except Exception as e:
            messagebox.showerror("Database Error", f"Failed to save face: {e}")
        finally:
            conn.close()

    def quit_app(self):
        # Cleanly shut down the application
        if self.update_frame_id:
            self.window.after_cancel(self.update_frame_id)
        if self.vid.isOpened():
            self.vid.release()
        self.window.destroy()
        print("Application closed.")

# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = FaceRecognitionApp(root)
        root.mainloop()
    except Exception as e:
        print(f"A fatal error occurred: {e}")