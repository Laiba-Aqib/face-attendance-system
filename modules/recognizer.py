# modules/recognizer.py
"""
REAL-TIME RECOGNITION & ATTENDANCE MODULE
==========================================
Purpose: Recognize faces live from webcam and mark attendance.

Flow:
  1. Load trained model (trainer.yml)
  2. Load names lookup (names.json)
  3. Load today's attendance (to prevent duplicates)
  4. Open webcam
  5. For each frame:
     a. Detect faces (Haar Cascade)
     b. For each detected face:
        - Crop and preprocess
        - Ask LBPH recognizer: "Who is this?"
        - If confidence good enough: mark attendance
     c. Display results on screen
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime
from modules.database import AttendanceDB

def run_attendance_system():
    """Main attendance recognition loop."""
    
    # ─────────────────────────────────────────────────────────────
    # STEP 1: Load the trained model
    # ─────────────────────────────────────────────────────────────
    trainer_path = os.path.join("trainer", "trainer.yml")
    
    if not os.path.exists(trainer_path):
        print("ERROR: No trained model found!")
        print("Please train the model first (run trainer.py)")
        return
    
    # Create recognizer and load saved model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(trainer_path)
    
    print("✓ Trained model loaded")
    
    # ─────────────────────────────────────────────────────────────
    # STEP 2: Load names dictionary
    # Maps user IDs to names: {1: "Alice", 2: "Bob", ...}
    # ─────────────────────────────────────────────────────────────
    if not os.path.exists("names.json"):
        print("ERROR: names.json not found!")
        return
    
    with open("names.json", 'r') as f:
        names = json.load(f)
    
    print(f"✓ Loaded {len(names)} student names")
    
    # ─────────────────────────────────────────────────────────────
    # STEP 3: Load Haar Cascade
    # ─────────────────────────────────────────────────────────────
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    if face_cascade.empty():
        print("ERROR: Haar Cascade failed to load")
        return
    
    # ─────────────────────────────────────────────────────────────
    # STEP 4: Initialize attendance database
    # ─────────────────────────────────────────────────────────────
    db = AttendanceDB()
    
    # Set for tracking who has been marked today in this session
    # Using a set means O(1) lookup — instant check for duplicates
    marked_today = set()
    
    # Pre-load today's attendance to handle app restarts
    today = datetime.now().strftime("%Y-%m-%d")
    already_marked = db.get_names_marked_today(today)
    marked_today.update(already_marked)
    
    if marked_today:
        print(f"✓ Loaded {len(marked_today)} existing attendance records for today")
    
    # ─────────────────────────────────────────────────────────────
    # STEP 5: Open webcam
    # ─────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return
    
    
    
    # ─────────────────────────────────────────────────────────────
    # RECOGNITION PARAMETERS
    # ─────────────────────────────────────────────────────────────
    
    # Confidence threshold — tune based on your testing
    # LBPH confidence = Chi-Square distance (lower = more similar)
    # < 50: High confidence match
    # 50-80: Moderate confidence (might be wrong)
    # > 80: Low confidence (treat as unknown)
    CONFIDENCE_THRESHOLD = 70
    
    # How long to show "Attendance Marked" message
    # We use a simple timer: record time of last marking
    last_marked_display = {}  # {user_id: timestamp}
    DISPLAY_DURATION = 3.0    # Show message for 3 seconds
    
    print("\n" + "="*50)
    print("ATTENDANCE SYSTEM RUNNING")
    print("Press 'q' to quit")
    print("="*50)
    
    # ─────────────────────────────────────────────────────────────
    # STEP 6: Main recognition loop
    # ─────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Frame capture failed")
            break
        
        # ─────────────────────────────────────────────────────────
        # Preprocess frame for detection
        # ─────────────────────────────────────────────────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        
        # ─────────────────────────────────────────────────────────
        # Detect all faces in this frame
        # ─────────────────────────────────────────────────────────
        faces = face_cascade.detectMultiScale(
            gray_eq,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)   # Reasonable minimum for recognition
                                # Too small = bad recognition quality
        )
        
        current_time = datetime.now()
        
        # ─────────────────────────────────────────────────────────
        # Process each detected face
        # ─────────────────────────────────────────────────────────
        for (x, y, w, h) in faces:
            
            # ─────────────────────────────────────────────────────
            # Crop face from GRAYSCALE image
            # CRITICAL: Use gray_eq (equalized) — same preprocessing
            # as training images. Consistency is essential for LBPH!
            # If training used equalized grayscale, testing must too.
            # ─────────────────────────────────────────────────────
            face_crop = gray_eq[y:y+h, x:x+w]
            
            # Resize to EXACT same dimensions used during training
            face_resized = cv2.resize(face_crop, (200, 200))
            
            # ─────────────────────────────────────────────────────
            # ASK THE RECOGNIZER: "Who is this face?"
            # 
            # recognizer.predict() returns:
            # - label: the user ID (integer) of the best match
            # - confidence: Chi-Square distance (float, lower = better)
            # 
            # How it works internally:
            # 1. Compute LBP histogram of face_resized
            # 2. Compare to ALL stored histograms using Chi-Square
            # 3. Return label and distance of closest match
            # ─────────────────────────────────────────────────────
            label, confidence = recognizer.predict(face_resized)
            
            # ─────────────────────────────────────────────────────
            # DETERMINE IDENTITY based on confidence threshold
            # ─────────────────────────────────────────────────────
            if confidence < CONFIDENCE_THRESHOLD:
                # RECOGNIZED — confidence is good
                user_name = names.get(str(label), f"Unknown_ID_{label}")
                
                # Choose color for display
                if user_name in marked_today:
                    color = (255, 165, 0)  # Orange = already marked
                else:
                    color = (0, 255, 0)    # Green = recognized, not yet marked
                
                # ─────────────────────────────────────────────────
                # MARK ATTENDANCE if not already marked today
                # ─────────────────────────────────────────────────
                if user_name not in marked_today:
                    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Record in database
                    db.mark_attendance(user_name, label, timestamp)
                    
                    # Add to in-memory set to prevent duplicates
                    marked_today.add(user_name)
                    
                    # Record time for display purposes
                    last_marked_display[user_name] = current_time.timestamp()
                    
                    print(f"✓ ATTENDANCE MARKED: {user_name} at {timestamp}")
                
                # ─────────────────────────────────────────────────
                # Display label
                # ─────────────────────────────────────────────────
                # Show confidence rounded to 1 decimal place
                conf_display = f"{confidence:.1f}"
                label_text = f"{user_name} ({conf_display})"
                
            else:
                # UNKNOWN — confidence too low to be sure
                label_text = "Unknown"
                color = (0, 0, 255)  # Red = unknown
            
            # ─────────────────────────────────────────────────────
            # DRAW RESULTS ON FRAME
            # ─────────────────────────────────────────────────────
            
            # Rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Background rectangle for text (improves readability)
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, 
                         (x, y-30), 
                         (x + text_size[0], y), 
                         color, -1)  # -1 thickness = filled rectangle
            
            # Name text
            cv2.putText(frame, label_text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show "MARKED!" if recently marked
            if confidence < CONFIDENCE_THRESHOLD and user_name in last_marked_display:
                elapsed = current_time.timestamp() - last_marked_display.get(user_name, 0)
                if elapsed < DISPLAY_DURATION:
                    cv2.putText(frame, "ATTENDANCE MARKED!", (x, y+h+25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # ─────────────────────────────────────────────────────────
        # Display status information on frame
        # ─────────────────────────────────────────────────────────
        
        # Current time
        time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, time_str, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Count of marked attendance
        cv2.putText(frame, f"Marked: {len(marked_today)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # Number of faces in current frame
        cv2.putText(frame, f"Faces detected: {len(faces)}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        cv2.imshow("Attendance System - Press 'q' to quit", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # ─────────────────────────────────────────────────────────────
    # CLEANUP
    # ─────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nSession ended. Total marked today: {len(marked_today)}")
    print("Attendance records saved.")