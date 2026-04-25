# main.py
"""
FACE RECOGNITION ATTENDANCE SYSTEM
====================================
Session-aware attendance system.

HOW SESSIONS WORK:
  - Every app launch asks the teacher to either:
      A) Create a NEW session (new subject/lecture)
      B) Resume an ACTIVE session (to add late students)
  - All attendance is tagged to the current session_id
  - "View Records" shows ONLY the current session
  - A session stays editable until its timer runs out
  - Duplicate detection is per-session (not per-day)
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import threading
import os
import json
import cv2
from PIL import Image, ImageTk
import numpy as np
from datetime import datetime, timedelta

import modules.database as db
from modules.dataset_creator import create_dataset
from modules.trainer import train_recognizer


# ════════════════════════════════════════════════════════════════
#  SESSION SETUP DIALOG — shown at every app launch
# ════════════════════════════════════════════════════════════════

class SessionSetupDialog(tk.Toplevel):
    """
    Modal dialog shown when the app starts.
    Teacher either creates a new session or resumes an active one.
    Sets self.result to the chosen session_id, or None if cancelled.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Session Setup")
        self.geometry("520x480")
        self.configure(bg="#0d1117")
        self.resizable(False, False)
        self.grab_set()           # Make it modal (blocks parent window)
        self.result = None        # Will hold chosen session_id

        self._build_ui()
        self._load_active_sessions()

        # Centre dialog over parent
        self.transient(parent)
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")

    def _build_ui(self):
        # Header
        tk.Label(self, text="📚  Session Setup",
                 font=("Helvetica", 16, "bold"),
                 fg="white", bg="#0d1117").pack(pady=(20, 5))
        tk.Label(self, text="Configure this attendance session before starting.",
                 font=("Helvetica", 9), fg="#8b949e", bg="#0d1117").pack()

        tk.Frame(self, bg="#30363d", height=1).pack(fill=tk.X, padx=20, pady=15)

        # ── CREATE NEW SESSION ────────────────────────────────
        new_frame = tk.LabelFrame(self, text="  Create New Session  ",
                                   font=("Helvetica", 10, "bold"),
                                   fg="#58a6ff", bg="#0d1117", bd=1)
        new_frame.pack(fill=tk.X, padx=20, pady=5)

        row1 = tk.Frame(new_frame, bg="#0d1117")
        row1.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(row1, text="Subject:", fg="#e6edf3", bg="#0d1117",
                 font=("Helvetica", 10), width=12, anchor=tk.W).pack(side=tk.LEFT)
        self.subject_var = tk.StringVar()
        tk.Entry(row1, textvariable=self.subject_var,
                 font=("Helvetica", 10), width=22,
                 bg="#161b22", fg="white", insertbackground="white",
                 relief=tk.FLAT).pack(side=tk.LEFT, padx=5)

        row2 = tk.Frame(new_frame, bg="#0d1117")
        row2.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(row2, text="Teacher:", fg="#e6edf3", bg="#0d1117",
                 font=("Helvetica", 10), width=12, anchor=tk.W).pack(side=tk.LEFT)
        self.teacher_var = tk.StringVar()
        tk.Entry(row2, textvariable=self.teacher_var,
                 font=("Helvetica", 10), width=22,
                 bg="#161b22", fg="white", insertbackground="white",
                 relief=tk.FLAT).pack(side=tk.LEFT, padx=5)

        row3 = tk.Frame(new_frame, bg="#0d1117")
        row3.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(row3, text="Duration (min):", fg="#e6edf3", bg="#0d1117",
                 font=("Helvetica", 10), width=12, anchor=tk.W).pack(side=tk.LEFT)
        self.duration_var = tk.StringVar(value="90")
        tk.Entry(row3, textvariable=self.duration_var,
                 font=("Helvetica", 10), width=8,
                 bg="#161b22", fg="white", insertbackground="white",
                 relief=tk.FLAT).pack(side=tk.LEFT, padx=5)
        tk.Label(row3, text="(e.g. 90 for 1.5 hours)",
                 fg="#8b949e", bg="#0d1117", font=("Helvetica", 9)).pack(side=tk.LEFT)

        tk.Button(new_frame, text="▶  Start New Session",
                  command=self._create_new,
                  bg="#238636", fg="white",
                  font=("Helvetica", 10, "bold"),
                  relief=tk.FLAT, cursor="hand2", pady=6
                  ).pack(fill=tk.X, padx=10, pady=8)

        tk.Frame(self, bg="#30363d", height=1).pack(fill=tk.X, padx=20, pady=10)

        # ── RESUME ACTIVE SESSION ─────────────────────────────
        resume_frame = tk.LabelFrame(self, text="  Resume Active Session (Add Late Students)  ",
                                      font=("Helvetica", 10, "bold"),
                                      fg="#f0883e", bg="#0d1117", bd=1)
        resume_frame.pack(fill=tk.X, padx=20, pady=5)

        self.active_listbox = tk.Listbox(resume_frame, height=4,
                                          bg="#161b22", fg="#e6edf3",
                                          font=("Courier", 9),
                                          selectbackground="#1f6feb",
                                          relief=tk.FLAT)
        self.active_listbox.pack(fill=tk.X, padx=10, pady=5)

        tk.Button(resume_frame, text="↩  Resume Selected Session",
                  command=self._resume_selected,
                  bg="#b45309", fg="white",
                  font=("Helvetica", 10, "bold"),
                  relief=tk.FLAT, cursor="hand2", pady=6
                  ).pack(fill=tk.X, padx=10, pady=(0, 8))

    def _load_active_sessions(self):
        """Populate the listbox with currently active sessions."""
        self._active_sessions = db.get_active_sessions()
        self.active_listbox.delete(0, tk.END)

        if not self._active_sessions:
            self.active_listbox.insert(tk.END, "  No active sessions found.")
        else:
            for s in self._active_sessions:
                expires = datetime.strptime(s["expires_at"], "%Y-%m-%d %H:%M:%S")
                mins_left = max(0, int((expires - datetime.now()).total_seconds() / 60))
                line = (f"  [{s['session_id']}] {s['subject']} — "
                        f"{s['teacher']}  ({mins_left} min left)")
                self.active_listbox.insert(tk.END, line)

    def _create_new(self):
        subject = self.subject_var.get().strip()
        teacher = self.teacher_var.get().strip()
        dur_str = self.duration_var.get().strip()

        if not subject:
            messagebox.showwarning("Required", "Please enter subject name.", parent=self)
            return
        if not teacher:
            messagebox.showwarning("Required", "Please enter teacher name.", parent=self)
            return
        if not dur_str.isdigit() or int(dur_str) < 1:
            messagebox.showwarning("Required", "Duration must be a positive number.", parent=self)
            return

        session_id = db.create_session(subject, teacher, int(dur_str))
        self.result = session_id
        self.destroy()

    def _resume_selected(self):
        if not self._active_sessions:
            messagebox.showinfo("None", "No active sessions to resume.", parent=self)
            return
        sel = self.active_listbox.curselection()
        if not sel:
            messagebox.showwarning("Select", "Please select a session from the list.", parent=self)
            return
        idx = sel[0]
        self.result = self._active_sessions[idx]["session_id"]
        self.destroy()


# ════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ════════════════════════════════════════════════════════════════

class AttendanceApp:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("960x680")
        self.root.configure(bg="#0d1117")
        self.root.resizable(True, True)

        self.webcam_running = False
        self.current_session_id = None
        self.current_session = None
        self._timer_after_id = None   # for the countdown ticker

        db.init_db()

        self._build_ui()

        # Show session setup dialog immediately
        self.root.after(200, self._show_session_setup)

    # ── UI CONSTRUCTION ──────────────────────────────────────────

    def _build_ui(self):
        self._build_header()
        self._build_session_banner()

        content = tk.Frame(self.root, bg="#0d1117")
        content.pack(fill=tk.BOTH, expand=True, padx=15, pady=8)

        self._build_left_panel(content)
        self._build_right_panel(content)
        self._build_status_bar()

    def _build_header(self):
        hf = tk.Frame(self.root, bg="#161b22", pady=12)
        hf.pack(fill=tk.X)
        tk.Label(hf, text="🎓  Face Recognition Attendance System",
                 font=("Helvetica", 18, "bold"), fg="white",
                 bg="#161b22").pack()
        tk.Label(hf, text="OpenCV · LBPH · SQLite · Session-Based",
                 font=("Helvetica", 9), fg="#8b949e",
                 bg="#161b22").pack()

    def _build_session_banner(self):
        """Green/orange banner showing current session info + countdown."""
        self.banner_frame = tk.Frame(self.root, bg="#0d1117", pady=0)
        self.banner_frame.pack(fill=tk.X, padx=15, pady=(6, 0))

        self.banner_label = tk.Label(
            self.banner_frame,
            text="⚪  No session loaded — waiting for setup...",
            font=("Helvetica", 10, "bold"),
            fg="#8b949e", bg="#1c2128",
            pady=6, padx=12, anchor=tk.W
        )
        self.banner_label.pack(fill=tk.X)

    def _build_left_panel(self, parent):
        lf = tk.Frame(parent, bg="#161b22", width=290)
        lf.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        lf.pack_propagate(False)

        # ── Registration ─────────────────────────────────────
        s1 = tk.LabelFrame(lf, text="  Student Registration  ",
                            font=("Helvetica", 10, "bold"),
                            fg="#58a6ff", bg="#161b22", bd=1)
        s1.pack(fill=tk.X, padx=8, pady=8)

        tk.Label(s1, text="Student Name:", fg="#e6edf3", bg="#161b22",
                 font=("Helvetica", 9)).pack(anchor=tk.W, padx=8, pady=(6,0))
        self.name_var = tk.StringVar()
        tk.Entry(s1, textvariable=self.name_var, font=("Helvetica", 10),
                 bg="#0d1117", fg="white", insertbackground="white",
                 relief=tk.FLAT).pack(fill=tk.X, padx=8, pady=3)

        tk.Label(s1, text="Student ID (number):", fg="#e6edf3", bg="#161b22",
                 font=("Helvetica", 9)).pack(anchor=tk.W, padx=8)
        self.id_var = tk.StringVar()
        tk.Entry(s1, textvariable=self.id_var, font=("Helvetica", 10),
                 bg="#0d1117", fg="white", insertbackground="white",
                 relief=tk.FLAT).pack(fill=tk.X, padx=8, pady=3)

        tk.Button(s1, text="📷  Capture Dataset (30 images)",
                  command=self._start_dataset_capture,
                  bg="#1f6feb", fg="white", font=("Helvetica", 9, "bold"),
                  relief=tk.FLAT, cursor="hand2", pady=6
                  ).pack(fill=tk.X, padx=8, pady=6)

        # ── Training ─────────────────────────────────────────
        s2 = tk.LabelFrame(lf, text="  Model Training  ",
                            font=("Helvetica", 10, "bold"),
                            fg="#58a6ff", bg="#161b22", bd=1)
        s2.pack(fill=tk.X, padx=8, pady=4)

        tk.Button(s2, text="🧠  Train Recognizer",
                  command=self._start_training,
                  bg="#1f6feb", fg="white", font=("Helvetica", 9, "bold"),
                  relief=tk.FLAT, cursor="hand2", pady=6
                  ).pack(fill=tk.X, padx=8, pady=6)

        # ── Attendance ───────────────────────────────────────
        s3 = tk.LabelFrame(lf, text="  Attendance  ",
                            font=("Helvetica", 10, "bold"),
                            fg="#58a6ff", bg="#161b22", bd=1)
        s3.pack(fill=tk.X, padx=8, pady=4)

        self.attendance_btn = tk.Button(
            s3, text="▶  Start Attendance",
            command=self._toggle_attendance,
            bg="#238636", fg="white",
            font=("Helvetica", 10, "bold"),
            relief=tk.FLAT, cursor="hand2", pady=8)
        self.attendance_btn.pack(fill=tk.X, padx=8, pady=4)

        tk.Button(s3, text="📋  View Session Records",
                  command=self._show_session_records,
                  bg="#1f6feb", fg="white",
                  font=("Helvetica", 9, "bold"),
                  relief=tk.FLAT, cursor="hand2", pady=6
                  ).pack(fill=tk.X, padx=8, pady=(0, 6))

        tk.Button(s3, text="🔄  New Session Setup",
                  command=self._show_session_setup,
                  bg="#30363d", fg="#e6edf3",
                  font=("Helvetica", 9),
                  relief=tk.FLAT, cursor="hand2", pady=5
                  ).pack(fill=tk.X, padx=8, pady=(0, 6))

        # ── Registered Students ──────────────────────────────
        s4 = tk.LabelFrame(lf, text="  Registered Students  ",
                            font=("Helvetica", 10, "bold"),
                            fg="#58a6ff", bg="#161b22", bd=1)
        s4.pack(fill=tk.X, padx=8, pady=4)

        self.student_count_var = tk.StringVar(value="No students registered yet.")
        tk.Label(s4, textvariable=self.student_count_var,
                 fg="#8b949e", bg="#161b22",
                 font=("Helvetica", 9)).pack(padx=8, pady=(6, 2))

        tk.Button(s4, text="\U0001f465  View & Manage Students",
                  command=self._show_students_window,
                  bg="#1f6feb", fg="white",
                  font=("Helvetica", 9, "bold"),
                  relief=tk.FLAT, cursor="hand2", pady=6
                  ).pack(fill=tk.X, padx=8, pady=(2, 8))

        self._refresh_student_list()

    def _build_right_panel(self, parent):
        rf = tk.Frame(parent, bg="#161b22")
        rf.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.camera_label = tk.Label(
            rf,
            text="📷\n\nCamera feed appears here\nwhen attendance is started",
            bg="#0d1117", fg="#8b949e",
            font=("Helvetica", 12))
        self.camera_label.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        info = tk.Frame(rf, bg="#161b22")
        info.pack(fill=tk.X, padx=6, pady=(0, 4))

        self.last_recognized_var = tk.StringVar(value="Last recognised: —")
        tk.Label(info, textvariable=self.last_recognized_var,
                 fg="#e6edf3", bg="#161b22",
                 font=("Helvetica", 10)).pack(side=tk.LEFT)

        self.marked_count_var = tk.StringVar(value="Present: 0")
        tk.Label(info, textvariable=self.marked_count_var,
                 fg="#3fb950", bg="#161b22",
                 font=("Helvetica", 10, "bold")).pack(side=tk.RIGHT)

    def _build_status_bar(self):
        sf = tk.Frame(self.root, bg="#161b22", pady=3)
        sf.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_var = tk.StringVar(value="Waiting for session setup...")
        tk.Label(sf, textvariable=self.status_var,
                 fg="#8b949e", bg="#161b22",
                 font=("Helvetica", 8), anchor=tk.W
                 ).pack(fill=tk.X, padx=10)

    # ── HELPERS ──────────────────────────────────────────────────

    def _set_status(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.status_var.set(f"[{ts}]  {msg}")

    def _refresh_student_list(self):
        """Update the student count label in the sidebar."""
        students = db.get_all_students()
        if students:
            self.student_count_var.set(f"{len(students)} student(s) registered.")
        else:
            self.student_count_var.set("No students registered yet.")

    def _update_banner(self):
        """Update the session banner text and start/stop countdown ticker."""
        if not self.current_session:
            self.banner_label.config(
                text="⚪  No session loaded — click 'New Session Setup'",
                fg="#8b949e", bg="#1c2128")
            return

        s = self.current_session
        secs = db.get_seconds_remaining(s["session_id"])

        if secs > 0:
            mins, sec = divmod(secs, 60)
            hrs, mins = divmod(mins, 60)
            time_str = f"{hrs:02d}:{mins:02d}:{sec:02d}"
            text = (f"🟢  Session #{s['session_id']} │ {s['subject']} │ "
                    f"{s['teacher']} │ ⏱ {time_str} remaining")
            self.banner_label.config(text=text, fg="#3fb950", bg="#0d2911")

            # Schedule next tick in 1 second
            self._timer_after_id = self.root.after(1000, self._update_banner)
        else:
            text = (f"🔴  Session #{s['session_id']} │ {s['subject']} │ "
                    f"{s['teacher']} │ EXPIRED — start new session")
            self.banner_label.config(text=text, fg="#f85149", bg="#2d0f0f")

    # ── SESSION SETUP ─────────────────────────────────────────────

    def _show_session_setup(self):
        """Show the session setup dialog."""
        # Stop any running ticker
        if self._timer_after_id:
            self.root.after_cancel(self._timer_after_id)

        dialog = SessionSetupDialog(self.root)
        self.root.wait_window(dialog)

        if dialog.result is None:
            self._set_status("Session setup cancelled.")
            return

        session_id = dialog.result
        self.current_session_id = session_id
        self.current_session = db.get_session(session_id)

        s = self.current_session
        self._set_status(
            f"✓ Session #{session_id} loaded: {s['subject']} by {s['teacher']}")
        self._update_banner()

        # Update the present count for this session
        present = db.get_attendance_for_session(session_id)
        self.marked_count_var.set(f"Present: {len(present)}")

    # ── DATASET CAPTURE ──────────────────────────────────────────

    def _start_dataset_capture(self):
        name = self.name_var.get().strip()
        id_str = self.id_var.get().strip()

        if not name:
            messagebox.showwarning("Input Required", "Please enter student name.")
            return
        if not id_str or not id_str.isdigit():
            messagebox.showwarning("Input Required", "Please enter a valid numeric ID.")
            return

        self._set_status(f"Starting dataset capture for {name}...")
        threading.Thread(
            target=self._capture_thread,
            args=(int(id_str), name),
            daemon=True
        ).start()

    def _capture_thread(self, user_id, name):
        success = create_dataset(user_id, name)
        if success:
            self.root.after(0, lambda: self._set_status(
                f"✓ Dataset captured for {name}. Train the model next."))
            self.root.after(0, self._refresh_student_list)
            self.root.after(0, lambda: messagebox.showinfo(
                "Done", f"30 images captured for {name}!\nPlease train the model."))
        else:
            self.root.after(0, lambda: self._set_status(f"Capture failed for {name}."))

    # ── TRAINING ─────────────────────────────────────────────────

    def _start_training(self):
        self._set_status("Training model…")
        threading.Thread(target=self._training_thread, daemon=True).start()

    def _training_thread(self):
        success = train_recognizer()
        if success:
            self.root.after(0, lambda: self._set_status(
                "✓ Model trained. Ready for attendance."))
            self.root.after(0, lambda: messagebox.showinfo(
                "Done", "Model trained successfully!"))
        else:
            self.root.after(0, lambda: self._set_status("Training failed."))

    # ── ATTENDANCE ───────────────────────────────────────────────

    def _toggle_attendance(self):
        if self.current_session_id is None:
            messagebox.showwarning("No Session",
                "Please set up a session first using 'New Session Setup'.")
            return

        if not db.is_session_active(self.current_session_id):
            messagebox.showwarning("Session Expired",
                "This session has expired.\n"
                "Click 'New Session Setup' to create or resume a session.")
            return

        if not self.webcam_running:
            self.webcam_running = True
            self.attendance_btn.config(text="⏹  Stop Attendance", bg="#da3633")
            self._set_status("Attendance running…")
            threading.Thread(target=self._attendance_thread, daemon=True).start()
        else:
            self.webcam_running = False
            self.attendance_btn.config(text="▶  Start Attendance", bg="#238636")
            self._set_status("Attendance stopped.")

    def _attendance_thread(self):
        """
        Core recognition loop.
        Marks attendance against self.current_session_id ONLY.
        All errors are caught and reported — no more silent failures.
        """
        try:
            # ── Step 1: Check trainer exists ─────────────────
            trainer_path = os.path.join("trainer", "trainer.yml")
            if not os.path.exists(trainer_path):
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", "No trained model found.\nPlease click 'Train Recognizer' first."))
                self.webcam_running = False
                self.root.after(0, lambda: self.attendance_btn.config(
                    text="▶  Start Attendance", bg="#238636"))
                return

            # ── Step 2: Check names.json exists ──────────────
            if not os.path.exists("names.json"):
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", "names.json not found.\nPlease register students first."))
                self.webcam_running = False
                self.root.after(0, lambda: self.attendance_btn.config(
                    text="▶  Start Attendance", bg="#238636"))
                return

            # ── Step 3: Load recognizer ───────────────────────
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(trainer_path)
            print("✓ Recognizer loaded")

            # ── Step 4: Load names ────────────────────────────
            with open("names.json", "r") as f:
                content = f.read().strip()
            names = json.loads(content) if content else {}
            print(f"✓ Names loaded: {names}")

            # ── Step 5: Load face cascade ─────────────────────
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            if face_cascade.empty():
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", "Haar Cascade failed to load."))
                self.webcam_running = False
                self.root.after(0, lambda: self.attendance_btn.config(
                    text="▶  Start Attendance", bg="#238636"))
                return

            # ── Step 6: Snapshot session data into local vars ─
            # IMPORTANT: Never access self.current_session inside the loop.
            # It's a dict on the main thread. Copy what we need NOW.
            session_id = self.current_session_id
            session_subject = self.current_session["subject"]
            print(f"✓ Session: #{session_id} — {session_subject}")

            # ── Step 7: Load who is already marked ───────────
            marked_this_run = db.get_marked_names_for_session(session_id)
            print(f"✓ Already marked in this session: {marked_this_run}")

            # ── Step 8: Open webcam ───────────────────────────
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                self.root.after(0, lambda: messagebox.showerror(
                    "Webcam Error", "Cannot open webcam.\nCheck it is connected and not used by another app."))
                self.webcam_running = False
                self.root.after(0, lambda: self.attendance_btn.config(
                    text="▶  Start Attendance", bg="#238636"))
                return
            print("✓ Webcam opened")

            CONFIDENCE_THRESHOLD = 70
            fail_count = 0          # consecutive frame failures
            MAX_FAILS = 30          # stop if 30 consecutive failures

            # ── Step 9: Main loop ─────────────────────────────
            while self.webcam_running:

                # Safety: stop if session expired
                if not db.is_session_active(session_id):
                    self.root.after(0, lambda: messagebox.showinfo(
                        "Session Expired",
                        "The session timer has run out.\nAttendance stopped automatically."))
                    break

                ret, frame = cap.read()

                if not ret:
                    fail_count += 1
                    if fail_count >= MAX_FAILS:
                        self.root.after(0, lambda: messagebox.showerror(
                            "Webcam Error",
                            "Camera stopped sending frames.\n"
                            "Check your webcam and try again."))
                        break
                    continue

                fail_count = 0   # reset on successful frame

                # Preprocess
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_eq = cv2.equalizeHist(gray)

                # Detect faces
                faces = face_cascade.detectMultiScale(
                    gray_eq, scaleFactor=1.1, minNeighbors=5,
                    minSize=(80, 80))

                # Process each face
                for (x, y, w, h) in faces:
                    face_crop = cv2.resize(gray_eq[y:y+h, x:x+w], (200, 200))
                    label, confidence = recognizer.predict(face_crop)

                    if confidence < CONFIDENCE_THRESHOLD:
                        user_name = names.get(str(label), f"ID_{label}")

                        # Try to mark — returns True only if NEW (not duplicate)
                        newly_marked = db.mark_attendance(session_id, user_name, label)

                        if newly_marked:
                            marked_this_run.add(user_name)
                            present_count = len(marked_this_run)
                            self.root.after(0, lambda n=user_name:
                                self.last_recognized_var.set(f"Last: {n} ✓"))
                            self.root.after(0, lambda c=present_count:
                                self.marked_count_var.set(f"Present: {c}"))
                            print(f"✓ MARKED: {user_name}  confidence={confidence:.1f}")

                        # Green = newly marked, Orange = already marked this session
                        color = (0, 200, 80) if newly_marked or user_name not in marked_this_run else (0, 165, 255)
                        tag = "✓" if user_name in marked_this_run else ""
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, f"{user_name} {tag} ({confidence:.0f})",
                                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.65, color, 2)
                    else:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (80, 80, 220), 2)
                        cv2.putText(frame, "Unknown",
                                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.65, (80, 80, 220), 2)

                # HUD overlay — use local session_subject (not self.current_session)
                secs_left = db.get_seconds_remaining(session_id)
                mm, ss = divmod(secs_left, 60)
                hh, mm = divmod(mm, 60)

                cv2.putText(frame, f"Subject: {session_subject}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"Timer: {hh:02d}:{mm:02d}:{ss:02d}",
                            (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 220, 255), 1)
                cv2.putText(frame, f"Present: {len(marked_this_run)}",
                            (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 150), 1)
                cv2.putText(frame, datetime.now().strftime("%H:%M:%S"),
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                # Send to GUI
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (590, 450))
                pil_img = Image.fromarray(frame_resized)
                tk_img = ImageTk.PhotoImage(pil_img)
                self.root.after(0, lambda i=tk_img: self._set_camera_frame(i))

        except Exception as e:
            # Catch ANY unexpected error and show it — no more silent failures
            import traceback
            err_msg = traceback.format_exc()
            print(f"ATTENDANCE THREAD ERROR:\n{err_msg}")
            self.root.after(0, lambda: messagebox.showerror(
                "Unexpected Error",
                f"Attendance stopped due to an error:\n\n{e}\n\n"
                "Check the terminal for full details."))

        finally:
            # Always runs — cleans up even if exception occurred
            try:
                cap.release()
            except Exception:
                pass
            self.webcam_running = False
            self.root.after(0, lambda: self.attendance_btn.config(
                text="▶  Start Attendance", bg="#238636"))
            self.root.after(0, lambda: self.camera_label.config(
                text="📷\n\nCamera stopped.", image="", bg="#0d1117"))

    def _set_camera_frame(self, tk_img):
        self.camera_label.config(image=tk_img, text="", bg="#0d1117")
        self.camera_label.image = tk_img   # prevent garbage collection

    # ── VIEW RECORDS ─────────────────────────────────────────────

    def _show_students_window(self):
        """
        Opens a dedicated window listing all registered students.
        Each student row has a Delete button.
        Deleting removes them from names.json AND their dataset folder,
        then retrains the model automatically so recognition stays accurate.
        """
        win = tk.Toplevel(self.root)
        win.title("Registered Students")
        win.geometry("560x480")
        win.configure(bg="#0d1117")
        win.grab_set()

        # ── Header ───────────────────────────────────────────
        tk.Label(win, text="👥  Registered Students",
                 font=("Helvetica", 14, "bold"),
                 fg="white", bg="#0d1117").pack(pady=(16, 4))
        tk.Label(win,
                 text="Click Delete to remove a student. "
                      "This also deletes their dataset and retrains the model.",
                 font=("Helvetica", 9), fg="#8b949e",
                 bg="#0d1117", wraplength=500).pack(pady=(0, 10))

        tk.Frame(win, bg="#30363d", height=1).pack(fill=tk.X, padx=15)

        # ── Scrollable student list ───────────────────────────
        canvas_frame = tk.Frame(win, bg="#0d1117")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        canvas = tk.Canvas(canvas_frame, bg="#0d1117",
                           highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL,
                                   command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = tk.Frame(canvas, bg="#0d1117")
        inner_window = canvas.create_window((0, 0), window=inner,
                                             anchor=tk.NW)

        def _on_frame_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        def _on_canvas_configure(e):
            canvas.itemconfig(inner_window, width=e.width)
        inner.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        # Column headers
        hdr = tk.Frame(inner, bg="#161b22")
        hdr.pack(fill=tk.X, pady=(0, 4))
        tk.Label(hdr, text="  ID", width=6, anchor=tk.W,
                 fg="#8b949e", bg="#161b22",
                 font=("Helvetica", 9, "bold")).pack(side=tk.LEFT)
        tk.Label(hdr, text="Name", width=24, anchor=tk.W,
                 fg="#8b949e", bg="#161b22",
                 font=("Helvetica", 9, "bold")).pack(side=tk.LEFT)
        tk.Label(hdr, text="Dataset Images", width=16, anchor=tk.W,
                 fg="#8b949e", bg="#161b22",
                 font=("Helvetica", 9, "bold")).pack(side=tk.LEFT)
        tk.Label(hdr, text="Action", width=10, anchor=tk.W,
                 fg="#8b949e", bg="#161b22",
                 font=("Helvetica", 9, "bold")).pack(side=tk.LEFT)

        tk.Frame(inner, bg="#30363d", height=1).pack(fill=tk.X, pady=2)

        def _build_rows():
            # Clear all existing rows (rebuild after delete)
            for widget in inner.winfo_children()[2:]:  # keep header + divider
                widget.destroy()

            students = db.get_all_students()
            if not students:
                tk.Label(inner, text="  No students registered yet.",
                         fg="#8b949e", bg="#0d1117",
                         font=("Helvetica", 10)).pack(pady=20)
                return

            for id_str, name in sorted(students.items(),
                                        key=lambda x: int(x[0])):
                # Count dataset images for this student
                folder = os.path.join("dataset", f"user.{id_str}.{name}")
                if os.path.isdir(folder):
                    img_count = len([f for f in os.listdir(folder)
                                     if f.lower().endswith(('.jpg','.png'))])
                else:
                    img_count = 0

                row = tk.Frame(inner, bg="#161b22", pady=4)
                row.pack(fill=tk.X, pady=2)

                tk.Label(row, text=f"  {id_str}", width=6, anchor=tk.W,
                         fg="#58a6ff", bg="#161b22",
                         font=("Courier", 10)).pack(side=tk.LEFT)
                tk.Label(row, text=name, width=24, anchor=tk.W,
                         fg="#e6edf3", bg="#161b22",
                         font=("Helvetica", 10)).pack(side=tk.LEFT)
                tk.Label(row, text=f"{img_count} images", width=16,
                         anchor=tk.W, fg="#8b949e", bg="#161b22",
                         font=("Helvetica", 9)).pack(side=tk.LEFT)

                def _delete(sid=id_str, sname=name):
                    msg_text = (
                        f"Delete student '{sname}' (ID {sid})?\n\n"
                        "This will:\n"
                        "  - Remove them from names.json\n"
                        "  - Delete their dataset folder\n"
                        "  - Retrain the model automatically\n\n"
                        "Their past attendance records are kept."
                    )
                    confirm = messagebox.askyesno(
                        "Confirm Delete", msg_text, parent=win)
                    if not confirm:
                        return

                    success, msg = db.delete_student(sid, sname)
                    if success:
                        self._set_status(
                            f"✓ Deleted {sname}. Retraining model...")
                        self._refresh_student_list()
                        _build_rows()   # refresh the window rows

                        # Retrain in background so GUI stays responsive
                        def _retrain():
                            students_left = db.get_all_students()
                            if students_left:
                                train_recognizer()
                                self.root.after(0, lambda: self._set_status(
                                    f"✓ {sname} deleted. Model retrained."))
                            else:
                                # No students left — remove stale model
                                import shutil
                                if os.path.exists(
                                        os.path.join("trainer","trainer.yml")):
                                    os.remove(
                                        os.path.join("trainer","trainer.yml"))
                                self.root.after(0, lambda: self._set_status(
                                    "All students deleted. Model removed."))
                        threading.Thread(target=_retrain,
                                         daemon=True).start()
                    else:
                        messagebox.showerror("Error", msg, parent=win)

                tk.Button(row, text="🗑 Delete",
                          command=_delete,
                          bg="#da3633", fg="white",
                          font=("Helvetica", 8, "bold"),
                          relief=tk.FLAT, cursor="hand2",
                          padx=6, pady=2).pack(side=tk.LEFT)

        _build_rows()

        # ── Footer ────────────────────────────────────────────
        tk.Frame(win, bg="#30363d", height=1).pack(fill=tk.X, padx=15)
        tk.Button(win, text="Close",
                  command=win.destroy,
                  bg="#30363d", fg="white",
                  font=("Helvetica", 10),
                  relief=tk.FLAT, cursor="hand2",
                  pady=6).pack(pady=10)

    def _show_session_records(self):
        """Show present + absent students for the CURRENT session only."""
        if self.current_session_id is None:
            messagebox.showinfo("No Session", "No session is loaded.")
            return

        s = self.current_session
        session_id = self.current_session_id

        win = tk.Toplevel(self.root)
        win.title(f"Records — {s['subject']}")
        win.geometry("620x540")
        win.configure(bg="#0d1117")

        # Header
        tk.Label(win,
                 text=f"Session #{session_id}  │  {s['subject']}  │  {s['teacher']}",
                 font=("Helvetica", 12, "bold"), fg="white",
                 bg="#0d1117").pack(pady=(14, 2))
        tk.Label(win,
                 text=f"Date: {s['date']}   Started: {s['start_time']}   "
                      f"Duration: {s['duration_min']} min",
                 font=("Helvetica", 9), fg="#8b949e",
                 bg="#0d1117").pack()

        tk.Frame(win, bg="#30363d", height=1).pack(fill=tk.X, padx=15, pady=10)

        # Notebook with Present / Absent tabs
        nb = ttk.Notebook(win)
        nb.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        # ── Present tab ──────────────────────────────────────
        present_frame = tk.Frame(nb, bg="#0d1117")
        nb.add(present_frame, text="  ✅  Present  ")

        cols = ("Name", "Marked At")
        present_tree = ttk.Treeview(present_frame, columns=cols,
                                     show="headings", height=14)
        present_tree.heading("Name", text="Student Name")
        present_tree.heading("Marked At", text="Time Marked")
        present_tree.column("Name", width=280)
        present_tree.column("Marked At", width=180)
        present_tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        records = db.get_attendance_for_session(session_id)
        for r in records:
            present_tree.insert("", tk.END,
                                values=(r["name"], r["marked_at"]))

        # ── Absent tab ───────────────────────────────────────
        absent_frame = tk.Frame(nb, bg="#0d1117")
        nb.add(absent_frame, text="  ❌  Absent  ")

        absent_list = tk.Listbox(absent_frame,
                                  bg="#0d1117", fg="#f85149",
                                  font=("Helvetica", 11),
                                  selectbackground="#1f6feb",
                                  relief=tk.FLAT)
        absent_list.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        absent = db.get_absent_students(session_id)
        if absent:
            for name in absent:
                absent_list.insert(tk.END, f"  {name}")
        else:
            absent_list.insert(tk.END, "  All registered students are present!")

        # Footer counts
        footer = tk.Frame(win, bg="#161b22")
        footer.pack(fill=tk.X, padx=15, pady=6)

        all_count = len(db.get_all_students())
        present_count = len(records)
        absent_count = len(absent)

        tk.Label(footer, text=f"✅  Present: {present_count}",
                 fg="#3fb950", bg="#161b22",
                 font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=10)
        tk.Label(footer, text=f"❌  Absent: {absent_count}",
                 fg="#f85149", bg="#161b22",
                 font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=10)
        tk.Label(footer, text=f"👥  Total Registered: {all_count}",
                 fg="#8b949e", bg="#161b22",
                 font=("Helvetica", 10)).pack(side=tk.RIGHT, padx=10)


# ════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════

def main():
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()