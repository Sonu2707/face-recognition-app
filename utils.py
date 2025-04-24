import cv2
import face_recognition
import sqlite3
import os
import numpy as np
from datetime import datetime
import logging

def initialize_database():
    """Initialize SQLite database for known faces and recognition history."""
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS known_faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            encoding BLOB NOT NULL,
            image_path TEXT,
            created_at TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS recognition_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            confidence REAL,
            timestamp TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def load_known_faces():
    """Load known face encodings and names from the database."""
    known_face_encodings = []
    known_face_names = []
    
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT name, encoding FROM known_faces")
    for name, encoding_blob in c.fetchall():
        encoding = np.frombuffer(encoding_blob, dtype=np.float64)
        known_face_names.append(name)
        known_face_encodings.append(encoding)
    conn.close()
    
    return known_face_encodings, known_face_names

def add_known_face(image_path, name):
    """Add a new known face to the database."""
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    
    if not encodings:
        raise ValueError("No face found in the image")
    
    encoding = encodings[0]
    encoding_blob = encoding.tobytes()
    
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO known_faces (name, encoding, image_path, created_at) VALUES (?, ?, ?, ?)",
        (name, encoding_blob, image_path, datetime.now())
    )
    conn.commit()
    conn.close()
    
    logging.info(f"Added known face to database: {name}")

def recognize_face(image, known_face_encodings, known_face_names):
    """Recognize faces in the image with confidence scores."""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    recognized_names = []
    confidences = []
    
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        confidence = 0.0
        
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]
        
        recognized_names.append(name)
        confidences.append(confidence)
        
        # Log to recognition history
        c.execute(
            "INSERT INTO recognition_history (name, confidence, timestamp) VALUES (?, ?, ?)",
            (name, confidence, datetime.now())
        )
        
        # Draw rectangle and label
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, f"{name} ({confidence:.2f})", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    conn.commit()
    conn.close()
    
    return image, recognized_names, confidences
