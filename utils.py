import cv2
import face_recognition
import os
import numpy as np

def load_known_faces(folder_path):
    """
    Load known face encodings and names from the specified folder.
    """
    known_face_encodings = []
    known_face_names = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            # Load image
            image_path = os.path.join(folder_path, filename)
            image = face_recognition.load_image_file(image_path)
            # Get face encodings
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                # Use filename (without extension) as the person's name
                name = os.path.splitext(filename)[0]
                known_face_names.append(name)
    
    return known_face_encodings, known_face_names

def recognize_face(image, known_face_encodings, known_face_names):
    """
    Recognize faces in the provided image.
    Returns the image with bounding boxes and names, and a list of recognized names.
    """
    # Convert image to RGB (face_recognition expects RGB)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Find face locations and encodings
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    recognized_names = []
    
    # Loop through each face found in the image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        # Use the closest match if available
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
        
        recognized_names.append(name)
        
        # Draw rectangle and label on the image
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return image, recognized_names
