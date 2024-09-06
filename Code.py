import face_recognition
import cv2
import numpy as np
import os
import time
import math
import logging
from datetime import datetime
import sys
# Load name_map from file
with open("C:\\Users\\p2p 1\\Desktop\\Scalda-FR-2\\Gezichten\\Naamlijst.txt") as file:
    name_map = {line.strip(): i for i, line in enumerate(file)}


def configure_logger(filename='log.txt', level=logging.INFO):
  """Configures the logger for face recognition events.

  Args:
    filename: The filename for log storage (default: 'log.txt').
    level: The logging level (default: logging.INFO).
  """
  # Create a logger for this module
  logger = logging.getLogger(__name__) # Define once, reuse here.
  logger.setLevel(level)

  # Create a file handler for writing logs to a file
  file_handler = logging.FileHandler(filename)
  formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
  file_handler.setFormatter(formatter)
  logger.addHandler(file_handler)

  # Add a stream handler (optional, for console output)
  # stream_handler = logging.StreamHandler()
  # stream_handler.setFormatter(formatter)
  # Uncomment the line above and the next to enable console output
  # logger.addHandler(stream_handler)

def log_facerec(message, level=logging.INFO, name_map="C:\\Users\\p2p 1\\Desktop\\Scalda-FR-2\\Gezichten\\Naamlijst.txt"):
  """Logs a message related to face recognition, including timestamp, name (if applicable), and confidence (if available).

  Args:
    message: The message to log.
    level: The logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING).
    name_map: A dictionary mapping internal identifiers to names (optional).
  """

 
  # Get current timestamp
  timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

  

  # Extract recognized face name (if applicable)
  if message.startswith("Face recognized:"):
    recognized_face = message.split(": ")[1]

    # Check for filenames and retrieve name from the dictionary (if provided)
    if name_map is not None and recognized_face in name_map:
      name = name_map[recognized_face]
      log_message = f"{timestamp} - Recognized: {name} ({recognized_face})"
    else:
      log_message = f"{timestamp} - Recognized unknown face: {recognized_face}"

  else:
    log_message = message  # Log the original message otherwise


    logger = logging.getLogger(__name__) # Define once, reuse here
  # Log the message with timestamp
    logger.log(level, log_message)
face_locations = []


    # Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Example logging messages
logging.info("Starting face recognition process.")

# After detecting a face
logging.debug("Face detected at coordinates:", face_locations)

# After recognizing a face
logging.info("Recognized face:", face_recognition)

 


# Configure the logger before using the log_facerec function
configure_logger() 

log_facerec("Starting face recognition process.")



# Function to select camera
def select_camera():
    print("Select camera:")
    print("1. Default (0)")
    print("2. Other")
    print("Waiting for response...")
    start_time = time.time()
    while time.time() - start_time < 5:  # Wait for 5 seconds for user input
        if cv2.waitKey(1) & 0xFF == ord('1'):  # If user selects default (1)
            print("Default camera selected.")
            return 0
        elif cv2.waitKey(1) & 0xFF == ord('2'):  # If user selects other (2)
            camera_index = input("Enter camera index: ")
            print(f"Selected camera index: {camera_index}")
            return int(camera_index)
    print("No response in 5 seconds. Defaulting to option 1...")
    return 0

# Get the webcam
camera_index = select_camera()
video_capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
if not(video_capture is None or not video_capture.isOpened()):
    print("\nWebcam successfully found!")
elif video_capture is None or not video_capture.isOpened():
    print("\nWebcam not found! Please check your camera connection.")
    exit()

# FPS
camera_fps_start_time = time.time()
camera_fps = 0 
camera_fps_display_time = 0
frame_count = 30
frame_time = 0.1

# Naam- en gezichtslijst
known_face_encodings = []
known_face_names = []

# Ga door alle gezichtafbeeldingen in de folder 'Gezichten' en analyseer ze
face_analysis_number = 1
while True:
    if not(os.path.exists("Gezichten/"+str(face_analysis_number)+".png")):
        break
    face_image = face_recognition.load_image_file("Gezichten/"+str(face_analysis_number)+".png")
    face_encoding = face_recognition.face_encodings(face_image)[0]
    if len(face_encoding) == 0:
        print(f"No face encoding found for image {face_analysis_number}")
    else:
        known_face_encodings.append(face_encoding)
        print(f"Face encoding loaded successfully for image {face_analysis_number}")
    face_analysis_number = face_analysis_number + 1

# Naamlijst lezen en in array met namen plaatsen
with open("Gezichten/Naamlijst.txt", 'r') as file: 
    known_face_names = file.read().splitlines()

# Check of gezicht bestaat in de files
matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
name = "Onbekend"

face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
if len(face_distances) > 0:
          best_match_index = np.argmin(face_distances)
if matches[best_match_index]:
                name = known_face_names[best_match_index]

                # Get current timestamp with date and time
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Open the text file in append mode ('a')
                with open("recognized_faces.txt", "a") as text_file:
                    # Write recognized face information to the file
                    text_file.write(f"{current_time} - Recognized: {name}\n")



# Waarden initializeren
face_locations = []
face_encodings = [] 
face_names = []
camera_size = 1
font = cv2.FONT_HERSHEY_DUPLEX
face_match_threshold = 0.6

# Programmaloop
while True:
    # Voor FPS
    frame_count = frame_count + 15
    frame_time = time.time() - camera_fps_start_time
    if frame_time >= camera_fps_display_time and frame_time != 0:
        camera_fps = int(frame_count / frame_time)
        frame_count = 0
        camera_fps_start_time = time.time()

    # Lees videoframe
    ret, frame = video_capture.read()
    print("Frame read status:", ret)  # Check the status of reading the frame

    # Als het frame niet correct is gelezen, sla de rest van de loop over
    if not ret:
        continue
    if frame_count % 10 == 0:
    # Frameresolutie
        small_frame = cv2.resize(frame, (0, 0), fx=camera_size, fy=camera_size)

    # BGR naar RGB kleur
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

    # Alle gezichten in het huidige frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        
        def process_recognition(message, timestamp, name_map, logger):
            """Processes face recognition message and logs details."""
            name = name_map.get(message.split(": ")[1], message.split(": ")[1])  # Extract and potentially map name
            logger.info(f"{timestamp} - Recognized: {name} ({message.split(': ')[1]})") if name in name_map else logger.info(f"{timestamp} - Recognized unknown face: {message.split(': ')[1]}")

        # Check of gezicht bestaat in de files
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Onbekend"


    
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
             

            increased_distance = face_distances[best_match_index] * 1.1
            # Calculate confidence percentage
    confidence_percentage = int((1 - face_distances[best_match_index]) * 100)

    face_names.append(name)


    # Frames per seconde
    cv2.putText(frame, str(camera_fps), (16, 32), font, 1.0, (0, 255, 0), 1)

    # Laat resultaten zien
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Streamresolutie
        top *= int(1 / camera_size)
        right *= int(1 / camera_size)
        bottom *= int(1 / camera_size)
        left *= int(1 / camera_size)

        # Log the recognized face
        log_facerec(f"Face recognized: {name}", name_map=name_map)

        # Teken rechthoek om gezicht
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Naamlabel bij de rechthoek met zekerheidswaarde erbij
        cv2.putText(frame, name, (left, bottom + 48), font, 1.0, (0, 255, 0), 1)
        if name != "Onbekend":
            cv2.putText(frame, str(confidence_percentage) + " % zekerheid", (left, bottom + 24), font, 1.0, (0, 255, 0), 1)

    # Laat videostream zien
    cv2.imshow('Video', frame)

    # Druk op toets om af te sluiten
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Afsluiten
video_capture.release()
cv2.destroyAllWindows()
