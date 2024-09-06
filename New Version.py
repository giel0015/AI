import cv2
import dlib

# Define paths to your pictures folder and log file
pictures_folder = "path/to/your/pictures"
log_file = "log.txt"

# Load face detection and recognition models (paths might need adjustments)
detector = dlib.get_frontal_face_detector()
recognizer = dlib.dnn_recognition_win.load_recognition_model("path/to/dlib_shape_predictor_68_face_landmarks.dat")
facerec = dlib.load_model("path/to/dlib_cnn_face_recognition_net_int8.dat")

def recognize_faces(image):
  """
  This function detects faces in an image and recognizes them from pictures folder.
  """
  # Convert image to grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
  # Detect faces
  faces = detector(gray, 1)
  
  # Loop through detected faces
  for face in faces:
    # Get face landmarks (optional, for visualization)
    landmarks = recognizer(image, face)
    
    # Extract face region of interest (ROI)
    face_encoding = facerec.compute_face_descriptor(image, landmarks)
    
    # Loop through known faces (pictures) in the folder
    for filename in os.listdir(pictures_folder):
      known_face_path = os.path.join(pictures_folder, filename)
      known_img = cv2.imread(known_face_path)
      known_encoding = facerec.compute_face_descriptor(known_img, recognizer(known_img, detector(cv2.cvtColor(known_img, cv2.COLOR_BGR2GRAY), 1)[0]))
      
      # Compare face encodings (similarity)
      distance = dlib.distance(face_encoding, known_encoding)
      threshold = 0.6  # Adjust threshold for better/stricter recognition
      
      # If distance is below threshold, it's a recognized face
      if distance < threshold:
        name = os.path.splitext(filename)[0]  # Extract name from filename
        log_face(name)  # Log recognized face with timestamp
        # Optional: Draw rectangle around the face and display name (uncomment)
        # cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)
        # cv2.putText(image, name, (face.left(), face.top() - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
  # Return the image with optional rectangle and name (uncomment)
  # return image
  return None  # If returning only recognized faces with logging is desired

def log_face(name):
  """
  This function logs the recognized face name with timestamp to a file.
  """
  # Get current time
  now = datetime.datetime.now()
  timestamp = now.strftime("%H:%M:%S")
  
  # Open log file in append mode
  with open(log_file, "a") as f:
    f.write(f"{timestamp} - {name}\n")  # Write timestamp and name

# Load an image (replace with your image loading method)
image = cv2.imread("path/to/your/image.jpg")  

# Recognize faces and potentially display results (uncomment)
# recognized_image = recognize_faces(image)
# cv2.imshow("Face Recognition", recognized_image)  # Uncomment to display image
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Recognize faces and log only (recommended)
recognize_faces(image)

print("Face recognition complete. Check log.txt for results.")

# Specify paths (adjust as needed)
shape_predictor_path = "path/to/dlib_shape_predictor_68_face_landmarks.dat"
face_recognition_model_path = "path/to/dlib_cnn_face_recognition_net_int8.dat"

# Load models
recognizer = shape_predictor(shape_predictor_path)
facerec = cnn_face_recognition_model(face_recognition_model_path)
