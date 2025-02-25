from keras import models 
import cv2  
import numpy as np 
import os

#Fix Encoding Issues for Console Output (UTF-8)
os.environ['PYTHONIOENCODING'] = 'utf-8'
import sys
sys.stdout.reconfigure(encoding='utf-8')
#Load Pre-Trained CNN Model
json_file = open("emotiondetector.json", "r") 
model_json = json_file.read() #Read model architecture from JSON file
json_file.close() 
model = models.model_from_json(model_json) #Load model architecture from JSON
model.load_weights("emotiondetector.h5") #Load model weights

# Define Feature Extraction Function
def extract_features(image): # Preprocess the image for prediction
    feature = np.array(image)  # Convert image to NumPy array
    feature = feature.reshape(1, 48, 48, 1)  # Reshape to match CNN input shape (batch_size, height, width, channels)
    return feature / 255.0  # Normalize pixel values to range [0, 1]

#  Initialize Video Capture (Webcam) and Emotion Labels
cap = cv2.VideoCapture(0) #Opens the default camera (0 refers to the first available webcam).
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'] 

# Start Real-Time Emotion Detection
while True:
    _, frame = cap.read() # Read a frame from the webcam
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    # Loads OpenCV’s Haar Cascade face detection model.
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) # Detect faces in the frame
    # Process Each Detected Face
    for (x, y, w, h) in faces: 
        face_roi = gray[y:y+h, x:x+w] 
        face_roi = cv2.resize(face_roi, (48, 48))  # Resize to match model input size
        processed_face = extract_features(face_roi)  # Preprocess the face for prediction
        
        pred = model.predict(processed_face)[0] # Predict emotion probabilities
        prediction_label = labels[np.argmax(pred)] # Get the emotion with the highest probability
        accuracy = "{:.2f}".format(np.max(pred) * 100) # Get confidence score
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 2) # Draw rectangle around face
        cv2.putText(frame, f'{prediction_label} {accuracy}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2) # Display emotion text
    
    cv2.imshow("Emotion recognition", frame) # Display the frame with emotion text
    if cv2.waitKey(27) & 0xFF == 27:  # Exit on ESC key
        break
   
cap.release() # Release the webcam
cv2.destroyAllWindows() # Close the OpenCV windows