import cv2

# Build the classifier
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
trained_smile_data = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# Capture video from webcam
webcam = cv2.VideoCapture(2, cv2.CAP_DSHOW)

# Iterate over video frames
while True:
    # Read the frame
    successful_frame_read, frame = webcam.read()

    # Convert the frame to grayscale
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the face coordinates
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame) 
    #smile_coordinates = trained_smile_data.detectMultiScale(grayscaled_frame, scaleFactor = 1.7, minNeighbours = 20)
    
    # Draw rectangles around faces
    for (x,y,w,h) in face_coordinates:      
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        # Create the sub-image of face within the frame
        the_face = frame[y:y+h, x:x+w]    # Face sub-image is sliced out from frame like a numpy array
        
        # Convert the sub-image to grayscale
        grayscaled_face_img = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        # Detect the smile coordinates within the face sub-image
        face_smile_coordinates = trained_smile_data.detectMultiScale(grayscaled_face_img, scaleFactor=1.7, minNeighbors=20)

        # Detect smile inside the face sub-imageq
        """for (x_,y_,w_,h_) in face_smile_coordinates: 
            cv2.rectangle(the_face, (x_,y_), (x_+w_, y_+h_), (50,50,200), 2)"""

        # Label the face as smilling
        if len(face_smile_coordinates)>0:
            cv2.putText(frame, "Smiling", (x, y+h+40), fontScale=2, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,255,255))

    # Display the frame
    cv2.imshow("THE Smile Detector",frame)

    # Keeps the image open until a key is pressed
    key = cv2.waitKey(1)     
    
    # Press q to quit video feed
    if key==81 or key==113:
        break
    
# Release the VideoCapture object
webcam.release()

# Clean up code
cv2.destroyAllWindows()        