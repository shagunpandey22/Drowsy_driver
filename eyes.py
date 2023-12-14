import cv2 #open cv
import imutils
from imutils import face_utils #it is used for landmarks of left and right eye
import dlib
from scipy.spatial import distance
from pygame import mixer 

# to play music
mixer.init()
mixer.music.load("music.wav") #music file

thresh = 0.25 #it is a threshold value which can tell if the eye is close/blink or not
flag = 0 #frame count
frame_check = 30 #if eye blinks that much(means eyes closing) then alarm will start beeping

(lStart , lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye'] #Eye Landmarks For left eye
(rStart , rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye'] #Eye Landmarks For right eye

detect = dlib.get_frontal_face_detector()   #initializing a frontal face detector, it returns the sum of pre-trained hog(histogram oriented gradients)
#and linear svm phase detector which is included in the dlib library
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #file contains 68 landmarks , it will dtect from our face(these landmarks are some points
#on our face which includes eyes, eyebrows, jawline ,lips,etc)

# 6 points are detected for eyes (p0 and p3 are k/a long points or horizontal points,p1 and p5 , p2 and p4 are called short points or vertical points)
#we will use these points to calculate EAR (Eye Aspect Ratio), in this we dont require image processing
# EAR = SUM OF VERTICAL DISTANCE(p1, p5 and p2, p4 dist) / 2*HORIZONTAL DISTANCE OF EYE (p0,p3 dist)

def eye_aspect_ratio(eye): #to calculate EAR
     A= distance.euclidean(eye[1],eye[5])
     B= distance.euclidean(eye[2],eye[4])
     C= distance.euclidean(eye[0],eye[3])
     ear = (A+B) / (2.0*C) #put on formula
     return ear
#value to EAR remains constant when eye is open and suddenly drop when eye closes (value of (A+B) decreases)


cap = cv2.VideoCapture(0)
# it returns frames detected by the camera
# VideoCapture is the cv2's builtin function
# here (0) means using primary camera, we can provide 1,2... instead of 0 as many as camera increases (multi-cameras)

while True: #infinite loop
     
     ret, frame = cap.read()
     #read is a builtin func which returns 2 values , first one is boolean values (True or False) based on if the frame is
     #available or not (ret will store the returned/output value)

     frame = imutils.resize(frame, width=450) #resize the frame
     
     #frame is an image array vector captured based on the default frames per second defined explicitly or implicitly

     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converting frames into gray scale
     subjects = detect(gray, 0) #here detect is the detector which initialized before , 0 is the index
     
     #after detecting face, lets detect landmarks on it
     for subject in subjects:
          shape = predict(gray, subject) #here predict is the predictor, gray is a gray scale frames , subject is detector output
          shape = face_utils.shape_to_np(shape) #convering shapes into list of x,y coordinates
          #pass landmarks of left and right eye
          leftEye = shape[lStart:lEnd]
          rightEye = shape[rStart:rEnd]
          # calculate individual EAR using eye_aspect_ratio function
          leftEar = eye_aspect_ratio(leftEye)
          rightEar = eye_aspect_ratio(rightEye)
          #actual EAR is the average of leftEar and rightEar
          ear = (leftEar + rightEar)/2.0
          #find convex hull is the minimum boundary of the object that can completely enclose or wrap the object
          leftEyeHull = cv2.convexHull(leftEye)
          rightEyeHull = cv2.convexHull(rightEye)
          
          #contour - a curve that will join all the points lying on the boundary of that shape
          
          #LEFT EYE -
          cv2.drawContours(frame, [leftEyeHull],  -1 , (0, 255, 0), 1) #this function contains 5 arguments
          #frame - image on which we have to draw the contours , [leftEyeHull] - that is the indicated contour ,
          # -1 - contour index which represents the pixels coordinates that are listed in the contour (passing any negative value
          #bcoz we have to print all contour points)
          # (0, 255, 0) - indicate green color of the line, 1 - thickness of the line
          
          #RIGHT EYE -
          cv2.drawContours(frame, [rightEyeHull],  -1 , (0, 255, 0), 1)

          #above four line code will return the green outline on the image or eye part

          if ear<thresh : #thresh is a certain value from which if the ear value is less then it is considered as close / blink eye
               flag+=1
               print(flag)
               if flag>=frame_check: #if ear will be less that continuously for frame_check count, this will help to beep alarm when closes not just blinks
                    #warning message
                    cv2.putText(frame, "***** ALERT *****", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    #(10,30) coordinates of x and y axis
                    #FONT_HERSHEY_SIMPLEX - font
                    #0.7 - thickness of font
                    # (0, 0, 255) - color of font
                    # 2 - thickness of line
                    cv2.putText(frame, "***** ALERT *****", (10,325), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    #play music
                    mixer.music.play()
          else:
               flag = 0
                    
     cv2.imshow("Frame", frame)
     #it is used to display the image on a window, it accepts 2 parameters, 1st is window name (means name of the window on
     #which we have to display image) and 2nd the image name which we have to display(in our code image is stored in "frame" variable)

     if cv2.waitKey(1) and 0xFF == ord("q"):
          break
     #it allows us to display a window for our given millisecond(1,2,...) or until any case pressed(0), it then destroys the window

cv2.destroyAllWindows()
cap.release()