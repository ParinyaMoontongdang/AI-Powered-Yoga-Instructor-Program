import cv2
import mediapipe as mp
import math
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
# Setup Pose function for video.
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
#--------------------------------------------------------------------------------------------------
def calculate_angle(a, b, c):
  a = np.array([a.x, a.y])
  b = np.array([b.x, b.y])
  c = np.array([c.x, c.y])

  radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
  angle = np.abs(radians*180.0/np.pi)

  if angle > 180.0:
    angle = 360 - angle

  return angle
#--------------------------------------------------------------------------------------------------
def classifyPose(output_image, display=False):
    label = 'Unknown Pose' #start text
    color = (255, 0, 0) #RED 'UNKNOWN POSE'
    # Calculate angles
    #--------------------------------------------------------------------------------------------------
    #LEFT
    left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    #--------------------------------------------------------------------------------------------------
    #RIGHT
    right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
    right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
    #--------------------------------------------------------------------------------------------------
    # Calculate angle between shoulder, elbow, and wrist
    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    left_shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
    right_shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_hip)
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    # Display angle
    cv2.putText(frame, str(round(left_elbow_angle,2)),
                tuple(np.multiply([left_elbow.x, left_elbow.y], [640, 480]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, str(round(right_elbow_angle,2)),
                tuple(np.multiply([right_elbow.x, right_elbow.y], [640, 480]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, str(round(left_shoulder_angle,2)),
                tuple(np.multiply([left_shoulder.x, left_shoulder.y], [640, 480]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, str(round(right_shoulder_angle,2)),
                tuple(np.multiply([right_shoulder.x, right_shoulder.y], [640, 480]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, str(round(left_knee_angle,2)),
                tuple(np.multiply([left_knee.x, left_knee.y], [640, 480]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, str(round(right_knee_angle,2)),
                tuple(np.multiply([right_knee.x, right_knee.y], [640, 480]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    ##### WARRIOR II / T POSE #####
    if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:
        if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:
            if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:
                if left_knee_angle > 90 and left_knee_angle < 120 or right_knee_angle > 90 and right_knee_angle < 120:
                    label = 'Warrior II Pose'
            if left_knee_angle > 160 and left_knee_angle < 195 and right_knee_angle > 160 and right_knee_angle < 195:
                label = 'T Pose'
    ##### TREE POSE #####
    if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:
      if left_elbow_angle > 50 and left_elbow_angle < 65 and right_elbow_angle > 50 and right_elbow_angle < 65:
        # Check if the other leg is bended at the required angle.
        if left_knee_angle > 25 and left_knee_angle < 45 or right_knee_angle > 25 and right_knee_angle < 45:
            # Specify the label of the pose that is tree pose.
            label = 'Tree Pose'
    ##### LUNGE POSE #####
    if left_shoulder_angle and left_elbow_angle > 165 and left_shoulder_angle and left_elbow_angle < 195 and right_shoulder_angle and right_elbow_angle > 165 and right_shoulder_angle and right_elbow_angle < 195 :
      if left_knee_angle > 60 and left_knee_angle < 70 or right_knee_angle > 60 and right_knee_angle < 70 :
        if left_knee_angle > 125 and left_knee_angle < 140 or right_knee_angle > 125 and right_knee_angle < 140 :
          label = 'Lunge Pose'
    ##### Mountain Pose #####
    if left_knee_angle > 165 and left_knee_angle < 195 and right_knee_angle > 165 and right_knee_angle < 195:
      if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:
        if left_shoulder_angle > 0 and left_shoulder_angle < 20 and right_shoulder_angle > 0 and right_shoulder_angle < 20:
          label = 'Mountain Pose'
    if label != 'Unknown Pose':
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)

        # Write the label on the output image.
    cv2.putText(output_image, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
# Initialize the results variable
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # Process the pose detection
    results = pose.process(frame)
    
    # Convert the image color to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Draw the pose landmarks
    if results.pose_landmarks is not None:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Classify the pose
        pose_result = classifyPose(frame, display=False)
        if pose_result is not None:
            frame, _ = pose_result
            
    # Convert the image color back to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Show the frame
    cv2.namedWindow('Mediapipe Feed', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Mediapipe Feed', 1200, 720)
    cv2.imshow('Mediapipe Feed', frame)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()