import cv2
import mediapipe as mp
import math
import numpy

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture('bicep_curl.mp4')

def angles(frame,p1,p2,p3,h,w):
    x1,y1=(p1.x)*w,(p1.y)*h
    x2,y2=(p2.x)*w,(p2.y)*h
    x3,y3=(p3.x)*w,(p3.y)*h
    angle=math.degrees(math.atan2((y3-y2),(x3-x2))-math.atan2((y2-y1),(x2-x1)))

    if angle<0:
        angle+=360
    return angle
count=0
dir=0
while True:
    success, frame = cap.read()
    if success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            first= results.pose_landmarks.landmark[11]
            second=results.pose_landmarks.landmark[13]
            third=results.pose_landmarks.landmark[15]
            h, w, _ = frame.shape
            angle=angles(frame,first,second,third,h,w)
            #cv2.circle(frame, (int(first.x * w), int(first.y * h)), 5, (255, 255, 0), 4)
            per=numpy.interp(angle, (25,110), (0,100))
            color=(255, 100, 100)
            if per == 100:
                 color=(100, 255, 100)
                 if dir==0:
                     count+=0.5
                     dir=1
            if per == 0:
                 color=(100, 100, 255)
                 if dir == 1:
                     count+=0.5
                     dir=0
            cv2.putText(frame,'COUNT',(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3,cv2.LINE_AA)
            cv2.putText(frame,str(count),(10,90),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3,cv2.LINE_AA)
        cv2.imshow('Pose Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(count)
            break
    else:
        break


cap.release()
cv2.destroyAllWindows()

