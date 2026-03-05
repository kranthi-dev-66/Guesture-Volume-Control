import cv2                   
import mediapipe as mp       

mp_hands = mp.solutions.hands                   
mp_draw = mp.solutions.drawing_utils  

hands = mp_hands.Hands( 
    static_image_mode=False,        
    max_num_hands=2,               
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7    
)  

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()     
    if not success:                
        break                       

    frame = cv2.flip(frame, 1)      
   
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            
           
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
           
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape                              
                cx, cy = int(lm.x * w), int(lm.y * h)  
                print(f"Landmark {id}: x={cx}, y={cy}")

    cv2.imshow("Hand Detection", frame)
   
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()