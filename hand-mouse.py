import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Disable PyAutoGUI failsafe
pyautogui.FAILSAFE = False

# Settings
CLICK_THRESHOLD = 40  # Lower value = harder to click
SMOOTHING = 5

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

screen_width, screen_height = pyautogui.size()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

prev_x, prev_y = 0, 0
was_clicking = False

print("Hand Mouse Control Started!")
print("- Move your index finger to control the mouse")
print("- Pinch thumb and index finger to click")
print("- Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            thumb = hand_landmarks.landmark[4]
            index = hand_landmarks.landmark[8]
            
            thumb_x, thumb_y = int(thumb.x * frame_width), int(thumb.y * frame_height)
            index_x, index_y = int(index.x * frame_width), int(index.y * frame_height)
            
            distance = np.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
            
            screen_x = np.interp(index_x, [0, frame_width], [0, screen_width])
            screen_y = np.interp(index_y, [0, frame_height], [0, screen_height])
            
            current_x = prev_x + (screen_x - prev_x) / SMOOTHING
            current_y = prev_y + (screen_y - prev_y) / SMOOTHING
            prev_x, prev_y = current_x, current_y
            
            pyautogui.moveTo(current_x, current_y)
            
            if distance < CLICK_THRESHOLD:
                cv2.circle(frame, (thumb_x, thumb_y), 15, (0, 0, 255), -1)
                cv2.circle(frame, (index_x, index_y), 15, (0, 0, 255), -1)
                cv2.putText(frame, "CLICKING!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                if not was_clicking:
                    pyautogui.click()
                    was_clicking = True
            else:
                cv2.circle(frame, (thumb_x, thumb_y), 15, (0, 255, 0), -1)
                cv2.circle(frame, (index_x, index_y), 15, (0, 255, 0), -1)
                was_clicking = False
            
            cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 0), 2)
    
    cv2.putText(frame, "Press 'q' to quit", (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow('Hand Mouse Control', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

