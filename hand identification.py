import cv2
import mediapipe as mp
import threading
import tkinter as tk
from tkinter import Button


class HandDetector:
    def __init__(self, staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        # Initializing the hand detection object
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.staticMode,
            max_num_hands=self.maxHands,
            model_complexity=self.modelComplexity,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.minTrackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True, flipType=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                lmList = []
                xList, yList = [], []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    lmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = xmin, ymin, xmax - xmin, ymax - ymin
                cx, cy = xmin + (xmax - xmin) // 2, ymin + (ymax - ymin) // 2

                myHand['lmList'] = lmList
                myHand['bbox'] = bbox
                myHand['center'] = (cx, cy)
                if flipType:
                    myHand['type'] = "Left" if handType.classification[0].label == "Right" else "Right"
                else:
                    myHand['type'] = handType.classification[0].label
                allHands.append(myHand)

                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    color = (0, 255, 0) if myHand['type'] == 'Right' else (255, 0, 0)
                    cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), color, 2)
                    cv2.putText(img, myHand['type'], (xmin - 30, ymin - 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        return allHands, img

    def recognizeGesture(self, hand):
        lmList = hand['lmList']
        if not lmList:
            return None
        fingers = []
        tipIds = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

        # Thumb (x-axis comparison for left/right hands)
        if hand['type'] == "Right":
            fingers.append(1 if lmList[tipIds[0]][0] > lmList[tipIds[0] - 1][0] else 0)
        else:
            fingers.append(1 if lmList[tipIds[0]][0] < lmList[tipIds[0] - 1][0] else 0)

        # Other fingers (y-axis comparison)
        for id in range(1, 5):
            fingers.append(1 if lmList[tipIds[id]][1] < lmList[tipIds[id] - 2][1] else 0)

        totalFingers = fingers.count(1)

        # Gesture detection
        if totalFingers == 0:
            return "Closed Hand"
        elif totalFingers == 5:
            return "Open Hand"
        elif totalFingers == 1 and fingers[1] == 1:
            return "Thumbs Up"
        return None


class App:
    def __init__(self):
        self.running = True

    def stop(self):
        self.running = False


def main():
    app = App()
    threading.Thread(target=lambda: run_hand_detection(app)).start()

    # Create a simple GUI for quitting
    root = tk.Tk()
    root.title("Hand Detection")
    Button(root, text="Stop Detection", command=app.stop, font=("Arial", 14), bg="red", fg="white").pack(pady=20)
    root.mainloop()


def run_hand_detection(app):
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=2, detectionCon=0.8)
    while app.running:
        success, img = cap.read()
        if not success:
            break

        hands, img = detector.findHands(img)
        if hands:
            for hand in hands:
                gesture = detector.recognizeGesture(hand)
                if gesture:
                    bbox = hand['bbox']
                    color = (0, 255, 255) if gesture == "Open Hand" else (0, 165, 255)
                    cv2.putText(img, gesture, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

        cv2.putText(img, f"Hands Detected: {len(hands)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Hand Detection", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
