import cv2
import mediapipe as mp
import math
from time import sleep
import threading
import sys


class HandDetector:
    def __init__(self, staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.8, minTrackCon=0.8):
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

    def findHands(self, img, draw=True, flipType=False):
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


class Button:
    def __init__(self, pos, text, size=[45, 45]):
        self.pos = pos
        self.text = text
        self.size = size


class VirtualKeyboard:
    def __init__(self):
        self.text = ""
        self.keys = [
            ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
            ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
            ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/", "Terminate"]
        ]
        self.buttonList = []
        self.create_buttons()

    def create_buttons(self):
        for i in range(len(self.keys)):
            for j, key in enumerate(self.keys[i]):
                self.buttonList.append(Button([50 * j + 80, 60 * i + 20], key))

    def draw(self, img):
        for button in self.buttonList:
            x, y = button.pos
            w, h = button.size
            cv2.rectangle(img, button.pos, (x + w, y + h), (223, 125, 68), cv2.FILLED)
            cv2.putText(img, button.text, (x + 8, y + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        return img

    def update_text(self, img, hand, sleep_time=0.25):
        lmList1 = hand['lmList']
        for button in self.buttonList:
            x, y = button.pos
            w, h = button.size
            if x < lmList1[8][0] < x + w and y < lmList1[12][1] < y + h:
                cv2.rectangle(img, button.pos, (x + w, y + h), (223, 94, 20), cv2.FILLED)
                cv2.putText(img, button.text, (x + 8, y + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                x1, y1 = lmList1[8][0], lmList1[8][1]
                x2, y2 = lmList1[12][0], lmList1[12][1]
                length = math.hypot(x2 - x1, y2 - y1)
                if length < 26:
                    if button.text == "Terminate":
                        print("Terminating...")
                        cv2.destroyAllWindows()
                        sys.exit()  # Terminate the program
                    self.text += button.text
                    sleep(sleep_time)
        cv2.putText(img, self.text, (60, 425), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)


def run_hand_detection():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=0.8)
    keyboard = VirtualKeyboard()

    while True:
        success, img = cap.read()
        if not success:
            break

        hands, img = detector.findHands(img, flipType=False, draw=True)
        img = keyboard.draw(img)

        if hands:
            hand1 = hands[0]
            keyboard.update_text(img, hand1)

        cv2.imshow("Virtual Keyboard", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    threading.Thread(target=run_hand_detection).start()


if __name__ == "__main__":
    main()
