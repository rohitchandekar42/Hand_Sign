import cv2
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from Speak import SpeakWindow  # Ensure SpeakWindow is implemented

# Define the SignDetection function
def SignDetection(box):
    detector = HandDetector(maxHands=1)
    classifier = Classifier("hand_sign_with_digits_mobilenetv2.h5", "labels.txt")
    
    offset = 20
    imgSize = 224
    labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "del", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", " ", "t", "u", "v", "w", "x", "y", "z"]
    
    copy_last_word = ""
    output_sentence = ""
    prev_prediction = ""
    prev_prediction_count = 0
    del_count = 0
    ready_for_speech = False
    
    st.write("🔍 Trying to access the camera...")

    # Using Streamlit's camera input
    camera_input = st.camera_input("Capture Hand Gestures")
    if camera_input:
        img = camera_input

        imgOutput = img.copy()
        hands, img = detector.findHands(img, draw=False)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            x_pad = max(0, x - offset)
            y_pad = max(0, y - offset)
            w_pad = min(img.shape[1], x + w + offset) - x_pad
            h_pad = min(img.shape[0], y + h + offset) - y_pad
            imgCrop = img[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad]

            if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                imgResize = cv2.resize(imgCrop, (imgSize, imgSize))
                prediction, index = classifier.getPrediction(imgResize, draw=False)

                if labels[index] == "del":
                    del_count += 1
                    if del_count >= 7:
                        if len(output_sentence) > 0:
                            output_sentence = output_sentence[:-1]
                        del_count = 0
                else:
                    del_count = 0
                    if labels[index] != prev_prediction:
                        prev_prediction_count = 0
                    else:
                        prev_prediction_count += 1

                    if prev_prediction_count >= 10:
                        output_sentence += labels[index]
                        prev_prediction_count = 0

                prev_prediction = labels[index]

                cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                cv2.rectangle(imgOutput, (x_pad, y_pad), (x_pad + w_pad, y_pad + h_pad), (255, 0, 0), 4)
                cv2.putText(imgOutput, output_sentence, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        st.image(imgOutput, channels="BGR", use_container_width=True)

        # Speech handling
        if output_sentence and ready_for_speech:
            words = output_sentence.split()
            last_word = words[-1]
            copy_last_word = last_word
            SpeakWindow(last_word.strip(), box)
            ready_for_speech = False

        if output_sentence and (output_sentence[-1] == " "):
            words = output_sentence.split()
            last_word = words[-1]
            if last_word and (last_word == copy_last_word):
                ready_for_speech = False
            else:
                ready_for_speech = True

# Main function
def main(box):
    st.title("🖐️ HandSpeak: Real-Time Hand Sign Detection")
    st.write("This application detects and classifies hand signs in real time.")
    
    if st.button("Start Detection"):
        SignDetection(box)

if __name__ == "__main__":
    box = st.empty()
    SpeakWindow("Started...", box)
    main(box)
