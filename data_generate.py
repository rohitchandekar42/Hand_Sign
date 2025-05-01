import cv2
import os
from cvzone.HandTrackingModule import HandDetector


label = '9'                  # Class label for the gesture being captured
samples = 50                # Number of samples to capture
img_size = 224             # Resize cropped images to this size
padding = 30                # Padding around the hand
output_dir = f'data_2_0/{label}'  # Output directory

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Initialize hand detector
detector = HandDetector(maxHands=1, detectionCon=0.7)

# Start webcam
cap = cv2.VideoCapture(0)
count = 0

print(f"📸 Starting capture for label '{label}'... Press 'q' to quit early.")

while count < samples:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    hands, _ = detector.findHands(frame, draw=False)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Padding and boundaries
        x_pad = max(0, x - padding)
        y_pad = max(0, y - padding)
        w_pad = min(frame.shape[1], x + w + padding) - x_pad
        h_pad = min(frame.shape[0], y + h + padding) - y_pad

        # Crop and resize
        hand_img = frame[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad]
        hand_img_resized = cv2.resize(hand_img, (img_size, img_size))

        # Save the image
        img_path = os.path.join(output_dir, f'{label}_{count + 1}.jpg')
        cv2.imwrite(img_path, hand_img_resized)
        print(f'✅ Captured {img_path}')
        count += 1

        # Display with caption
        display_frame = frame.copy()
        cv2.rectangle(display_frame, (x_pad, y_pad), (x_pad + w_pad, y_pad + h_pad), (0, 255, 0), 2)
        cv2.putText(display_frame, f'Capturing: {label} ({count}/{samples})', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Hand Capture", display_frame)
    else:
        cv2.putText(frame, "Show one hand clearly...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Hand Capture", frame)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        print("🛑 Capture stopped by user.")
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Finished capturing hand images.")
