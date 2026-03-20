# collect_data.py
import cv2, os

GESTURES = ["gauche", "droite", "haut", "tiet", "neutre"]
SAVE_DIR = "dataset"
NB_IMAGES = 200  # 200 images par geste

for gesture in GESTURES:
    os.makedirs(f"{SAVE_DIR}/{gesture}", exist_ok=True)

cap = cv2.VideoCapture(0)
current_gesture = 0
count = 0

while True:
    ret, frame = cap.read()
    # Recadre une zone carrée (64x64) centrée sur la main
    roi = frame[100:300, 150:350]        # Region Of Interest
    roi_resized = cv2.resize(roi, (64, 64))

    cv2.imshow("Collecte", frame)
    cv2.rectangle(frame, (150,100), (350,300), (0,255,0), 2)
    cv2.putText(frame, f"Geste: {GESTURES[current_gesture]} ({count}/200)",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    key = cv2.waitKey(1)
    if key == ord('s'):  # Appuie sur S pour sauvegarder
        path = f"{SAVE_DIR}/{GESTURES[current_gesture]}/{count}.jpg"
        cv2.imwrite(path, roi_resized)
        count += 1
        if count >= NB_IMAGES:
            current_gesture += 1
            count = 0
    if key == ord('q'):
        break
```

**Structure du dataset obtenu :**
```
dataset/
  gauche/   → 200 images
  droite/   → 200 images
  haut/     → 200 images
  tiet/     → 200 images
  neutre/   → 200 images

  