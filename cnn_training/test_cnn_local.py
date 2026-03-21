"""
test_cnn_local.py — Test temps réel du CNN landmarks (sans jeu, sans WebSocket)

Modèle : cnn_training/modeles/gestes_landmarks.keras
Classes : ENTER, FIRE, LEFT, NEUTRAL, RIGHT

Lancer : python test_cnn_local.py
Quitter : Q ou Echap dans la fenêtre webcam
"""

import cv2
import numpy as np
import os
import sys
import time
import urllib.request

# ============================================================
# CHEMINS
# ============================================================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "modeles", "gestes_landmarks.keras")
CLASSES_PATH = os.path.join(BASE_DIR, "modeles", "classes_landmarks.txt")
MP_MODEL     = os.path.join(os.path.dirname(BASE_DIR), "hand_landmarker.task")
MP_URL       = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# Vérifications
if not os.path.exists(MODEL_PATH):
    print(f"ERREUR : modèle introuvable -> {MODEL_PATH}")
    sys.exit(1)
if not os.path.exists(CLASSES_PATH):
    print(f"ERREUR : fichier classes introuvable -> {CLASSES_PATH}")
    sys.exit(1)

# Téléchargement modèle MediaPipe si besoin
if not os.path.exists(MP_MODEL):
    print("Téléchargement du modèle MediaPipe...")
    urllib.request.urlretrieve(MP_URL, MP_MODEL)
    print("OK\n")

# ============================================================
# CHARGEMENT
# ============================================================
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

print("Chargement du modèle CNN...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASSES_PATH, 'r') as f:
    CLASSES = [l.strip() for l in f if l.strip()]
print(f"Classes : {CLASSES}")
model.summary()

# Warmup
_ = model.predict(np.zeros((1, 63)), verbose=0)
print("Prêt.\n")

# ============================================================
# CONFIGURATION
# ============================================================
CAMERA_INDEX    = 0
SEUIL_CONFIANCE = 0.75   # seuil pour afficher le geste

COULEURS = {
    'LEFT'   : (150, 150, 255),
    'RIGHT'  : (150, 255, 150),
    'FIRE'   : (0,   255, 255),
    'ENTER'  : (255, 200,   0),
    'NEUTRAL': (150, 150, 150),
}

CONN = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# ============================================================
# NORMALISATION (identique à l'entraînement)
# ============================================================
def normaliser(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]                        # centrer sur le poignet
    s = np.max(np.abs(pts))
    if s > 0:
        pts /= s                         # normaliser [-1, 1]
    return pts.flatten()                 # vecteur de 63 valeurs

# ============================================================
# MEDIAPIPE
# ============================================================
base_opts  = mp_python.BaseOptions(model_asset_path=MP_MODEL)
options    = mp_vision.HandLandmarkerOptions(
    base_options=base_opts,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = mp_vision.HandLandmarker.create_from_options(options)

# ============================================================
# BOUCLE WEBCAM
# ============================================================
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("ERREUR : impossible d'ouvrir la webcam.")
    sys.exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("=== TEST CNN LOCAL ===")
print("  Montre ta main → le CNN prédit le geste")
print("  Q / Echap = quitter\n")

start_ms    = int(time.time() * 1000)
fps_t       = time.time()
fps_count   = 0
fps_display = 0.0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        ts_ms = int(time.time() * 1000) - start_ms

        # FPS
        fps_count += 1
        if time.time() - fps_t >= 1.0:
            fps_display = fps_count / (time.time() - fps_t)
            fps_count   = 0
            fps_t       = time.time()

        # --- MediaPipe ---
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_img, ts_ms)

        geste     = None
        confiance = 0.0
        probs_all = None
        hand_ok   = bool(result.hand_landmarks)

        if hand_ok:
            lm_list = result.hand_landmarks[0]

            # Dessiner la main
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]
            for a, b in CONN:
                cv2.line(frame, pts[a], pts[b], (0, 200, 100), 2)
            for (px, py) in pts:
                cv2.circle(frame, (px, py), 5, (255, 255, 255), -1)
                cv2.circle(frame, (px, py), 5, (0, 150, 70),    1)

            # Prédiction CNN
            vecteur   = normaliser(lm_list).reshape(1, -1)
            probs_all = model.predict(vecteur, verbose=0)[0]
            idx       = int(np.argmax(probs_all))
            confiance = float(probs_all[idx])
            geste     = CLASSES[idx] if confiance >= SEUIL_CONFIANCE else None

        # ============================================================
        # AFFICHAGE
        # ============================================================
        # Fond haut
        cv2.rectangle(frame, (0, 0), (w, 80), (25, 25, 25), -1)

        if geste and geste != 'NEUTRAL':
            col = COULEURS.get(geste, (200, 200, 200))
            cv2.putText(frame, geste,
                        (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 2.0, col, 4)
            # Barre de confiance
            bw = int((w - 220) * confiance)
            cv2.rectangle(frame, (185, 30), (185 + bw, 62), col, -1)
            cv2.rectangle(frame, (185, 30), (w - 20,   62), (80, 80, 80), 1)
            cv2.putText(frame, f"{confiance*100:.0f}%",
                        (190 + bw, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        elif hand_ok:
            col = COULEURS.get(CLASSES[int(np.argmax(probs_all))], (150,150,150)) if probs_all is not None else (100,100,100)
            cv2.putText(frame, f"Incertain  {confiance*100:.0f}%",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
        else:
            cv2.putText(frame, "Aucune main detectee",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2)

        # Barres de probabilité par classe
        if probs_all is not None:
            yb      = 95
            max_idx = int(np.argmax(probs_all))
            for i, cls in enumerate(CLASSES):
                p   = float(probs_all[i])
                bw2 = int(130 * p)
                col = COULEURS.get(cls, (150,150,150))
                cv2.rectangle(frame, (10, yb), (10 + bw2, yb + 15),
                              col if i == max_idx else (60, 60, 60), -1)
                cv2.putText(frame, f"{cls:<8} {p*100:5.1f}%",
                            (145, yb + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            col if i == max_idx else (120, 120, 120), 1)
                yb += 20

        # FPS + indicateur détection
        cv2.putText(frame, f"{fps_display:.0f} fps",
                    (w - 75, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)
        cv2.circle(frame, (w - 20, 20), 10,
                   (0, 220, 0) if hand_ok else (0, 0, 200), -1)
        cv2.putText(frame, "Q = quitter",
                    (w - 115, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)

        cv2.imshow("Test CNN Local - Landmarks", frame)

        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

finally:
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Terminé.")
