"""
5_jouer_avec_cnn.py — Contrôler Space Invaders avec le modèle entraîné

Architecture fiable :
  Thread principal  : Webcam → MediaPipe → CNN → commande dans une Queue
  Thread secondaire : lit la Queue et envoie sur WebSocket (jamais bloqué)

Lancement :
  1) node server.js  (dans projet_computer_vision)
  2) python 5_jouer_avec_cnn.py
  3) Ouvrir http://localhost:8000  puis rafraîchir APRÈS que le serveur soit lancé
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import tensorflow as tf
import websocket          # websocket-client (synchrone, thread-safe)
import threading
import queue
import time
import os
import urllib.request

# ============================================================
# CHEMINS
# ============================================================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "modeles", "gestes_landmarks.keras")
CLASSES_PATH = os.path.join(BASE_DIR, "modeles", "classes_landmarks.txt")
MP_MODEL     = os.path.join(BASE_DIR, "hand_landmarker.task")
MP_URL       = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
WS_URI = "ws://localhost:8765"

# ============================================================
# VÉRIFICATIONS
# ============================================================
if not os.path.exists(MODEL_PATH):
    print(f"ERREUR : modele introuvable -> {MODEL_PATH}")
    print("Lance d'abord : python 4_entrainer_landmarks.py")
    exit(1)
if not os.path.exists(CLASSES_PATH):
    print(f"ERREUR : classes introuvable -> {CLASSES_PATH}")
    exit(1)
if not os.path.exists(MP_MODEL):
    print("Telechargement du modele MediaPipe...")
    urllib.request.urlretrieve(MP_URL, MP_MODEL)
    print("OK\n")

# ============================================================
# CHARGEMENT DU MODELE
# ============================================================
print("Chargement du CNN...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASSES_PATH, 'r') as f:
    CLASSES = [l.strip() for l in f if l.strip()]
print(f"Classes : {CLASSES}\n")

# ============================================================
# CONFIGURATION
# ============================================================
SEUIL_CONFIANCE = 0.70   # même seuil que le test local
# Pas de buffer de stabilisation : le serveur gère le "hold" avec HOLD_TIMEOUT=180ms

COOLDOWN = {
    'LEFT'   : 0.08,   # envoie toutes les 80ms → touche maintenue (HOLD_TIMEOUT=180ms)
    'RIGHT'  : 0.08,
    'FIRE'   : 0.08,
    'ENTER'  : 0.80,
    'NEUTRAL': 9999,
}
COULEURS = {
    'LEFT'   : (150, 150, 255),
    'RIGHT'  : (150, 255, 150),
    'FIRE'   : (0,   255, 255),
    'ENTER'  : (255, 200,   0),
    'NEUTRAL': (150, 150, 150),
}

# ============================================================
# THREAD WEBSOCKET (séparé du thread vidéo)
# ============================================================
cmd_queue   = queue.Queue(maxsize=20)
ws_connecte = threading.Event()   # True quand la connexion est établie
ws_app      = None

def ws_worker():
    """Thread dédié au WebSocket — jamais bloqué par la vidéo."""
    global ws_app

    def on_open(ws):
        ws_connecte.set()
        print("WebSocket connecte au jeu !")
        # Boucle d'envoi : lit la queue et envoie les commandes
        while True:
            try:
                cmd = cmd_queue.get(timeout=0.5)
                if cmd is None:          # signal d'arrêt
                    ws.close()
                    return
                ws.send(cmd)
            except queue.Empty:
                pass                     # rien à envoyer, on attend

    def on_error(ws, error):
        print(f"Erreur WebSocket : {error}")

    def on_close(ws, code, msg):
        ws_connecte.clear()
        print("WebSocket ferme.")

    ws_app = websocket.WebSocketApp(
        WS_URI,
        on_open=on_open,
        on_error=on_error,
        on_close=on_close,
    )
    ws_app.run_forever()

# Démarrer le thread WebSocket en arrière-plan
t = threading.Thread(target=ws_worker, daemon=True)
t.start()

print(f"Connexion WebSocket ({WS_URI})...")
if not ws_connecte.wait(timeout=5):
    print("ERREUR : impossible de se connecter.")
    print("Lance d'abord : node server.js (dans projet_computer_vision)")
    exit(1)

# ============================================================
# NORMALISATION (identique à la collecte)
# ============================================================
def normaliser(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]
    s = np.max(np.abs(pts))
    if s > 0:
        pts /= s
    return pts.flatten()

def envoyer(cmd):
    """Envoie une commande dans la queue sans bloquer."""
    try:
        cmd_queue.put_nowait(cmd)
    except queue.Full:
        pass

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
# BOUCLE PRINCIPALE (thread principal)
# ============================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

last_sent    = {c: 0.0 for c in CLASSES}
last_command = None
start_ms     = int(time.time() * 1000)

CONN = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

print("\n=== PRET A JOUER ===")
print("  LEFT / RIGHT : bouger le vaisseau")
print("  FIRE         : tirer")
print("  ENTER        : demarrer")
print("  Q            : quitter\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        ts_ms = int(time.time() * 1000) - start_ms

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
            for (x, y) in pts:
                cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)

            # Prédiction CNN
            vecteur   = normaliser(lm_list).reshape(1, -1)
            probs_all = model.predict(vecteur, verbose=0)[0]
            idx       = int(np.argmax(probs_all))
            confiance = float(probs_all[idx])
            geste     = CLASSES[idx] if confiance >= SEUIL_CONFIANCE else None

        # Envoi commande — direct, sans buffer de stabilisation
        now = time.time()
        if geste and geste != 'NEUTRAL' and ws_connecte.is_set():
            if now - last_sent[geste] >= COOLDOWN[geste]:
                envoyer(geste)
                last_sent[geste] = now
                last_command     = geste
                print(f"  {geste:<8} {confiance*100:.0f}%")

        # ============================================================
        # AFFICHAGE
        # ============================================================
        cv2.rectangle(frame, (0, 0), (w, 75), (25, 25, 25), -1)

        if geste and geste != 'NEUTRAL':
            col = COULEURS[geste]
            cv2.putText(frame, geste,
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.8, col, 4)
            bw = int((w - 210) * confiance)
            cv2.rectangle(frame, (175, 28), (175 + bw, 60), col, -1)
            cv2.putText(frame, f"{confiance*100:.0f}%",
                        (180 + bw, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        elif hand_ok:
            cv2.putText(frame, f"Incertain {confiance*100:.0f}%",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100,100,100), 2)
        else:
            cv2.putText(frame, "Aucune main",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80,80,80), 2)

        # Barres de probabilité
        if probs_all is not None:
            yb = 90
            for i, cls in enumerate(CLASSES):
                p  = float(probs_all[i])
                bw = int(120 * p)
                col = COULEURS[cls]
                cv2.rectangle(frame, (10, yb), (10 + bw, yb + 14), col, -1)
                cv2.putText(frame, f"{cls:<8} {p*100:4.0f}%",
                            (135, yb + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            col if i == int(np.argmax(probs_all)) else (130,130,130), 1)
                yb += 18

        if last_command:
            cv2.putText(frame, f"Envoye: {last_command}",
                        (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

        # Indicateur WebSocket
        col_ws = (0, 220, 0) if ws_connecte.is_set() else (0, 0, 200)
        cv2.circle(frame, (w - 20, 20), 10, col_ws, -1)
        cv2.putText(frame, "WS OK" if ws_connecte.is_set() else "WS OFF",
                    (w - 75, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col_ws, 1)

        cv2.putText(frame, "Q = quitter",
                    (w - 110, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)

        cv2.imshow("Space Invaders - CNN Landmarks", frame)

        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

finally:
    # Signal d'arrêt propre
    cmd_queue.put(None)
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Ferme. Au revoir !")
