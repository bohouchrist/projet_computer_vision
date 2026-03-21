"""
cv_control_mediapipe.py - Contrôle de Space Invaders par gestes

Gestes (simples et intuitifs) :
  - Index pointé vers la GAUCHE (main dans zone gauche) → LEFT
  - Index pointé vers la DROITE (main dans zone droite) → RIGHT
  - Poing fermé vers l'avant (0 doigt, strict) → FIRE  (tirer)
  - Paume ouverte (5 doigts écartés)           → ENTER (démarrer)

  PRIORITÉ : FIRE > ENTER > LEFT / RIGHT
  Pour FIRE/ENTER, la position de la main n'a pas d'importance.

Pour arrêter : appuie sur 'Q' dans la fenêtre webcam, ou Ctrl+C dans le terminal.

Installation :
    pip install mediapipe opencv-python websockets numpy

Lancement :
    1) node server.js
    2) python cv_control_mediapipe.py
    3) python -m http.server 8000  puis ouvrir http://localhost:8000
"""

import cv2
import numpy as np
import asyncio
import websockets
import time
import os
import urllib.request
import sys

# ============================================================
# TÉLÉCHARGEMENT AUTOMATIQUE DU MODÈLE
# ============================================================
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print("Téléchargement du modèle MediaPipe...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Modèle téléchargé !\n")
    except Exception as e:
        print(f"ERREUR téléchargement : {e}")
        print(f"Télécharge manuellement depuis :\n  {MODEL_URL}")
        print(f"et place-le ici : {MODEL_PATH}")
        sys.exit(1)

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ============================================================
# CONFIGURATION
# ============================================================
WS_URI       = "ws://localhost:8765"
CAMERA_INDEX = 0

# Zones horizontales (fraction de la largeur du cadre)
LEFT_ZONE  = 0.40   # main à gauche de 40%  → LEFT
RIGHT_ZONE = 0.60   # main à droite de 60% → RIGHT

# Cooldowns (secondes) — temps minimum entre deux envois de la même commande
COOLDOWN = {
    'left' : 0.09,   # rapide pour mouvement fluide
    'right': 0.09,
    'fire' : 0.45,   # pas trop de spam
    'enter': 0.80,
}

# ============================================================
# DÉTECTION DE GESTES
# ============================================================

def count_fingers(landmarks):
    """
    Compte les doigts étendus (hors pouce).
    Retourne un entier entre 0 et 4.
    On ignore le pouce pour éviter les erreurs de latéralité.
    """
    tips = [8, 12, 16, 20]   # bouts des doigts : index, majeur, annulaire, auriculaire
    pips = [6, 10, 14, 18]   # articulations PIP
    return sum(landmarks[t].y < landmarks[p].y for t, p in zip(tips, pips))


def is_thumb_extended(landmarks):
    """Pouce étendu si le bout est loin du poignet horizontalement."""
    wrist = landmarks[0]
    tip   = landmarks[4]
    return abs(tip.x - wrist.x) > 0.15


def classify_gesture(landmarks, frame_w, frame_h):
    """
    Retourne (geste, palm_x, palm_y).
    geste ∈ {'fire', 'enter', 'left', 'right', None}
    """
    n = count_fingers(landmarks)
    thumb = is_thumb_extended(landmarks)

    # Centre de la paume = moyenne poignet + bases des 4 doigts
    key_pts = [0, 5, 9, 13, 17]
    palm_x = int(np.mean([landmarks[i].x * frame_w for i in key_pts]))
    palm_y = int(np.mean([landmarks[i].y * frame_h for i in key_pts]))

    total = n + (1 if thumb else 0)

    # Poing fermé strict (0 élément étendu) → FIRE
    if total == 0:
        return 'fire', palm_x, palm_y

    # Paume ouverte (4-5 éléments étendus) → ENTER
    if total >= 4:
        return 'enter', palm_x, palm_y

    # Autrement (1-3 doigts, index pointé) → navigation selon position
    ratio = palm_x / frame_w
    if ratio < LEFT_ZONE:
        return 'left', palm_x, palm_y
    if ratio > RIGHT_ZONE:
        return 'right', palm_x, palm_y

    return None, palm_x, palm_y


def draw_hand(frame, landmarks, frame_w, frame_h):
    """Dessine les landmarks et connexions sur le frame."""
    CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]
    pts = [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 100), 2)
    for (x, y) in pts:
        cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 5, (0, 150, 70), 1)


# ============================================================
# BOUCLE PRINCIPALE
# ============================================================

async def main():
    # --- Webcam ---
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERREUR : impossible d'ouvrir la webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # --- WebSocket ---
    print(f"Connexion à {WS_URI} ...")
    try:
        websocket = await websockets.connect(WS_URI)
        print("Connecté au jeu !\n")
    except Exception as e:
        print(f"ERREUR WebSocket : {e}")
        print("Lance d'abord : node server.js")
        cap.release()
        return

    # --- MediaPipe ---
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    COMMAND = {'left':'LEFT', 'right':'RIGHT', 'fire':'FIRE', 'enter':'ENTER'}
    last_sent   = {g: 0.0 for g in COMMAND}
    last_gesture = None
    start_ms     = int(time.time() * 1000)

    print("=== CONTRÔLES ===")
    print("  Main à GAUCHE      →  LEFT")
    print("  Main à DROITE      →  RIGHT")
    print("  Poing fermé        →  FIRE")
    print("  Paume ouverte      →  ENTER")
    print("  'Q' dans la fenêtre ou Ctrl+C  →  Quitter\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erreur de lecture webcam.")
                break

            frame    = cv2.flip(frame, 1)   # miroir horizontal
            h, w     = frame.shape[:2]
            ts_ms    = int(time.time() * 1000) - start_ms

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_img, ts_ms)

            gesture      = None
            palm_x, palm_y = w // 2, h // 2
            hand_detected  = False

            if result.hand_landmarks:
                lm = result.hand_landmarks[0]
                draw_hand(frame, lm, w, h)
                gesture, palm_x, palm_y = classify_gesture(lm, w, h)
                hand_detected = True

            # --- Envoi WebSocket ---
            now = time.time()
            if gesture and hand_detected:
                if now - last_sent[gesture] >= COOLDOWN[gesture]:
                    await websocket.send(COMMAND[gesture])
                    last_sent[gesture] = now
                    last_gesture = gesture
                    print(f"  {gesture.upper():>6}  →  {COMMAND[gesture]}")

            # ============================================================
            # AFFICHAGE
            # ============================================================
            left_x  = int(LEFT_ZONE  * w)
            right_x = int(RIGHT_ZONE * w)

            # Zones colorées semi-transparentes
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0),       (left_x,  h), (30,  30, 180), -1)
            cv2.rectangle(overlay, (right_x, 0), (w,       h), (30, 180,  30), -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

            # Lignes de séparation
            cv2.line(frame, (left_x,  0), (left_x,  h), (80, 80, 220), 2)
            cv2.line(frame, (right_x, 0), (right_x, h), (80, 220, 80), 2)

            # Labels zones
            cv2.putText(frame, "LEFT",  (8, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 255), 2)
            cv2.putText(frame, "RIGHT", (right_x + 8, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 255, 150), 2)

            # Rappel gestes FIRE/ENTER au centre
            cx = left_x + (right_x - left_x) // 2
            cv2.putText(frame, "Paume=FIRE", (cx - 60, h - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 50), 1)
            cv2.putText(frame, "Poing=ENTER", (cx - 65, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 50), 1)

            # Geste actuel (grand texte en haut)
            COLORS = {
                'left' : (150, 150, 255),
                'right': (150, 255, 150),
                'fire' : (0,   255, 255),
                'enter': (255, 200,   0),
            }
            if gesture:
                cv2.putText(frame, gesture.upper(),
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                            COLORS[gesture], 4)
            else:
                txt = "Aucune main" if not hand_detected else "---"
                cv2.putText(frame, txt,
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)

            # Dernier geste envoyé
            if last_gesture:
                cv2.putText(frame, f"Envoye: {COMMAND[last_gesture]}",
                            (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            # Point centre paume
            if hand_detected:
                cv2.circle(frame, (palm_x, palm_y), 12, (0, 255, 255), -1)
                cv2.circle(frame, (palm_x, palm_y), 14, (0, 0, 0),      2)

            # Indicateur état (vert = main détectée, rouge = rien)
            cv2.circle(frame, (w - 20, 20), 10,
                       (0, 220, 0) if hand_detected else (0, 0, 200), -1)

            # Instructions arrêt
            cv2.putText(frame, "Q = quitter", (w - 110, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

            cv2.imshow("Space Invaders - MediaPipe", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:   # Q ou Echap
                print("\nArrêt demandé.")
                break

    except KeyboardInterrupt:
        print("\nCtrl+C reçu, arrêt...")
    except websockets.exceptions.ConnectionClosed:
        print("Connexion WebSocket perdue.")
    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()
        try:
            await websocket.close()
        except Exception:
            pass
        print("Fermé proprement. Au revoir !")


if __name__ == "__main__":
    asyncio.run(main())
