"""
3_collecter_landmarks.py — Collecte des landmarks MediaPipe

Au lieu de sauvegarder des images brutes, on extrait directement les
21 points de la main (x, y, z) = 63 coordonnées normalisées, et on
les sauvegarde dans un fichier CSV.

Avantages vs images brutes :
  - Zéro bruit de fond
  - Données légères (quelques Ko au lieu de centaines de Mo)
  - Le modèle n'apprend QUE la forme de la main

Classes : LEFT, RIGHT, FIRE, ENTER, NEUTRAL

Touches :
  1/2/3/4/5   → Sélectionner la classe
  ESPACE       → Capturer un landmark manuellement
  A            → Lancer la capture automatique (toutes les 0.3s)
  S            → Arrêter la capture automatique
  Q            → Quitter et sauvegarder
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import csv
import os
import time
import urllib.request

# ============================================================
# MODÈLE MEDIAPIPE
# ============================================================
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
if not os.path.exists(MODEL_PATH):
    print("Téléchargement du modèle MediaPipe...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Modèle téléchargé !\n")

# ============================================================
# CONFIGURATION
# ============================================================
CLASSES    = ['LEFT', 'RIGHT', 'FIRE', 'ENTER', 'NEUTRAL']
CSV_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_landmarks", "landmarks.csv")
AUTO_DELAY = 0.3   # secondes entre deux captures automatiques

os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

# ============================================================
# NORMALISATION DES LANDMARKS
# ============================================================
def normaliser_landmarks(landmarks):
    """
    Normalise les 21 landmarks par rapport au poignet (point 0).
    On soustrait la position du poignet et on divise par la distance
    maximale pour rendre les coordonnées invariantes à la position
    et à la taille de la main dans l'image.
    Retourne un vecteur de 63 valeurs (21 points × x,y,z).
    """
    # Extraire x, y, z
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

    # Centrer sur le poignet
    pts -= pts[0]

    # Normaliser par la distance max (invariance à la taille)
    scale = np.max(np.abs(pts))
    if scale > 0:
        pts /= scale

    return pts.flatten().tolist()   # 63 valeurs


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================
def main():
    # Charger CSV existant pour continuer une collecte précédente
    compteurs = {c: 0 for c in CLASSES}
    lignes_existantes = 0
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)   # skip header
            for row in reader:
                if row:
                    label = row[0]
                    if label in compteurs:
                        compteurs[label] += 1
                    lignes_existantes += 1
        print(f"CSV existant chargé : {lignes_existantes} landmarks déjà collectés")

    # Ouvrir le CSV en mode ajout
    csv_file = open(CSV_PATH, 'a', newline='')
    writer   = csv.writer(csv_file)

    # Écrire l'en-tête si le fichier est vide
    if lignes_existantes == 0:
        header = ['classe'] + [f'{axe}{i}' for i in range(21) for axe in ('x', 'y', 'z')]
        writer.writerow(header)

    # MediaPipe
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options      = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    classe_active  = CLASSES[0]
    auto_capture   = False
    dernier_auto   = 0.0
    start_ms       = int(time.time() * 1000)
    feedback_msg   = ""
    feedback_timer = 0.0

    COULEURS_CLASSE = {
        'LEFT'   : (150, 150, 255),
        'RIGHT'  : (150, 255, 150),
        'FIRE'   : (0,   255, 255),
        'ENTER'  : (255, 200,   0),
        'NEUTRAL': (200, 200, 200),
    }

    print("\n=== COLLECTE DE LANDMARKS ===")
    print("  1→LEFT  2→RIGHT  3→FIRE  4→ENTER  5→NEUTRAL")
    print("  ESPACE → capture manuelle")
    print("  A → auto  |  S → stop auto  |  Q → quitter\n")

    def capturer(landmarks_list, classe):
        """Sauvegarde un vecteur de landmarks dans le CSV."""
        vecteur = normaliser_landmarks(landmarks_list)
        writer.writerow([classe] + vecteur)
        csv_file.flush()
        compteurs[classe] += 1

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame    = cv2.flip(frame, 1)
            h, w     = frame.shape[:2]
            ts_ms    = int(time.time() * 1000) - start_ms

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_img, ts_ms)

            hand_detected = bool(result.hand_landmarks)
            landmarks_list = None

            if hand_detected:
                landmarks_list = result.hand_landmarks[0]

                # Dessiner la main
                CONNEXIONS = [
                    (0,1),(1,2),(2,3),(3,4),
                    (0,5),(5,6),(6,7),(7,8),
                    (0,9),(9,10),(10,11),(11,12),
                    (0,13),(13,14),(14,15),(15,16),
                    (0,17),(17,18),(18,19),(19,20),
                    (5,9),(9,13),(13,17),
                ]
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks_list]
                for a, b in CONNEXIONS:
                    cv2.line(frame, pts[a], pts[b], (0, 200, 100), 2)
                for (x, y) in pts:
                    cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)

                # Capture automatique
                now = time.time()
                if auto_capture and (now - dernier_auto) >= AUTO_DELAY:
                    capturer(landmarks_list, classe_active)
                    dernier_auto    = now
                    feedback_msg   = f"AUTO: {classe_active} #{compteurs[classe_active]}"
                    feedback_timer = now

            # ============================================================
            # AFFICHAGE
            # ============================================================
            couleur = COULEURS_CLASSE[classe_active]

            # Bandeau supérieur
            cv2.rectangle(frame, (0, 0), (w, 70), (30, 30, 30), -1)
            cv2.putText(frame, f"Classe : {classe_active}",
                        (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, couleur, 3)

            # Indicateur AUTO
            if auto_capture:
                cv2.circle(frame, (w - 20, 20), 10, (0, 0, 255), -1)
                cv2.putText(frame, "AUTO", (w - 65, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.circle(frame, (w - 20, 20), 10, (0, 200, 0) if hand_detected else (100, 100, 100), -1)

            # Compteurs
            y_off = 90
            for cls in CLASSES:
                marker = "►" if cls == classe_active else " "
                texte  = f"{marker} {cls:<8}: {compteurs[cls]:>4}"
                cv2.putText(frame, texte, (10, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            COULEURS_CLASSE[cls] if cls == classe_active else (160, 160, 160), 1)
                y_off += 22

            # Total
            total = sum(compteurs.values())
            cv2.putText(frame, f"Total : {total}", (10, y_off + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Feedback capture
            if feedback_msg and (time.time() - feedback_timer) < 1.0:
                cv2.putText(frame, feedback_msg, (10, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2)

            # Instructions
            cv2.putText(frame, "1-5:classe  ESP:capture  A:auto  S:stop  Q:quitter",
                        (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            if not hand_detected:
                cv2.putText(frame, "Aucune main detectee", (w//2 - 100, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)

            cv2.imshow("Collecte Landmarks — Space Invaders", frame)

            # ============================================================
            # TOUCHES
            # ============================================================
            key = cv2.waitKey(1) & 0xFF
            now = time.time()

            if key == ord('1'):
                classe_active = 'LEFT';    auto_capture = False
            elif key == ord('2'):
                classe_active = 'RIGHT';   auto_capture = False
            elif key == ord('3'):
                classe_active = 'FIRE';    auto_capture = False
            elif key == ord('4'):
                classe_active = 'ENTER';   auto_capture = False
            elif key == ord('5'):
                classe_active = 'NEUTRAL'; auto_capture = False

            elif key == ord(' '):   # capture manuelle
                if hand_detected and landmarks_list:
                    capturer(landmarks_list, classe_active)
                    feedback_msg   = f"CAPTURE: {classe_active} #{compteurs[classe_active]}"
                    feedback_timer = now

            elif key == ord('a') or key == ord('A'):
                auto_capture = True
                dernier_auto = 0.0
                print(f"Auto capture ON → {classe_active}")

            elif key == ord('s') or key == ord('S'):
                auto_capture = False
                print("Auto capture OFF")

            elif key == ord('q') or key == ord('Q') or key == 27:
                break

    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()
        csv_file.close()

        total = sum(compteurs.values())
        print("\n=== RÉSUMÉ FINAL ===")
        for cls in CLASSES:
            barre = '█' * (compteurs[cls] // 10)
            print(f"  {cls:<8}: {compteurs[cls]:>5} landmarks  {barre}")
        print(f"\n  Total : {total} landmarks sauvegardés dans :")
        print(f"  {CSV_PATH}")
        print("\nLance maintenant : python 4_entrainer_landmarks.py")


if __name__ == "__main__":
    main()
