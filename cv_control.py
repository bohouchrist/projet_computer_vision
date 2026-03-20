"""
cv_control.py - Contrôle de Space Invaders par gestes via webcam
Utilise le modèle CNN du professeur (gesture_model.h5) entraîné sur la base
Kaggle "Hand Navigation Landmarks" (4 classes: down, left, right, up)

Architecture : Webcam → OpenCV → CNN → WebSocket → Jeu

Prérequis :
    pip install opencv-python tensorflow websockets numpy

Lancement :
    1) python -m http.server 8000        (serveur du jeu)
    2) node server.js                     (WebSocket bridge)
    3) python cv_control.py               (ce script)
    4) Ouvrir http://localhost:8000 dans le navigateur
"""

import cv2
import numpy as np
import asyncio
import websockets
import time

# Charger le modèle CNN
from tensorflow.keras.models import load_model

# ============================================================
# CONFIGURATION - Ajuste ces valeurs si nécessaire
# ============================================================
MODEL_PATH = "gesture_model.h5"          # Chemin vers le modèle sauvegardé
IMG_SIZE = (64, 64)                       # Taille d'entrée du CNN (vérifie dans le notebook)
CONFIDENCE_THRESHOLD = 0.75               # Seuil de confiance (baisser si trop peu de détections)
COOLDOWN = 0.35                           # Temps minimum entre 2 commandes (en secondes)
WS_URI = "ws://localhost:8765"            # Adresse du serveur WebSocket
CAMERA_INDEX = 0                          # Index de la webcam (0 = défaut)

# Classes dans l'ORDRE ALPHABÉTIQUE des dossiers du dataset
CLASS_NAMES = ['down', 'left', 'right', 'up']

# Mapping geste → commande du jeu
GESTURE_TO_COMMAND = {
    'left':  'LEFT',
    'right': 'RIGHT',
    'up':    'FIRE',       # Main vers le haut = tirer
    'down':  'ENTER',      # Main vers le bas = enter/start
}

# ============================================================
# FONCTIONS
# ============================================================

def preprocess_frame(frame):
    """
    Prétraite une image de la webcam pour le CNN.
    - Redimensionne à IMG_SIZE
    - Convertit en RGB
    - Normalise les pixels [0, 1]
    - Ajoute la dimension batch
    """
    resized = cv2.resize(frame, IMG_SIZE)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype('float32') / 255.0
    return np.expand_dims(normalized, axis=0)  # shape: (1, 64, 64, 3)


def predict_gesture(model, frame):
    """
    Prédit le geste à partir d'une frame webcam.
    Retourne (nom_du_geste, confiance) ou (None, 0) si en dessous du seuil.
    """
    processed = preprocess_frame(frame)
    predictions = model.predict(processed, verbose=0)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]

    if confidence >= CONFIDENCE_THRESHOLD:
        return CLASS_NAMES[class_idx], confidence
    return None, confidence


async def main():
    """Boucle principale : capture webcam → prédiction → envoi WebSocket."""

    # --- Charger le modèle ---
    print(f"Chargement du modèle depuis '{MODEL_PATH}'...")
    try:
        model = load_model(MODEL_PATH)
        print("Modèle chargé avec succès !")
    except Exception as e:
        print(f"ERREUR : Impossible de charger le modèle : {e}")
        print("Vérifie que 'gesture_model.h5' est dans le dossier du projet.")
        print("Pour le créer, ajoute model_cnn.save('gesture_model.h5') à la fin du notebook.")
        return

    # --- Ouvrir la webcam ---
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERREUR : Impossible d'ouvrir la webcam.")
        return

    print(f"Webcam ouverte (index {CAMERA_INDEX})")
    print(f"Seuil de confiance : {CONFIDENCE_THRESHOLD * 100:.0f}%")
    print(f"Cooldown entre commandes : {COOLDOWN}s")
    print("Appuie sur 'q' dans la fenêtre webcam pour quitter.\n")

    # --- Connexion WebSocket ---
    print(f"Connexion au serveur WebSocket ({WS_URI})...")
    try:
        websocket = await websockets.connect(WS_URI)
        print("Connecté au jeu !\n")
    except Exception as e:
        print(f"ERREUR : Impossible de se connecter au WebSocket : {e}")
        print("Vérifie que 'node server.js' est lancé.")
        cap.release()
        return

    last_command_time = 0
    last_gesture = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erreur de lecture webcam.")
                break

            # Miroir horizontal (plus intuitif pour le joueur)
            frame = cv2.flip(frame, 1)

            # Prédiction
            gesture, confidence = predict_gesture(model, frame)
            current_time = time.time()

            # Affichage sur la fenêtre webcam
            display_frame = frame.copy()

            if gesture:
                color = (0, 255, 0)  # Vert si détecté
                text = f"{gesture.upper()} ({confidence:.0%})"

                # Envoyer la commande si cooldown respecté
                if current_time - last_command_time >= COOLDOWN:
                    command = GESTURE_TO_COMMAND.get(gesture)
                    if command:
                        await websocket.send(command)
                        last_command_time = current_time
                        last_gesture = gesture
                        print(f"Geste: {gesture.upper():>5} | Confiance: {confidence:.0%} | Commande: {command}")
            else:
                color = (0, 0, 255)  # Rouge si pas détecté
                text = f"? ({confidence:.0%})"

            # Dessiner les infos sur l'image
            cv2.putText(display_frame, text, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            # Indicateur de dernière commande envoyée
            if last_gesture:
                cv2.putText(display_frame, f"Derniere: {last_gesture.upper()}",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Barre de confiance
            bar_width = int(confidence * 300)
            cv2.rectangle(display_frame, (10, 90), (10 + bar_width, 110), color, -1)
            cv2.rectangle(display_frame, (10, 90), (310, 110), (255, 255, 255), 1)

            cv2.imshow("Space Invaders - Gesture Control", display_frame)

            # Quitter avec 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nArrêt demandé par l'utilisateur.")
                break

    except websockets.exceptions.ConnectionClosed:
        print("Connexion WebSocket perdue.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        await websocket.close()
        print("Webcam et WebSocket fermés. Au revoir !")


# ============================================================
# LANCEMENT
# ============================================================
if __name__ == "__main__":
    asyncio.run(main())
