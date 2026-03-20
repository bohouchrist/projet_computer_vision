"""
test_model.py - Test rapide du modèle CNN avec la webcam
Lance ce script AVANT cv_control.py pour vérifier que le modèle
détecte bien tes gestes. Pas besoin du serveur WebSocket ni du jeu.

Utilisation :
    python test_model.py

Appuie 'q' pour quitter.
"""

import cv2
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# ============================================================
# CONFIGURATION
# ============================================================
WEIGHTS_PATH = "gesture_model_final.weights.h5"
IMG_SIZE = (180, 180)  # Taille utilisée dans le notebook du professeur

# ORDRE DES CLASSES tel que défini dans le notebook (classes_choisies)
CLASS_NAMES = ['left', 'right', 'up', 'down']

COLORS = {
    'down':  (255, 100, 50),
    'left':  (255, 50, 50),
    'right': (50, 255, 50),
    'up':    (50, 200, 255),
}


def build_cnn():
    """
    Reconstruit l'architecture CNN exactement comme dans le notebook.
    Le Rescaling(1./255) est INTEGRE au modèle, donc on ne normalise
    PAS les images dans le prétraitement.
    """
    model_cnn = Sequential([
        layers.Input(shape=(180, 180, 3)),
        layers.Rescaling(1./255),  # Normalisation intégrée

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4, activation='softmax')
    ])

    model_cnn.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model_cnn


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Reconstruction de l'architecture CNN...")
    model = build_cnn()

    print(f"Chargement des poids depuis '{WEIGHTS_PATH}'...")
    try:
        model.load_weights(WEIGHTS_PATH)
        print("Poids chargés avec succès !")
    except Exception as e:
        print(f"\nERREUR : {e}")
        print("\n--- SOLUTION ---")
        print("Vérifie que 'gesture_model_final.weights.h5' est dans ce dossier.")
        print(f"Dossier actuel : {__import__('os').getcwd()}")
        print(f"Fichiers disponibles : {__import__('os').listdir('.')}")
        return

    print("\nArchitecture du modèle :")
    model.summary()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERREUR : Webcam non disponible.")
        return

    print(f"\n--- TEST EN COURS ---")
    print(f"Classes : {CLASS_NAMES}")
    print(f"Taille d'entrée : {IMG_SIZE}")
    print(f"Normalisation : intégrée au modèle (Rescaling 1/255)")
    print("Montre tes gestes (up/down/left/right) devant la webcam.")
    print("Appuie 'q' pour quitter.\n")

    fps_time = time.time()
    frame_count = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        display = frame.copy()

        # Prétraitement : redimensionner + convertir en RGB
        # PAS de normalisation /255 ici car le modèle a un layer Rescaling intégré
        resized = cv2.resize(frame, IMG_SIZE)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        batch = np.expand_dims(rgb.astype('float32'), axis=0)

        # Prédiction
        predictions = model.predict(batch, verbose=0)[0]
        best_idx = np.argmax(predictions)
        best_class = CLASS_NAMES[best_idx]
        best_conf = predictions[best_idx]

        # Afficher le geste détecté
        color = COLORS.get(best_class, (255, 255, 255))
        cv2.putText(display, f"{best_class.upper()} {best_conf:.0%}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        # Barres de confiance pour chaque classe
        y_start = 80
        for i, name in enumerate(CLASS_NAMES):
            conf = predictions[i]
            bar_w = int(conf * 250)
            c = COLORS[name]
            y = y_start + i * 35

            cv2.rectangle(display, (10, y), (10 + bar_w, y + 25), c, -1)
            cv2.rectangle(display, (10, y), (260, y + 25), (200, 200, 200), 1)
            cv2.putText(display, f"{name}: {conf:.0%}", (270, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # FPS
        frame_count += 1
        elapsed = time.time() - fps_time
        if elapsed > 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_time = time.time()
        cv2.putText(display, f"FPS: {fps:.0f}", (10, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # Ligne seuil 75%
        cv2.line(display, (10 + int(0.75 * 250), y_start),
                 (10 + int(0.75 * 250), y_start + len(CLASS_NAMES) * 35),
                 (0, 255, 255), 1)
        cv2.putText(display, "75%", (10 + int(0.75 * 250) - 15, y_start - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        cv2.imshow("Test Modele - Gestes", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Test terminé.")


if __name__ == "__main__":
    main()
