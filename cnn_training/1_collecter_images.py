"""
ÉTAPE 1 — Collecte d'images pour entraîner le CNN
===================================================

Ce script ouvre ta webcam et te permet de capturer des images
pour chaque geste. Les images sont sauvegardées automatiquement
dans les bons dossiers.

GESTES À CAPTURER :
  [1] LEFT    — main à gauche (pointez vers la gauche)
  [2] RIGHT   — main à droite (pointez vers la droite)
  [3] FIRE    — paume ouverte face caméra
  [4] ENTER   — poing fermé
  [5] NEUTRAL — pas de geste (fond, main baissée, etc.)

CONTRÔLES :
  Appuie sur 1/2/3/4/5  → commence à capturer cette classe
  Appuie sur ESPACE      → capture une seule image manuellement
  Appuie sur A           → capture automatique (toutes les 0.5s)
  Appuie sur S           → arrête la capture automatique
  Appuie sur Q           → quitter

CONSEIL :
  - Vise 200-300 images par classe
  - Varie la position, l'angle, la distance et l'éclairage
  - Bouge légèrement la main pour avoir de la diversité

Installation :
  pip install opencv-python numpy
"""

import cv2
import os
import time
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
IMG_SIZE   = 64          # taille des images sauvegardées (64x64 pixels)
AUTO_DELAY = 0.4         # secondes entre 2 captures automatiques

CLASSES = {
    '1': 'LEFT',
    '2': 'RIGHT',
    '3': 'FIRE',
    '4': 'ENTER',
    '5': 'NEUTRAL',
}

# Couleur affichée pour chaque classe
COLORS = {
    'LEFT'   : (150, 150, 255),
    'RIGHT'  : (150, 255, 150),
    'FIRE'   : (0,   255, 255),
    'ENTER'  : (255, 200,  50),
    'NEUTRAL': (180, 180, 180),
}

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def compter_images(classe):
    """Retourne le nombre d'images déjà dans le dossier d'une classe."""
    dossier = os.path.join(DATA_DIR, classe)
    if not os.path.exists(dossier):
        return 0
    return len([f for f in os.listdir(dossier) if f.endswith('.jpg')])


def sauvegarder_image(frame, classe):
    """Redimensionne et sauvegarde une image dans le bon dossier."""
    dossier = os.path.join(DATA_DIR, classe)
    # Redimensionner en carré 64x64
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    # Nom de fichier unique basé sur le timestamp
    nom = f"{classe}_{int(time.time() * 1000)}.jpg"
    chemin = os.path.join(dossier, nom)
    cv2.imwrite(chemin, img)
    return chemin


def dessiner_interface(frame, classe_active, auto_mode, compteurs):
    """Affiche les informations sur le frame."""
    h, w = frame.shape[:2]

    # Fond semi-transparent en haut
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 140), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Titre
    cv2.putText(frame, "COLLECTE D'IMAGES", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Classe active
    if classe_active:
        couleur = COLORS[classe_active]
        statut  = "AUTO" if auto_mode else "PRET"
        cv2.putText(frame, f"Classe : {classe_active}  [{statut}]",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, couleur, 2)
    else:
        cv2.putText(frame, "Appuie sur 1/2/3/4/5 pour choisir une classe",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Compteurs par classe
    x = 10
    for i, (key, cls) in enumerate(CLASSES.items()):
        n     = compteurs[cls]
        coul  = COLORS[cls]
        barre = min(int(n / 300 * 100), 100)  # barre de progression sur 300 images
        texte = f"[{key}] {cls:<8} {n:>3}"
        cv2.putText(frame, texte, (x, 100 + i * 0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, coul, 1)
        x += 130

    # Compteurs ligne 2
    cv2.putText(frame, "  ".join(
        f"[{k}]{v}={compteurs[v]}" for k, v in CLASSES.items()),
        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    # Instructions bas
    cv2.putText(frame, "ESPACE=capturer  A=auto  S=stop auto  Q=quitter",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)

    # Flash rouge si capture en cours
    if auto_mode and classe_active:
        cv2.circle(frame, (w - 20, 20), 10, (0, 0, 255), -1)


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERREUR : impossible d'ouvrir la webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    classe_active = None
    auto_mode     = False
    derniere_capture = 0

    print("=== COLLECTE D'IMAGES ===")
    print("Appuie sur 1=LEFT  2=RIGHT  3=FIRE  4=ENTER  5=NEUTRAL")
    print("Appuie sur A pour la capture automatique, S pour arrêter")
    print("Appuie sur Q pour quitter\n")

    # Afficher le nombre d'images existantes
    compteurs = {cls: compter_images(cls) for cls in CLASSES.values()}
    for cls, n in compteurs.items():
        print(f"  {cls:<8} : {n} images existantes")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)   # miroir
        compteurs = {cls: compter_images(cls) for cls in CLASSES.values()}

        # --- Capture automatique ---
        now = time.time()
        if auto_mode and classe_active:
            if now - derniere_capture >= AUTO_DELAY:
                chemin = sauvegarder_image(frame, classe_active)
                derniere_capture = now
                n = compteurs[classe_active]
                print(f"  Sauvegardé [{classe_active}] #{n}  → {os.path.basename(chemin)}")

        dessiner_interface(frame, classe_active, auto_mode, compteurs)
        cv2.imshow("Collecte d'images - Space Invaders CNN", frame)

        key = cv2.waitKey(1) & 0xFF

        # Choisir classe
        if chr(key) in CLASSES:
            classe_active = CLASSES[chr(key)]
            auto_mode     = False
            print(f"\nClasse sélectionnée : {classe_active}")

        # Capture manuelle
        elif key == ord(' ') and classe_active:
            chemin = sauvegarder_image(frame, classe_active)
            compteurs[classe_active] += 1
            print(f"  Sauvegardé [{classe_active}] #{compteurs[classe_active]}")

        # Démarrer auto
        elif key == ord('a') and classe_active:
            auto_mode = True
            print(f"  Capture AUTO activée pour {classe_active}")

        # Arrêter auto
        elif key == ord('s'):
            auto_mode = False
            print("  Capture AUTO arrêtée")

        # Quitter
        elif key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n=== RÉSUMÉ FINAL ===")
    total = 0
    for cls in CLASSES.values():
        n = compter_images(cls)
        total += n
        barre = '█' * (n // 10) + '░' * max(0, 30 - n // 10)
        print(f"  {cls:<8} : {n:>4} images  {barre}")
    print(f"\n  TOTAL : {total} images")
    if total >= 500:
        print("  Prêt pour l'entraînement ! Lance : python 2_entrainer_cnn.py")
    else:
        manquant = 500 - total
        print(f"  Collecte encore ~{manquant} images pour de bons résultats.")


if __name__ == "__main__":
    main()
