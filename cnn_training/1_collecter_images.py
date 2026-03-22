# script pour collecter les images d'entrainement
# on prend des photos de nos gestes avec la webcam
# Christ Bohou - projet vision par ordinateur M2

import cv2
import os
import time

# dossier ou on sauvegarde les images
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
IMG_SIZE = 64       # taille des images 64x64
DELAI    = 0.4      # temps entre chaque capture auto en secondes

# nos 5 gestes
CLASSES = {
    '1': 'LEFT',
    '2': 'RIGHT',
    '3': 'FIRE',
    '4': 'ENTER',
    '5': 'NEUTRAL',
}

COULEURS = {
    'LEFT'   : (150, 150, 255),
    'RIGHT'  : (150, 255, 150),
    'FIRE'   : (0,   255, 255),
    'ENTER'  : (255, 200,  50),
    'NEUTRAL': (180, 180, 180),
}

def compter(classe):
    dossier = os.path.join(DATA_DIR, classe)
    if not os.path.exists(dossier):
        return 0
    return len([f for f in os.listdir(dossier) if f.endswith('.jpg')])

def sauvegarder(frame, classe):
    dossier = os.path.join(DATA_DIR, classe)
    os.makedirs(dossier, exist_ok=True)
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    nom = f"{classe}_{int(time.time()*1000)}.jpg"
    cv2.imwrite(os.path.join(dossier, nom), img)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("erreur : webcam pas accessible")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    classe_active = 'LEFT'
    auto_mode = False
    derniere = 0

    print("touches : 1=LEFT 2=RIGHT 3=FIRE 4=ENTER 5=NEUTRAL")
    print("A = capture auto | S = stop | ESPACE = capture manuelle | Q = quitter")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # capture automatique si mode actif
        now = time.time()
        if auto_mode and classe_active:
            if now - derniere >= DELAI:
                sauvegarder(frame, classe_active)
                derniere = now
                print(f"  capture {classe_active} #{compter(classe_active)}")

        # affichage basique
        couleur = COULEURS[classe_active]
        cv2.rectangle(frame, (0, 0), (w, 60), (20, 20, 20), -1)
        cv2.putText(frame, f"classe: {classe_active}  {'AUTO' if auto_mode else 'manuel'}",
                    (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, couleur, 2)

        # compteurs en bas
        txt = "  |  ".join(f"{c}:{compter(c)}" for c in CLASSES.values())
        cv2.putText(frame, txt, (10, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        if auto_mode:
            cv2.circle(frame, (w-20, 20), 10, (0, 0, 255), -1)

        cv2.imshow("collecte images", frame)
        key = cv2.waitKey(1) & 0xFF

        if chr(key) in CLASSES:
            classe_active = CLASSES[chr(key)]
            auto_mode = False
            print(f"\n-> classe : {classe_active}")
        elif key == ord(' ') and classe_active:
            sauvegarder(frame, classe_active)
            print(f"  capture manuelle {classe_active} #{compter(classe_active)}")
        elif key == ord('a'):
            auto_mode = True
            print(f"auto ON -> {classe_active}")
        elif key == ord('s'):
            auto_mode = False
            print("auto OFF")
        elif key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n--- bilan ---")
    for cls in CLASSES.values():
        print(f"  {cls} : {compter(cls)} images")

if __name__ == "__main__":
    main()
