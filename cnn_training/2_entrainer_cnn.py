"""
ÉTAPE 2 — Entraînement du CNN
==============================

Ce script lit les images que tu as collectées et entraîne
un réseau de neurones convolutif (CNN) pour reconnaître
tes gestes.

ARCHITECTURE :
  - Entrée  : image 64×64 pixels en couleur (RGB)
  - Couches : 3 blocs Conv2D + MaxPooling + BatchNorm
  - Sortie  : 5 classes (LEFT, RIGHT, FIRE, ENTER, NEUTRAL)

COMMENT ÇA MARCHE (rappel) :
  1. Les couches Conv2D détectent des formes (bords, courbes, doigts)
  2. MaxPooling réduit la taille et garde l'essentiel
  3. BatchNorm stabilise l'apprentissage
  4. Les couches Dense font la classification finale
  5. On divise les données : 80% entraînement, 20% validation

Le modèle entraîné est sauvegardé dans : modeles/gestes_cnn.keras

Installation :
  pip install tensorflow opencv-python numpy matplotlib scikit-learn
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # réduit les logs TensorFlow
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modeles")
MODEL_PATH = os.path.join(MODEL_DIR, "gestes_cnn.keras")

CLASSES   = ['LEFT', 'RIGHT', 'FIRE', 'ENTER', 'NEUTRAL']
IMG_SIZE  = 64       # doit correspondre à 1_collecter_images.py
BATCH     = 32       # nombre d'images traitées en même temps
EPOCHS    = 30       # passages complets sur les données

# ============================================================
# ÉTAPE A — CHARGEMENT DES IMAGES
# ============================================================

def charger_donnees():
    """Lit toutes les images et retourne (X, y) avec les labels."""
    images, labels = [], []
    print("Chargement des images...")

    for classe in CLASSES:
        dossier = os.path.join(DATA_DIR, classe)
        if not os.path.exists(dossier):
            print(f"  [!] Dossier manquant : {dossier}")
            continue

        fichiers = [f for f in os.listdir(dossier) if f.endswith('.jpg')]
        if not fichiers:
            print(f"  [!] Aucune image dans {classe}")
            continue

        for nom in fichiers:
            chemin = os.path.join(dossier, nom)
            img    = cv2.imread(chemin)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # BGR → RGB
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))   # redimensionner si besoin
            images.append(img)
            labels.append(classe)

        print(f"  {classe:<8} : {len(fichiers)} images chargées")

    if not images:
        raise ValueError("Aucune image trouvée ! Lance d'abord 1_collecter_images.py")

    X = np.array(images, dtype='float32') / 255.0   # normaliser [0, 1]
    y = np.array(labels)
    print(f"\nTotal : {len(X)} images  |  Classes : {np.unique(y)}\n")
    return X, y


# ============================================================
# ÉTAPE B — CONSTRUCTION DU CNN
# ============================================================

def construire_cnn(n_classes):
    """
    Architecture CNN simple et efficace pour la classification de gestes.

    Bloc 1 : détecte les contours et textures simples
    Bloc 2 : détecte les formes (doigts, paume)
    Bloc 3 : détecte les patterns complexes (configuration main)
    """
    modele = models.Sequential([

        # --- Bloc 1 ---
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # --- Bloc 2 ---
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # --- Bloc 3 ---
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # --- Classification ---
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(n_classes, activation='softmax'),   # sortie : probabilité par classe
    ])

    modele.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return modele


# ============================================================
# ÉTAPE C — DATA AUGMENTATION (diversifier les données)
# ============================================================

def creer_augmentation():
    """
    Augmentation = créer des variantes de chaque image pour rendre
    le modèle plus robuste (rotation, zoom, flip, luminosité).
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ])


# ============================================================
# ÉTAPE D — ENTRAÎNEMENT
# ============================================================

def afficher_courbes(historique):
    """Affiche les courbes d'apprentissage (accuracy et loss)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(historique.history['accuracy'],     label='Entraînement')
    ax1.plot(historique.history['val_accuracy'], label='Validation')
    ax1.set_title('Précision (Accuracy)')
    ax1.set_xlabel('Époque')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(historique.history['loss'],     label='Entraînement')
    ax2.plot(historique.history['val_loss'], label='Validation')
    ax2.set_title('Erreur (Loss)')
    ax2.set_xlabel('Époque')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    graphique_path = os.path.join(MODEL_DIR, "courbes_entrainement.png")
    plt.savefig(graphique_path)
    print(f"\nGraphiques sauvegardés : {graphique_path}")
    plt.show()


def main():
    print("=" * 50)
    print("  ENTRAÎNEMENT DU CNN — Space Invaders Gestes")
    print("=" * 50 + "\n")

    # A — Charger les données
    X, y_texte = charger_donnees()

    # Encoder les labels texte → entiers (LEFT=0, RIGHT=1, etc.)
    encoder = LabelEncoder()
    encoder.fit(CLASSES)   # ordre fixe des classes
    y = encoder.transform(y_texte)
    n_classes = len(CLASSES)

    print(f"Correspondance classes : { {c: i for i, c in enumerate(encoder.classes_)} }\n")

    # Sauvegarder l'ordre des classes pour l'inférence
    classes_path = os.path.join(MODEL_DIR, "classes.txt")
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(classes_path, 'w') as f:
        f.write('\n'.join(encoder.classes_))
    print(f"Classes sauvegardées : {classes_path}")

    # B — Diviser : 80% train, 20% validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Entraînement : {len(X_train)} images")
    print(f"Validation   : {len(X_val)} images\n")

    # C — Construire le modèle
    modele = construire_cnn(n_classes)
    modele.summary()

    # D — Augmentation des données
    augmentation = creer_augmentation()

    # Créer les datasets TensorFlow
    ds_train = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(len(X_train))
        .batch(BATCH)
        .map(lambda x, y: (augmentation(x, training=True), y),
             num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    ds_val = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(BATCH)
        .prefetch(tf.data.AUTOTUNE)
    )

    # E — Callbacks (arrêt anticipé + sauvegarde du meilleur modèle)
    cb_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=8,           # arrête si pas d'amélioration pendant 8 époques
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    # F — Entraîner
    print(f"\nDébut de l'entraînement ({EPOCHS} époques max)...\n")
    historique = modele.fit(
        ds_train,
        validation_data=ds_val,
        epochs=EPOCHS,
        callbacks=cb_list,
    )

    # G — Résultats finaux
    val_acc = max(historique.history['val_accuracy'])
    print(f"\n{'='*50}")
    print(f"  Meilleure précision en validation : {val_acc*100:.1f}%")
    print(f"  Modèle sauvegardé : {MODEL_PATH}")
    print(f"{'='*50}\n")

    if val_acc >= 0.90:
        print("  Excellent ! Lance 3_tester_modele.py pour tester en temps réel.")
    elif val_acc >= 0.75:
        print("  Bon résultat. Tu peux tester, mais collecte plus d'images pour améliorer.")
    else:
        print("  Résultat à améliorer. Collecte plus d'images variées et relance.")

    afficher_courbes(historique)


if __name__ == "__main__":
    main()
