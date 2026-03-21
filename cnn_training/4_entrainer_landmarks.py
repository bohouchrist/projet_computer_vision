"""
4_entrainer_landmarks.py — Entraîner un réseau dense sur les landmarks

Pourquoi ça marche mieux que le CNN sur images brutes :
  - Entrée : 63 coordonnées normalisées (forme pure de la main)
  - Zéro bruit de fond, zéro éclairage, zéro couleur de peau
  - Réseau beaucoup plus petit → entraîne en quelques secondes sur CPU
  - Précision attendue : 95%+

Architecture :
  63 → Dense(128, ReLU) → Dropout → Dense(64, ReLU) → Dropout → Dense(5, Softmax)
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras

# ============================================================
# CONFIGURATION
# ============================================================
CSV_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_landmarks", "landmarks.csv")
MODEL_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modeles")
MODEL_PATH  = os.path.join(MODEL_DIR, "gestes_landmarks.keras")
CLASSES_PATH= os.path.join(MODEL_DIR, "classes_landmarks.txt")
COURBES_PATH= os.path.join(MODEL_DIR, "courbes_landmarks.png")

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================
print("=" * 50)
print("  ENTRAÎNEMENT — Réseau Dense sur Landmarks")
print("=" * 50)

if not os.path.exists(CSV_PATH):
    print(f"ERREUR : fichier CSV introuvable : {CSV_PATH}")
    print("Lance d'abord : python 3_collecter_landmarks.py")
    exit(1)

df = pd.read_csv(CSV_PATH)
print(f"\nDonnées chargées : {len(df)} échantillons")
print("\nRépartition par classe :")
for cls, cnt in df['classe'].value_counts().sort_index().items():
    barre = '█' * (cnt // 5)
    print(f"  {cls:<8}: {cnt:>5}  {barre}")

# Séparer features et labels
X = df.drop('classe', axis=1).values.astype(np.float32)
y = df['classe'].values

# Encoder les labels
encoder = LabelEncoder()
y_enc   = encoder.fit_transform(y)
classes = encoder.classes_
n_classes = len(classes)

print(f"\nClasses : {list(classes)}")
print(f"Taille entrée : {X.shape[1]} features (21 points × 3 coordonnées)")

# Sauvegarder les classes
with open(CLASSES_PATH, 'w') as f:
    for c in classes:
        f.write(c + '\n')
print(f"Classes sauvegardées : {CLASSES_PATH}")

# Split train / validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
print(f"\nEntraînement : {len(X_train)} échantillons")
print(f"Validation   : {len(X_val)} échantillons")

# ============================================================
# ARCHITECTURE DU RÉSEAU DENSE
# ============================================================
print("\nConstruction du réseau dense...")

model = keras.Sequential([
    keras.Input(shape=(X.shape[1],)),

    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(64, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(n_classes, activation='softmax'),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

model.summary()

# ============================================================
# CALLBACKS
# ============================================================
callbacks = [
    keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1,
    ),
]

# ============================================================
# ENTRAÎNEMENT
# ============================================================
print(f"\nDébut de l'entraînement (50 époques max)...")

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1,
)

# ============================================================
# RÉSULTATS
# ============================================================
best_val_acc = max(history.history['val_accuracy'])
print("\n" + "=" * 50)
print(f"  Meilleure précision validation : {best_val_acc*100:.1f}%")
print(f"  Modèle sauvegardé : {MODEL_PATH}")
print("=" * 50)

if best_val_acc >= 0.90:
    print("  ✓ Excellent ! Prêt à jouer.")
elif best_val_acc >= 0.75:
    print("  ~ Correct. Collecte quelques landmarks supplémentaires pour améliorer.")
else:
    print("  ✗ À améliorer. Collecte plus de données variées.")

# ============================================================
# ÉVALUATION DÉTAILLÉE
# ============================================================
print("\nÉvaluation sur la validation :")
y_pred    = np.argmax(model.predict(X_val, verbose=0), axis=1)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_val, y_pred, target_names=classes))

# Matrice de confusion
cm = confusion_matrix(y_val, y_pred)
print("Matrice de confusion :")
print(f"{'':>10}", end='')
for c in classes:
    print(f"{c:>10}", end='')
print()
for i, row in enumerate(cm):
    print(f"{classes[i]:>10}", end='')
    for val in row:
        print(f"{val:>10}", end='')
    print()

# ============================================================
# GRAPHIQUES
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Entraînement — Réseau Dense sur Landmarks MediaPipe", fontsize=13)

axes[0].plot(history.history['accuracy'],     label='Entraînement', color='steelblue')
axes[0].plot(history.history['val_accuracy'], label='Validation',   color='orange')
axes[0].set_title('Précision (Accuracy)')
axes[0].set_xlabel('Époque')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0, 1])

axes[1].plot(history.history['loss'],     label='Entraînement', color='steelblue')
axes[1].plot(history.history['val_loss'], label='Validation',   color='orange')
axes[1].set_title('Erreur (Loss)')
axes[1].set_xlabel('Époque')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(COURBES_PATH, dpi=120)
plt.show()
print(f"\nGraphiques sauvegardés : {COURBES_PATH}")
print("\nLance maintenant : python 5_jouer_avec_cnn.py")
