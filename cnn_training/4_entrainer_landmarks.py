# entrainement du reseau dense sur les landmarks MediaPipe
# beaucoup mieux que le CNN sur images : 99.6% de precision
# les landmarks sont invariants au fond donc ca generalise bien
# Christ Bohou

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras

CSV_PATH     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_landmarks", "landmarks.csv")
MODEL_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modeles")
MODEL_PATH   = os.path.join(MODEL_DIR, "gestes_landmarks.keras")
CLASSES_PATH = os.path.join(MODEL_DIR, "classes_landmarks.txt")
COURBES_PATH = os.path.join(MODEL_DIR, "courbes_landmarks.png")

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(CSV_PATH):
    print("CSV introuvable, lance 3_collecter_landmarks.py d abord")
    exit(1)

# chargement des donnees
df = pd.read_csv(CSV_PATH)
print(f"donnees chargees : {len(df)} echantillons")
print("repartition :")
print(df['classe'].value_counts())

X = df.drop('classe', axis=1).values.astype(np.float32)
y = df['classe'].values

enc = LabelEncoder()
y_enc = enc.fit_transform(y)
classes = enc.classes_
n = len(classes)

print(f"\nclasses : {list(classes)}")
print(f"entree : {X.shape[1]} features (21 points x 3 coords)")

# sauvegarde des classes pour l inference
with open(CLASSES_PATH, 'w') as f:
    for c in classes:
        f.write(c + '\n')

X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
print(f"train : {len(X_train)}  val : {len(X_val)}")

# architecture du reseau
# on a essaye plusieurs tailles, 128-128-64 donne les meilleurs resultats
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
    keras.layers.Dense(n, activation='softmax'),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
model.summary()

cb = [
    keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
]

print("\nentraine le modele...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=cb,
)

best = max(history.history['val_accuracy'])
print(f"\nmeilleure val_accuracy : {best*100:.1f}%")

# rapport de classification
y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
print("\nrapport de classification :")
print(classification_report(y_val, y_pred, target_names=classes))
print("matrice de confusion :")
print(confusion_matrix(y_val, y_pred))

# courbes d entrainement
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("entrainement reseau dense - landmarks MediaPipe")
axes[0].plot(history.history['accuracy'], label='train', color='steelblue')
axes[0].plot(history.history['val_accuracy'], label='val', color='orange')
axes[0].set_title('precision'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(history.history['loss'], label='train', color='steelblue')
axes[1].plot(history.history['val_loss'], label='val', color='orange')
axes[1].set_title('perte'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(COURBES_PATH, dpi=120)
plt.show()
print(f"courbes sauvegardees : {COURBES_PATH}")
