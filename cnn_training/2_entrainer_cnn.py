# entrainement du CNN sur les images brutes
# premiere approche : on donne directement les pixels au reseau
# resultat pas terrible (34%) mais on a essaye
# Christ Bohou

import os, numpy as np, cv2, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modeles")
MODEL_PATH = os.path.join(MODEL_DIR, "gestes_cnn.keras")
os.makedirs(MODEL_DIR, exist_ok=True)

CLASSES  = ['LEFT', 'RIGHT', 'FIRE', 'ENTER', 'NEUTRAL']
IMG_SIZE = 64
BATCH    = 32
EPOCHS   = 30

def charger_images():
    X, y = [], []
    print("chargement des images...")
    for classe in CLASSES:
        dossier = os.path.join(DATA_DIR, classe)
        if not os.path.exists(dossier):
            print(f"  dossier manquant : {classe}")
            continue
        fichiers = [f for f in os.listdir(dossier) if f.endswith('.jpg')]
        for nom in fichiers:
            img = cv2.imread(os.path.join(dossier, nom))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(classe)
        print(f"  {classe} : {len(fichiers)} images")

    X = np.array(X, dtype='float32') / 255.0
    print(f"total : {len(X)} images")
    return X, np.array(y)

def construire_modele(n_classes):
    # 3 blocs conv + tete dense
    m = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(n_classes, activation='softmax'),
    ])
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

def main():
    X, y_txt = charger_images()
    if len(X) == 0:
        print("pas d images trouvees, lance 1_collecter_images.py d abord")
        return

    enc = LabelEncoder()
    enc.fit(CLASSES)
    y = enc.transform(y_txt)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"train : {len(X_train)}  val : {len(X_val)}")

    modele = construire_modele(len(CLASSES))
    modele.summary()

    # augmentation de donnees pour compenser le peu d images
    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ])

    ds_train = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(len(X_train)).batch(BATCH)
        .map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE))
    ds_val = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(BATCH).prefetch(tf.data.AUTOTUNE))

    cb = [
        callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
    ]

    print(f"\nentrainement ({EPOCHS} epochs max)...")
    hist = modele.fit(ds_train, validation_data=ds_val, epochs=EPOCHS, callbacks=cb)

    best = max(hist.history['val_accuracy'])
    print(f"\nmeilleure val_accuracy : {best*100:.1f}%")
    print(f"modele sauvegarde : {MODEL_PATH}")

    # courbes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(hist.history['accuracy'], label='train')
    ax1.plot(hist.history['val_accuracy'], label='val')
    ax1.set_title('accuracy'); ax1.legend(); ax1.grid(True)
    ax2.plot(hist.history['loss'], label='train')
    ax2.plot(hist.history['val_loss'], label='val')
    ax2.set_title('loss'); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "courbes_entrainement.png"))
    plt.show()

if __name__ == "__main__":
    main()
