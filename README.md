# Space Invaders — Contrôle par Gestes de la Main 🎮🖐️

> Projet académique — Initiation à la Vision par Ordinateur
> Master Mathématiques — Mars 2026
> **CHRIST BOHOU**

---

## Présentation

Ce projet implémente un système de contrôle du jeu **Space Invaders** par reconnaissance de gestes de la main en temps réel. La webcam capte les mouvements, un réseau de neurones convolutif (CNN) entraîné sur des données personnelles classifie le geste détecté, et la commande correspondante est transmise au jeu via un pont WebSocket.

**Pipeline complet :**

```
Webcam → MediaPipe (21 landmarks) → Normalisation → CNN → WebSocket → Navigateur (jeu)
```

👉 **Guide visuel des gestes :** ouvrir [`comment_jouer.html`](comment_jouer.html) dans le navigateur après avoir lancé le serveur HTTP.

---

## Architecture du système

```
projet_computer_vision/
│
├── index.html                  # Page du jeu Space Invaders
├── comment_jouer.html          # Guide interactif des gestes (squelettes animés)
├── game.bundle.js              # Moteur du jeu (JavaScript)
├── server.js                   # Serveur WebSocket — relaie les commandes au jeu
├── hand_landmarker.task        # Modèle MediaPipe pré-entraîné (détection main)
│
├── core/                       # Boucle de jeu, rendu, collisions, mises à jour
├── entities/                   # Entités : joueur, envahisseurs, missiles, blocs
├── utils/                      # Utilitaires : input, canvas, score, sprites
│
└── cnn_training/               # Pipeline complet d'entraînement
    ├── 1_collecter_images.py       # Collecte d'images par geste (webcam)
    ├── 2_entrainer_cnn.py          # Entraînement CNN sur images 64×64
    ├── 3_collecter_landmarks.py    # Collecte de vecteurs landmarks par geste
    ├── 4_entrainer_landmarks.py    # Entraînement CNN sur landmarks (version retenue)
    ├── 5_jouer_avec_cnn.py         # Contrôle du jeu en temps réel ← version finale
    ├── test_cnn_local.py           # Test du modèle sans lancer le jeu
    └── modeles/
        ├── gestes_landmarks.keras  # Modèle retenu (CNN sur landmarks)
        ├── classes_landmarks.txt   # Ordre des classes
        └── courbes_landmarks.png   # Courbes d'apprentissage
```

---

## Fonctionnement du CNN

### 1. Détection de la main — MediaPipe

MediaPipe `HandLandmarker` détecte **21 points caractéristiques** (landmarks) de la main dans chaque frame vidéo : poignet, articulations et bouts des 5 doigts. Chaque point est un triplet $(x, y, z)$ en coordonnées normalisées dans l'image.

### 2. Normalisation du vecteur d'entrée

Pour que le modèle soit **invariant à la position et à la taille** de la main dans l'image, les landmarks sont normalisés avant toute prédiction :

```python
pts -= pts[0]          # Translation : le poignet devient l'origine (0, 0, 0)
pts /= max(|pts|)      # Mise à l'échelle : valeurs dans [−1, 1]
vecteur = pts.flatten()  # Vecteur de dimension 21 × 3 = 63
```

Ce prétraitement est identique lors de la collecte des données et lors de l'inférence — condition nécessaire pour que le modèle généralise correctement.

### 3. Architecture du réseau

Le modèle retenu est un **réseau entièrement connecté** (MLP/DNN) entraîné sur des vecteurs de landmarks plutôt que sur des images brutes. Ce choix est motivé par deux raisons :

- **Robustesse** : les landmarks sont invariants aux conditions d'éclairage et à la texture du fond — problèmes majeurs des CNN sur images brutes
- **Efficacité** : un vecteur de 63 scalaires suffit pour encoder la configuration géométrique de la main

```
Entrée : vecteur de 63 valeurs (21 points × 3 coordonnées)
    ↓
Dense(256, ReLU) + BatchNorm + Dropout(0.5)
    ↓
Dense(128, ReLU) + BatchNorm + Dropout(0.3)
    ↓
Dense(5, Softmax)  →  [ENTER, FIRE, LEFT, NEUTRAL, RIGHT]
```

La couche de sortie produit une **distribution de probabilité** sur les 5 classes. Le geste prédit est la classe de probabilité maximale, sous réserve d'un seuil de confiance à 0,70.

### 4. Entraînement

- **Données** : collectées manuellement via webcam (script `3_collecter_landmarks.py`)
- **Augmentation** : variations de position, d'angle et de distance à la caméra lors de la collecte
- **Optimiseur** : Adam — adaptatif, robuste au choix du taux d'apprentissage initial
- **Fonction de perte** : entropie croisée catégorielle (classification multi-classes)
- **Callbacks** : `EarlyStopping` + `ModelCheckpoint` — arrêt dès que la précision de validation stagne, sauvegarde du meilleur modèle uniquement
- **Séparation** : 80 % entraînement / 20 % validation, stratifiée par classe

---

## Gestes reconnus

| Geste | Action | Description |
|---|---|---|
| ☝️ Index pointé à **gauche** | `LEFT` | Main droite, index horizontal vers la gauche |
| ☝️ Index pointé à **droite** | `RIGHT` | Main gauche, index horizontal vers la droite |
| ✊ **Poing fermé** | `FIRE` | Main gauche, knuckles face à la caméra |
| 🖐️ **Paume ouverte** | `ENTER` | Main gauche, 5 doigts écartés face à la caméra |
| — | `NEUTRAL` | Tout autre geste — aucune action |

> Le seuil de confiance à **0,70** filtre les prédictions incertaines avant envoi.
> Un cooldown par classe (80 ms pour LEFT/RIGHT/FIRE, 800 ms pour ENTER) évite le spam de commandes.

---

## Prérequis

- **Node.js** ≥ 18 — [nodejs.org](https://nodejs.org/)
- **Python** ≥ 3.10 — [python.org](https://www.python.org/)
- **Webcam**

---

## Installation

```powershell
git clone https://github.com/bohouchrist/projet_computer_vision.git
cd projet_computer_vision

# Dépendances Node.js
npm install

# Dépendances Python
pip install tensorflow mediapipe opencv-python websocket-client numpy
```

---

## Lancer le jeu

Ouvre **3 terminaux PowerShell** dans le dossier du projet :

**Terminal 1 — Serveur WebSocket**
```powershell
node server.js
# → "WebSocket server listening on port 8765"
```

**Terminal 2 — Détection des gestes (CNN + webcam)**
```powershell
cd cnn_training
python 5_jouer_avec_cnn.py
# → fenêtre webcam avec prédictions en temps réel
```

**Terminal 3 — Serveur du jeu**
```powershell
python -m http.server 8000
```

Puis ouvrir dans le navigateur : **`http://localhost:8000`**

> Pour quitter : appuyer sur `Q` dans la fenêtre webcam.

### Tester le modèle sans lancer le jeu

```powershell
cd cnn_training
python test_cnn_local.py
```

---

## Résultats

Le modèle entraîné sur landmarks atteint une **précision de validation > 95 %** sur les 5 classes, avec une inférence stable à environ **25–30 fps** sur CPU.

| Observation | Explication |
|---|---|
| Bonne robustesse aux fonds variables | Les landmarks sont invariants au fond |
| FIRE moins fiable avec la main droite | Les données FIRE collectées avec la main gauche uniquement — le CNN n'a pas appris la configuration symétrique |
| Transitions LEFT↔RIGHT rapides | Le cooldown à 80 ms permet une réactivité suffisante pour le jeu |
| NEUTRAL bien isolé | La classe NEUTRAL absorbe les gestes ambigus et évite les faux positifs |

---

## Perspectives d'amélioration

### Amélioration de la robustesse

- **Collecter FIRE avec les deux mains** : les landmarks d'une main droite et d'une main gauche faisant le même geste sont symétriques (les coordonnées $x$ sont inversées après normalisation). Entraîner sur les deux mains doublerait la couverture sans modifier l'architecture.

- **Augmentation par symétrie** : inverser les coordonnées $x$ des vecteurs de landmarks d'une classe pour synthétiser les exemples de la main miroir — technique simple ne nécessitant pas de recollecte.

### Amélioration de la jouabilité

- **Actions simultanées** : la logique actuelle ne peut envoyer qu'une commande à la fois. Or le jeu (`player.js`) traite le mouvement et le tir de façon indépendante à chaque frame — il accepte nativement `LEFT + FIRE` simultanément. Une approche hybride position/forme permettrait de tirer en se déplaçant.

- **Deux mains** : détecter `num_hands=2` dans MediaPipe et affecter une main au mouvement et l'autre au tir — supprime le conflit fondamental du modèle mono-classe.

### Amélioration du modèle

- **Données plus nombreuses et variées** : collecte à différentes distances, angles et niveaux d'éclairage pour améliorer la généralisation.

- **Réseau récurrent (LSTM/GRU)** : exploiter la **séquence temporelle** des landmarks plutôt qu'une seule frame — permettrait de reconnaître des gestes dynamiques (ex. balayage) et de réduire les faux positifs sur les gestes statiques ambigus.

---

## Contrôles clavier alternatifs

| Touche | Action |
|---|---|
| `←` Flèche gauche | Déplacer à gauche |
| `→` Flèche droite | Déplacer à droite |
| `Espace` | Tirer |
| `Entrée` | Démarrer |
