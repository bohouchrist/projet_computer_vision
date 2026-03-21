# Space Invaders — Contrôle par Gestes 🎮🖐️

Joue à Space Invaders avec tes mains via la webcam !
Le jeu détecte tes gestes en temps réel grâce à **MediaPipe** et les traduit en commandes de jeu.

---

## Prérequis

- **Node.js** — [nodejs.org](https://nodejs.org/)
- **Python 3.10+** — [python.org](https://www.python.org/)
- Une **webcam**

---

## Installation

### 1. Cloner le projet
```powershell
git clone https://github.com/bohouchrist/projet_computer_vision.git
cd projet_computer_vision
```

### 2. Installer les dépendances Node.js
```powershell
npm install
```

### 3. Installer les dépendances Python
```powershell
pip install mediapipe opencv-python websockets numpy
```

---

## Lancer le jeu

Ouvre **3 terminaux PowerShell** dans le dossier du projet et tape une commande dans chacun :

**Terminal 1 — Serveur WebSocket**
```powershell
node server.js
```

**Terminal 2 — Détection des gestes (webcam)**
```powershell
python cv_control_mediapipe.py
```

**Terminal 3 — Serveur du jeu**
```powershell
python -m http.server 8000
```

Puis ouvre ton navigateur sur :
```
http://localhost:8000
```

> Les 3 terminaux doivent rester ouverts pendant toute la partie.
> Pour arrêter la détection : appuie sur **Q** dans la fenêtre webcam.

---

## Gestes pour jouer

> 👉 **Guide visuel interactif** avec squelettes de mains animés :
> ouvre `comment_jouer.html` dans le navigateur après avoir lancé le serveur HTTP.

| Geste | Commande | Effet |
|---|---|---|
| ☝️ Index pointé à **gauche** | `LEFT` | Déplacer le vaisseau à gauche |
| ☝️ Index pointé à **droite** | `RIGHT` | Déplacer le vaisseau à droite |
| ✊ **Poing fermé vers l'avant** (knuckles face caméra) | `FIRE` | Tirer un missile |
| 🖐️ **Paume ouverte** (5 doigts écartés) | `ENTER` | Démarrer / valider |
| 🤷 Tout autre geste | `NEUTRAL` | Aucune action |

> La fenêtre webcam montre en temps réel le geste détecté et la zone dans laquelle se trouve ta main.

---

## Architecture du projet

```
projet_computer_vision/
│
├── index.html               # Page du jeu (chargée dans le navigateur)
├── comment_jouer.html       # Guide interactif — gestes animés en 3D
├── game.bundle.js           # Jeu Space Invaders (JavaScript bundlé)
├── server.js                # Serveur WebSocket — relaie les commandes au jeu
│
├── cv_control_mediapipe.py  # Détection gestes par webcam (MediaPipe)
│
├── core/                    # Moteur du jeu
├── entities/                # Entités (joueur, envahisseurs, balles)
└── utils/                   # Utilitaires (input, canvas, score...)
```

### Comment ça fonctionne

```
Webcam → MediaPipe (détection main) → Python → WebSocket → Node.js → Navigateur (jeu)
```

1. **MediaPipe** détecte les 21 points de ta main en temps réel
2. **Python** analyse la position et la forme de ta main → détermine le geste
3. La commande (`LEFT`, `RIGHT`, `FIRE`, `ENTER`) est envoyée via **WebSocket**
4. **Node.js** la reçoit et simule l'appui sur la touche correspondante
5. Le **jeu** réagit comme si tu avais appuyé sur le clavier

---

## Contrôles clavier alternatifs

Tu peux aussi jouer au clavier directement dans le navigateur :

| Touche | Action |
|---|---|
| `←` Flèche gauche | Déplacer à gauche |
| `→` Flèche droite | Déplacer à droite |
| `Espace` | Tirer |
| `Entrée` | Démarrer |

---

## Dépannage

**La webcam ne s'ouvre pas**
```powershell
# Changer l'index de la caméra dans cv_control_mediapipe.py
CAMERA_INDEX = 1  # essaie 0, 1 ou 2
```

**Erreur `EADDRINUSE port 8765`** — Un ancien serveur tourne encore :
```powershell
# Trouver et tuer le processus
netstat -ano | findstr :8765
taskkill /PID <le_numero> /F
```

**Le modèle MediaPipe se télécharge automatiquement** au premier lancement (~8 MB).
Si le téléchargement échoue, télécharge manuellement `hand_landmarker.task` depuis :
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

---

## Projet académique

Ce projet a été réalisé dans le cadre d'un cours d'initiation à la **Vision par Ordinateur**.
Le dossier d'entraînement du modèle CNN et le rapport complet sont disponibles ici :
👉 [github.com/bohouchrist/cnn_training](https://github.com/bohouchrist/cnn_training)
