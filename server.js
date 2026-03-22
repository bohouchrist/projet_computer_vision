// serveur websocket qui fait le lien entre python et le jeu
// recoit les commandes de gestes et simule les touches clavier
// Christ Bohou & Djafarou Oulare

const WebSocket = require('ws');

const wss = new WebSocket.Server({ port: 8765 });
console.log('serveur websocket sur ws://localhost:8765');

// correspondance geste -> code de touche
const TOUCHES = {
  'LEFT' : 37,
  'RIGHT': 39,
  'FIRE' : 32,
  'ENTER': 13
};

// ces commandes restent actives tant qu elles arrivent
const COMMANDES_CONTINUES = new Set(['LEFT', 'RIGHT', 'FIRE']);
const DELAI_RELACHE = 180; // ms avant de relacher la touche

function diffuser(expéditeur, data) {
  const msg = JSON.stringify(data);
  wss.clients.forEach(client => {
    if (client !== expéditeur && client.readyState === WebSocket.OPEN) {
      client.send(msg);
    }
  });
}

wss.on('connection', (ws) => {
  console.log('client connecte');
  const timers = {};
  const tenues = {};

  ws.on('message', (message) => {
    const cmd = message.toString().trim().toUpperCase();
    // on ignore les commandes inconnues
    if (!TOUCHES[cmd]) return;

    const code = TOUCHES[cmd];

    if (COMMANDES_CONTINUES.has(cmd)) {
      // si la touche est pas deja enfoncee on envoie keydown
      if (!tenues[cmd]) {
        diffuser(ws, { type: 'keydown', keyCode: code });
        tenues[cmd] = true;
      }
      // reset du timer a chaque nouvelle commande
      clearTimeout(timers[cmd]);
      timers[cmd] = setTimeout(() => {
        diffuser(ws, { type: 'keyup', keyCode: code });
        tenues[cmd] = false;
      }, DELAI_RELACHE);
    } else {
      // pour ENTER : juste un appui court
      diffuser(ws, { type: 'keydown', keyCode: code });
      setTimeout(() => diffuser(ws, { type: 'keyup', keyCode: code }), 100);
    }
  });

  ws.on('close', () => {
    // on relache toutes les touches quand le client se deconnecte
    Object.entries(tenues).forEach(([cmd, active]) => {
      if (active) {
        clearTimeout(timers[cmd]);
        diffuser(ws, { type: 'keyup', keyCode: TOUCHES[cmd] });
      }
    });
    console.log('client deconnecte');
  });
});
