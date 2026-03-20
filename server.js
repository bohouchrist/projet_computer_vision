const WebSocket = require('ws');

// Create a WebSocket server on port 8765
const wss = new WebSocket.Server({ port: 8765 });

console.log('WebSocket server running on ws://localhost:8765');

// Key code mapping
const KEYCODES = {
  'LEFT' : 37,
  'RIGHT': 39,
  'FIRE' : 32,
  'ENTER': 13
};

// Commandes de mouvement continu (tenue tant que les commandes arrivent)
const HOLD_COMMANDS = new Set(['LEFT', 'RIGHT', 'FIRE']);
// Délai avant de relâcher si aucune nouvelle commande n'arrive (ms)
const HOLD_TIMEOUT = 180;

// Broadcast à tous les clients sauf l'expéditeur
function broadcast(wss, sender, data) {
  const msg = JSON.stringify(data);
  wss.clients.forEach((client) => {
    if (client !== sender && client.readyState === WebSocket.OPEN) {
      client.send(msg);
    }
  });
}

wss.on('connection', (ws) => {
  console.log('Client connected');

  // Timers de relâchement par commande (pour cette connexion)
  const holdTimers  = {};  // setTimeout handle
  const heldKeys    = {};  // true si keydown déjà envoyé

  ws.on('message', (message) => {
    const command = message.toString().trim().toUpperCase();
    console.log(`Received: ${command}`);

    if (!KEYCODES[command]) {
      console.log(`Unknown command: ${command}`);
      return;
    }

    const keyCode = KEYCODES[command];

    if (HOLD_COMMANDS.has(command)) {
      // --- Commande tenue (LEFT, RIGHT, FIRE) ---
      // Envoyer keydown seulement si la touche n'est pas déjà enfoncée
      if (!heldKeys[command]) {
        broadcast(wss, ws, { type: 'keydown', keyCode });
        heldKeys[command] = true;
      }
      // Réinitialiser le timer de relâchement
      clearTimeout(holdTimers[command]);
      holdTimers[command] = setTimeout(() => {
        broadcast(wss, ws, { type: 'keyup', keyCode });
        heldKeys[command] = false;
      }, HOLD_TIMEOUT);

    } else {
      // --- Commande unique (ENTER) ---
      broadcast(wss, ws, { type: 'keydown', keyCode });
      setTimeout(() => {
        broadcast(wss, ws, { type: 'keyup', keyCode });
      }, 100);
    }
  });

  ws.on('close', () => {
    // Relâcher toutes les touches tenues à la déconnexion
    Object.entries(heldKeys).forEach(([cmd, held]) => {
      if (held) {
        clearTimeout(holdTimers[cmd]);
        broadcast(wss, ws, { type: 'keyup', keyCode: KEYCODES[cmd] });
      }
    });
    console.log('Client disconnected');
  });
}); 