# ============================================================
#  LANCER SPACE INVADERS - Script de demarrage
# ============================================================

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "   SPACE INVADERS - Lancement automatique  " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Aller dans le bon dossier
Set-Location $PSScriptRoot

# --- Liberer les ports si deja occupes ---
Write-Host "[1/4] Liberation des ports..." -ForegroundColor Yellow
$ports = @(8765, 8000)
foreach ($port in $ports) {
    $pid_found = (netstat -ano | Select-String ":$port ") -replace '.*LISTENING\s+', '' | Select-Object -First 1
    if ($pid_found) {
        taskkill /PID $pid_found /F 2>$null | Out-Null
        Write-Host "      Port $port libere (PID $pid_found)" -ForegroundColor Gray
    }
}
Start-Sleep -Milliseconds 500

# --- Terminal 1 : Serveur WebSocket ---
Write-Host "[2/4] Demarrage du serveur WebSocket (node server.js)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "Set-Location '$PSScriptRoot'; Write-Host 'WEBSOCKET SERVER' -ForegroundColor Cyan; node server.js"

Start-Sleep -Seconds 2

# --- Terminal 2 : Serveur HTTP du jeu ---
Write-Host "[3/4] Demarrage du serveur HTTP (port 8000)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "Set-Location '$PSScriptRoot'; Write-Host 'SERVEUR HTTP' -ForegroundColor Green; python -m http.server 8000"

Start-Sleep -Seconds 1

# --- Ouvrir le navigateur ---
Write-Host "[4/4] Ouverture du jeu dans le navigateur..." -ForegroundColor Yellow
Start-Process "http://localhost:8000"

Start-Sleep -Seconds 1

# --- Terminal 3 : Controle par webcam ---
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Jeu ouvert ! Lancement de la webcam...   " -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Appuie sur 'Q' dans la fenetre webcam pour quitter" -ForegroundColor Gray
Write-Host ""

# Lancer la webcam dans ce terminal
python cv_control_mediapipe.py
