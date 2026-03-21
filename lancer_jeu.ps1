# ============================================================
#  SPACE INVADERS - Lancement automatique
#  Usage : .\lancer_jeu.ps1
# ============================================================

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Clear-Host
Write-Host ""
Write-Host "  +==========================================+" -ForegroundColor Cyan
Write-Host "  |  SPACE INVADERS - Controle par gestes   |" -ForegroundColor Cyan
Write-Host "  +==========================================+" -ForegroundColor Cyan
Write-Host ""

# -- Etape 1 : Liberer les ports --
Write-Host "  [1/4] Liberation des ports 8765 et 8000..." -ForegroundColor Yellow

foreach ($port in @(8765, 8000)) {
    $pids = netstat -ano 2>$null |
            Select-String ":$port\s" |
            ForEach-Object { ($_ -split '\s+')[-1] } |
            Where-Object { $_ -match '^\d+$' } |
            Select-Object -Unique
    foreach ($p in $pids) {
        Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
        Write-Host "      Port $port libere (PID $p)" -ForegroundColor DarkGray
    }
}
Start-Sleep -Milliseconds 600

# -- Etape 2 : Serveur WebSocket --
Write-Host "  [2/4] Demarrage WebSocket (node server.js)..." -ForegroundColor Yellow

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$Root'; `$host.UI.RawUI.WindowTitle = 'WebSocket server.js'; Write-Host 'WebSocket Server' -ForegroundColor Cyan; node server.js"
)
Start-Sleep -Seconds 2

# -- Etape 3 : Serveur HTTP --
Write-Host "  [3/4] Demarrage serveur HTTP (port 8000)..." -ForegroundColor Yellow

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$Root'; `$host.UI.RawUI.WindowTitle = 'HTTP Serveur jeu'; Write-Host 'Serveur HTTP :8000' -ForegroundColor Green; python -m http.server 8000"
)
Start-Sleep -Seconds 1

# -- Etape 4 : Navigateur --
Write-Host "  [4/4] Ouverture du navigateur..." -ForegroundColor Yellow

Start-Process "http://localhost:8000"
Start-Sleep -Milliseconds 800

# -- Etape 5 : Webcam CNN --
Write-Host ""
Write-Host "  +==========================================+" -ForegroundColor Green
Write-Host "  |   Jeu ouvert ! Lancement de la webcam   |" -ForegroundColor Green
Write-Host "  +==========================================+" -ForegroundColor Green
Write-Host ""
Write-Host "  -> Montre ta main face a la camera pour jouer" -ForegroundColor White
Write-Host "  -> Appuie sur Q dans la fenetre webcam pour quitter" -ForegroundColor DarkGray
Write-Host ""

Set-Location "$Root\cnn_training"
python 5_jouer_avec_cnn.py

# -- Nettoyage --
Write-Host ""
Write-Host "  Arret des serveurs..." -ForegroundColor Yellow
Set-Location $Root

foreach ($port in @(8765, 8000)) {
    $pids = netstat -ano 2>$null |
            Select-String ":$port\s" |
            ForEach-Object { ($_ -split '\s+')[-1] } |
            Where-Object { $_ -match '^\d+$' } |
            Select-Object -Unique
    foreach ($p in $pids) {
        Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "  Termine. Bonne partie !" -ForegroundColor Cyan
Write-Host ""
