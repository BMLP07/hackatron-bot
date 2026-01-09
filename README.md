<div align = "Center">

# HackaTron AI Bot

</div>
Installation and usage guide for the HackaTron bot. The bot reads game-state JSON on stdin and prints a move per tick (1=LEFT, 2=UP, 3=RIGHT, 4=DOWN).

---

## Prerequisites
- Python 3.11+
- Docker (to run inside the HackaTron server)
- [HackaTron server](https://github.com/jnegrete2005/hackatron) repository

For a fresh, unmodified reference bot, see the official [HackaTron client](https://github.com/jnegrete2005/hackatron-client)

---

## Local setup
```powershell
cd ../main-bot
pip install -r requirements.txt
```

Quick sanity check without Docker (prints one move for a sample state):
```powershell
Get-Content sample_state.json | python -m src.hackatron-bot
```
Use the sample from the HackaTron server docs or craft your own JSON matching the game schema.

## Build Docker image
```powershell
cd ../main-bot
docker build -t <DOCKER_USERNAME>/<DOCKER_IMAGE_NAME> .
```

---

## Run in the HackaTron server
From the server repository (`../hackatron`):
```powershell
cd ../hackatron
set PYTHONPATH=%cd%\src;%PYTHONPATH%
python src/main.py --bot1 <DOCKER_USERNAME>/<DOCKER_IMAGE_NAME> (--manual1) --bot2 <DOCKER_USERNAME>/<DOCKER_IMAGE_NAME> (--manual2) (--auto)
```
- `--bot1`/`--bot2` expect Docker images (local or pullable).
- Add `--manual1` or `--manual2` to control a player via keyboard.
- Use `--auto` to advance ticks automatically.

### You vs this bot
```powershell
python src/main.py --bot1 <DOCKER_USERNAME>/<DOCKER_IMAGE_NAME> --manual1 --bot2 <DOCKER_USERNAME>/<DOCKER_IMAGE_NAME> --auto
```

### Bot vs another bot image
```powershell
python src/main.py --bot1 <DOCKER_USERNAME>/<DOCKER_IMAGE_NAME> --bot2 <DOCKER_USERNAME>/<DOCKER_IMAGE_NAME> --auto
```

---

Developed with ❤️ by Binh Minh L.P. & Héctor M.C.



