"""Point d'entrée : `python run.py` puis ouvrez http://127.0.0.1:8765"""
import argparse
import logging
import shutil
import subprocess
import sys
import threading
import webbrowser

import uvicorn

from coach.config import load_settings


def open_window(url: str) -> None:
    """Ouvre l'interface comme une application (fenêtre Edge/Chrome sans barre d'adresse)."""
    if sys.platform == "win32":
        for exe in ("msedge", "chrome"):
            try:
                subprocess.Popen(f'start "" {exe} --app={url} --window-size=760,940', shell=True)
                return
            except Exception:
                continue
    for exe in ("google-chrome", "chromium", "microsoft-edge"):
        if shutil.which(exe):
            subprocess.Popen([exe, f"--app={url}", "--window-size=760,940"])
            return
    webbrowser.open(url)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interview Coach — copilote d'entretien technique")
    parser.add_argument("--devices", action="store_true", help="liste les micros et sorties audio puis quitte")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if args.devices:
        from coach.audio import list_devices

        print(list_devices())
        return

    if sys.platform == "win32":
        try:  # initialise l'audio Windows (COM) sur le thread principal, avant tout autre thread
            import soundcard  # noqa: F401
        except Exception as exc:
            print(f"Avertissement audio : {exc}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s : %(message)s")
    settings = load_settings()
    port = args.port or settings.port

    from coach.server import create_app

    app = create_app(settings)
    url = f"http://{settings.host}:{port}"
    if not args.no_browser:
        threading.Timer(1.5, open_window, args=(url,)).start()
    print(f"\n  Interview Coach démarré : {url}\n")
    uvicorn.run(app, host=settings.host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
