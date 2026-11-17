"""Point d'entrée : `python run.py` puis ouvrez http://127.0.0.1:8765"""
import argparse
import logging
import threading
import webbrowser

import uvicorn

from coach.config import load_settings


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

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s : %(message)s")
    settings = load_settings()
    port = args.port or settings.port

    from coach.server import create_app

    app = create_app(settings)
    url = f"http://{settings.host}:{port}"
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"\n  Interview Coach démarré : {url}\n")
    uvicorn.run(app, host=settings.host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
