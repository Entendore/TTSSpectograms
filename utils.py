# utils.py
import os
import subprocess
import platform
import logging

logger = logging.getLogger("SpectrogramStudio")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def open_file_location(path: str):
    if not os.path.exists(path):
        return
    directory = os.path.dirname(path) if os.path.isfile(path) else path
    try:
        if platform.system() == "Windows":
            os.startfile(directory)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", directory])
        else:
            subprocess.Popen(["xdg-open", directory])
    except Exception:
        pass