import json
import logging
import requests
from tqdm import tqdm
from pathlib import Path

URLS_FILE = "models_urls.json"
CURRENT_DIR = Path(".")

MIN_SIZES = {
    "scrfd_2.5g_bnkps.onnx": 3_000_000,
    "scrfd_10g_bnkps.onnx": 15_000_000,
    "arcface_r100.onnx": 100_000_000,
    "realesrgan_x4plus.onnx": 40_000_000,
}

# LOGGER
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("download.log", mode="a", encoding="utf-8")
    ]
)
log = logging.getLogger("model_downloader")

# DESCARGA
def download_with_progress(url, output_path):
    try:
        r = requests.get(url, stream=True, timeout=25)
    except Exception as e:
        log.error(f"Error conectando a {url}: {e}")
        return False

    if r.status_code != 200:
        log.warning(f"HTTP {r.status_code} en {url}")
        return False

    total = int(r.headers.get("content-length", 0))
    block = 1024

    log.info(f"Descargando {output_path.name} desde {url}")

    with open(output_path, "wb") as f, tqdm(
        total=total if total > 0 else None,
        unit="B",
        unit_scale=True,
        desc=output_path.name,
        ascii=True,
        ncols=80,
    ) as t:
        for data in r.iter_content(block):
            f.write(data)
            t.update(len(data))

    return True

def file_is_valid(path, min_size):
    return path.exists() and path.stat().st_size >= min_size

# MAIN
def main():
    log.info("Leyendo JSON...")

    try:
        urls = json.load(open(URLS_FILE, "r"))
    except Exception as e:
        log.error(f"No se pudo leer {URLS_FILE}: {e}")
        return

    for model_name, mirrors in urls.items():
        out_path = CURRENT_DIR / model_name
        min_size = MIN_SIZES.get(model_name, 1_000)

        log.info(f"========== {model_name} ==========")

        # Si ya está correcto → saltar
        if file_is_valid(out_path, min_size):
            size_mb = out_path.stat().st_size / (1024 * 1024)
            log.info(f"{model_name} ya existe ({size_mb:.1f} MB)")
            continue

        success = False
        for url in mirrors:
            log.info(f"Intentando mirror: {url}")

            if download_with_progress(url, out_path):
                if file_is_valid(out_path, min_size):
                    log.info(f"{model_name} listo ✔")
                    success = True
                    break
                else:
                    log.error("Archivo corrupto, borrando…")
                    out_path.unlink(missing_ok=True)

            else:
                log.warning("Fallo la descarga.")

        if not success:
            log.error(f"No se pudo descargar {model_name} ❌")

    log.info(" end")

if __name__ == "__main__":
    main()
