import json
import os
import requests
from pathlib import Path

# ================================
# Config
# ================================

URLS_FILE = "models_urls.json"
OUT_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)

# Tamaños mínimos para validar integridad
MIN_SIZES = {
    "scrfd_2.5g_bnkps.onnx": 3_000_000,
    "scrfd_10g_bnkps.onnx": 15_000_000,
    "arcface_r100.onnx": 100_000_000,
    "realesrgan_x4plus.onnx": 40_000_000,
}

# ================================
# Funciones
# ================================

def download(url, output_path):
    """Descarga un archivo con stream."""
    try:
        r = requests.get(url, stream=True, timeout=25)
    except:
        return False

    if r.status_code != 200:
        return False

    with open(output_path, "wb") as f:
        for chunk in r.iter_content(1024):
            if chunk:
                f.write(chunk)
    return True


def file_is_valid(path, min_size):
    if not path.exists():
        return False
    return path.stat().st_size >= min_size


# ================================
# Main
# ================================

def main():
    urls = json.load(open(URLS_FILE))

    for model_name, mirrors in urls.items():
        out_path = OUT_DIR / model_name
        min_size = MIN_SIZES.get(model_name, 1)

        print(f"\n[INFO] Procesando: {model_name}")

        # Si ya existe y es válido → OK
        if file_is_valid(out_path, min_size):
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"[OK] Ya existe ({size_mb:.1f} MB)")
            continue

        # Intentar mirrors
        for url in mirrors:
            print(f"[INFO] Intentando: {url}")

            if download(url, out_path):
                if file_is_valid(out_path, min_size):
                    size_mb = out_path.stat().st_size / (1024 * 1024)
                    print(f"[OK] Descargado ({size_mb:.1f} MB)")
                    break
                else:
                    print("[ERR] Archivo incompleto. Borrando y reintentando...")
                    out_path.unlink(missing_ok=True)
            else:
                print("[WARN] Falló este mirror")

        if not file_is_valid(out_path, min_size):
            print(f"[ERR] No se pudo descargar {model_name} desde ningún mirror.")
        else:
            print(f"[SUCCESS] {model_name} listo.")

    print("\n[FIN] Todos los modelos procesados.")


if __name__ == "__main__":
    main()
