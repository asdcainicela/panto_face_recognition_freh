#!/usr/bin/env python3

import sys
import time
import numpy as np


def header(text):
    print("\n=== " + text + " ===\n")


def section(text):
    print("[" + text + "]")


def info(key, value):
    print("  " + key + ": " + value)


def success(text):
    print("  OK " + text)


def error(text):
    print("  ERROR " + text)


def warning(text):
    print("  WARNING " + text)


def test_python_info():
    section("1. Python")
    info("Version", sys.version.split()[0])
    info("Platform", sys.platform)
    info("Executable", sys.executable)


def test_numpy():
    section("2. NumPy")
    try:
        import numpy as np
        info("Version", np.__version__)

        a = np.random.rand(1000, 1000)
        b = np.random.rand(1000, 1000)

        start = time.time()
        _ = np.dot(a, b)
        elapsed = (time.time() - start) * 1000

        info("Matrix mult 1000x1000", f"{elapsed:.2f} ms")
        success("NumPy OK")
        return True

    except Exception as e:
        error("NumPy fail: " + str(e))
        return False


def test_opencv():
    section("3. OpenCV Python")
    try:
        import cv2
        info("OpenCV version", cv2.__version__)

        cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
        info("CUDA devices", str(cuda_devices))

        if cuda_devices > 0:
            success("OpenCV CUDA enabled")

            dev = cv2.cuda.DeviceInfo()
            info("GPU name", dev.name())
            info("Compute capability", f"{dev.majorVersion()}.{dev.minorVersion()}")
            info("Total memory GB", f"{dev.totalMemory() / 1e9:.2f}")

            img_cpu = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)

            gpu = cv2.cuda_GpuMat()
            gpu.upload(img_cpu)
            success("Upload GPU")

            gray_gpu = cv2.cuda.cvtColor(gpu, cv2.COLOR_BGR2GRAY)
            success("cvtColor GPU")

            _ = gray_gpu.download()
            success("Download GPU")

            # Benchmark
            start = time.time()
            for _ in range(100):
                cv2.cvtColor(img_cpu, cv2.COLOR_BGR2GRAY)
            cpu_t = (time.time() - start) * 10

            start = time.time()
            for _ in range(100):
                cv2.cuda.cvtColor(gpu, cv2.COLOR_BGR2GRAY)
            gpu_t = (time.time() - start) * 10

            info("CPU cvtColor 1080p ms", f"{cpu_t:.2f}")
            info("GPU cvtColor 1080p ms", f"{gpu_t:.2f}")
            info("Speedup", f"{cpu_t / gpu_t:.2f}")

        else:
            warning("OpenCV built without CUDA support")

        # DNN
        backends = cv2.dnn.getAvailableBackends()
        info("DNN backends", str(len(backends)))

        cuda_dnn = False
        for backend, target in backends:
            if backend == cv2.dnn.DNN_BACKEND_CUDA:
                cuda_dnn = True
                success("DNN CUDA backend")
                break

        if not cuda_dnn:
            warning("DNN CUDA backend not found")

        success("OpenCV OK")
        return True

    except Exception as e:
        error("OpenCV fail: " + str(e))
        return False


def test_scientific_libs():
    section("4. Scientific Libraries")

    libs = ["matplotlib", "pandas", "scipy", "sklearn", "seaborn", "PIL", "tqdm"]
    count = 0

    for m in libs:
        try:
            mod = __import__(m)
            version = getattr(mod, "__version__", "OK")
            info(m, str(version))
            count += 1
        except ImportError:
            error(m + " not installed")

    if count == len(libs):
        success("All libs OK")
    else:
        warning(f"{count}/{len(libs)} imported")

    return count > 0


def test_jupyter():
    section("5. JupyterLab")
    try:
        import jupyterlab
        info("Version", jupyterlab.__version__)
        info("URL", "http://localhost:8888")
        info("Token", "nvidia")
        success("JupyterLab OK")
        return True
    except ImportError:
        error("JupyterLab not installed")
        return False


def test_performance():
    section("6. Performance tests")

    try:
        import numpy as np
        sizes = [100, 500, 1000, 2000]

        info("Matrix multiply", "NumPy benchmark")

        for s in sizes:
            a = np.random.rand(s, s)
            b = np.random.rand(s, s)
            start = time.time()
            _ = np.dot(a, b)
            ms = (time.time() - start) * 1000
            info(f"{s}x{s}", f"{ms:.2f} ms")

        success("Performance OK")
        return True

    except Exception as e:
        error("Performance fail: " + str(e))
        return False


def main():
    header("PYTHON FULL VERIFICATION")

    tests = [
        ("Python Info", test_python_info),
        ("NumPy", test_numpy),
        ("OpenCV", test_opencv),
        ("Sci Libs", test_scientific_libs),
        ("JupyterLab", test_jupyter),
        ("Performance", test_performance)
    ]

    results = []

    for name, func in tests:
        try:
            res = func()
            if res is not None:
                results.append(res)
            print()
        except Exception as e:
            error(name + " fail: " + str(e))
            results.append(False)
            print()

    header("SUMMARY")

    total = len(results)
    passed = sum(results)

    info("Tests executed", str(total))
    info("Tests passed", str(passed))
    info("Tests failed", str(total - passed))

    if passed == total:
        success("All tests passed")
        return 0
    else:
        warning("Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
