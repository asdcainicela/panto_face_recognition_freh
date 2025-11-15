#!/usr/bin/env python3

import sys
import time
import numpy as np

# ANSI color codes
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    CYAN = '\033[36m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.CYAN}╔{'═' * 60}╗")
    print(f"║ {text:58} ║")
    print(f"╚{'═' * 60}╝{Colors.RESET}\n")

def print_section(text):
    print(f"{Colors.BOLD}{Colors.YELLOW}[{text}]{Colors.RESET}")

def print_success(text):
    print(f"{Colors.GREEN}  ✓ {text}{Colors.RESET}")

def print_error(text):
    print(f"{Colors.RED}  ✗ {text}{Colors.RESET}")

def print_info(key, value):
    print(f"  • {Colors.BOLD}{key}{Colors.RESET}: {value}")

def print_warning(text):
    print(f"{Colors.YELLOW}  ⚠ {text}{Colors.RESET}")

def test_python_info():
    print_section("1. Información de Python")
    print_info("Versión", sys.version.split()[0])
    print_info("Plataforma", sys.platform)
    print_info("Executable", sys.executable)

def test_numpy():
    print_section("2. NumPy")
    try:
        import numpy as np
        print_info("Versión", np.__version__)
        
        # Test básico
        a = np.random.rand(1000, 1000)
        b = np.random.rand(1000, 1000)
        
        start = time.time()
        c = np.dot(a, b)
        elapsed = (time.time() - start) * 1000
        
        print_info("Matrix mult 1000x1000", f"{elapsed:.2f} ms")
        print_info("Result shape", str(c.shape))
        print_success("NumPy funcionando correctamente")
        
    except Exception as e:
        print_error(f"NumPy FAIL: {e}")
        return False
    return True

def test_opencv():
    print_section("3. OpenCV Python")
    try:
        import cv2
        print_info("Versión OpenCV", cv2.__version__)
        
        # Build info
        build_info = cv2.getBuildInformation()
        
        # Check CUDA
        cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
        print_info("Dispositivos CUDA", str(cuda_devices))
        
        if cuda_devices > 0:
            print_success("OpenCV con soporte CUDA")
            
            # Get device info
            dev_info = cv2.cuda.DeviceInfo()
            print_info("GPU Name", dev_info.name())
            print_info("Compute Capability", f"{dev_info.majorVersion()}.{dev_info.minorVersion()}")
            print_info("Total Memory", f"{dev_info.totalMemory() / (1024**3):.2f} GB")
            
            # Test CUDA operations
            print("\n  Test CUDA Operations:")
            
            # Create test image
            img_cpu = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
            
            # Upload to GPU
            img_gpu = cv2.cuda_GpuMat()
            img_gpu.upload(img_cpu)
            print_success("Upload a GPU exitoso")
            
            # Convert to grayscale on GPU
            gray_gpu = cv2.cuda.cvtColor(img_gpu, cv2.COLOR_BGR2GRAY)
            print_success("Conversión BGR→GRAY en GPU")
            
            # Download from GPU
            gray_cpu = gray_gpu.download()
            print_success("Download de GPU exitoso")
            
            # Benchmark CPU vs GPU
            print("\n  Benchmark CPU vs GPU:")
            
            # CPU timing
            start = time.time()
            for _ in range(100):
                _ = cv2.cvtColor(img_cpu, cv2.COLOR_BGR2GRAY)
            cpu_time = (time.time() - start) * 10  # ms per operation
            
            # GPU timing
            start = time.time()
            for _ in range(100):
                gray_gpu = cv2.cuda.cvtColor(img_gpu, cv2.COLOR_BGR2GRAY)
            gpu_time = (time.time() - start) * 10  # ms per operation
            
            print(f"    CPU (cvtColor 1920x1080): {cpu_time:.2f} ms")
            print(f"    GPU (cvtColor 1920x1080): {gpu_time:.2f} ms")
            print(f"    Speedup: {cpu_time/gpu_time:.2f}x")
            
        else:
            print_warning("OpenCV sin soporte CUDA")
        
        # Check DNN module
        print("\n  DNN Module:")
        backends = cv2.dnn.getAvailableBackends()
        print_info("Backends disponibles", str(len(backends)))
        
        has_cuda = False
        for backend, target in backends:
            if backend == cv2.dnn.DNN_BACKEND_CUDA:
                has_cuda = True
                print_success("DNN CUDA backend disponible")
                break
        
        if not has_cuda:
            print_warning("DNN CUDA backend no disponible")
        
        print_success("OpenCV Python funcionando correctamente")
        
    except Exception as e:
        print_error(f"OpenCV FAIL: {e}")
        return False
    return True

def test_scientific_libs():
    print_section("4. Librerías Científicas")
    
    libs = {
        'matplotlib': 'Matplotlib',
        'pandas': 'Pandas',
        'scipy': 'SciPy',
        'sklearn': 'Scikit-learn',
        'seaborn': 'Seaborn',
        'PIL': 'Pillow',
        'tqdm': 'TQDM'
    }
    
    success_count = 0
    for module_name, display_name in libs.items():
        try:
            mod = __import__(module_name)
            version = getattr(mod, '__version__', 'OK')
            print_info(display_name, version)
            success_count += 1
        except ImportError:
            print_error(f"{display_name}: No instalado")
    
    if success_count == len(libs):
        print_success("Todas las librerías científicas disponibles")
    else:
        print_warning(f"{success_count}/{len(libs)} librerías disponibles")
    
    return success_count > 0

def test_jupyter():
    print_section("5. JupyterLab")
    try:
        import jupyterlab
        print_info("Versión", jupyterlab.__version__)
        print_info("URL", "http://localhost:8888")
        print_info("Token", "nvidia")
        print_success("JupyterLab instalado")
    except ImportError:
        print_error("JupyterLab no instalado")
        return False
    return True

def test_performance():
    print_section("6. Tests de Rendimiento")
    
    try:
        import numpy as np
        
        # Matrix operations
        sizes = [100, 500, 1000, 2000]
        print("\n  Matrix Multiplication (NumPy):")
        
        for size in sizes:
            a = np.random.rand(size, size)
            b = np.random.rand(size, size)
            
            start = time.time()
            c = np.dot(a, b)
            elapsed = (time.time() - start) * 1000
            
            print(f"    {size}x{size}: {elapsed:.2f} ms")
        
        print_success("Tests de rendimiento completados")
        
    except Exception as e:
        print_error(f"Performance tests FAIL: {e}")
        return False
    return True

def main():
    print_header("VERIFICACIÓN PYTHON COMPLETA")
    
    tests = [
        ("Python Info", test_python_info),
        ("NumPy", test_numpy),
        ("OpenCV", test_opencv),
        ("Librerías Científicas", test_scientific_libs),
        ("JupyterLab", test_jupyter),
        ("Rendimiento", test_performance)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result is not None:
                results.append(result)
            print()
        except Exception as e:
            print_error(f"{test_name} FAIL: {e}")
            results.append(False)
            print()
    
    # Final summary
    print_header("RESUMEN FINAL")
    
    passed = sum(results)
    total = len([r for r in results if r is not None])
    
    print(f"{Colors.BOLD}  Tests ejecutados: {total}{Colors.RESET}")
    print(f"{Colors.BOLD}  Tests exitosos: {Colors.GREEN}{passed}{Colors.RESET}")
    print(f"{Colors.BOLD}  Tests fallidos: {Colors.RED}{total - passed}{Colors.RESET}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}  ✓ TODOS LOS TESTS PASARON{Colors.RESET}\n")
        return 0
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}  ⚠ ALGUNOS TESTS FALLARON{Colors.RESET}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())