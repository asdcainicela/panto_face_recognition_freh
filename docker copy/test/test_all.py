#!/usr/bin/env python3

def test_numpy():
    print("\nTest NumPy")
    import numpy as np
    print(f"Version: {np.__version__}")
    
    a = np.random.rand(1000, 1000)
    b = np.random.rand(1000, 1000)
    c = np.dot(a, b)
    print(f"Matrix mult 1000x1000: OK")
    print(f"Result shape: {c.shape}")
    print("NumPy: PASS")

def test_onnxruntime():
    print("\nTest ONNX Runtime")
    import onnxruntime as ort
    print(f"Version: {ort.__version__}")
    
    providers = ort.get_available_providers()
    print(f"Providers: {providers}")
    print(f"Device: {ort.get_device()}")
    print("ONNX Runtime: PASS")

def test_opencv():
    print("\nTest OpenCV Python")
    try:
        import cv2
        print(f"Version: {cv2.__version__}")
        
        has_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if has_cuda:
            print(f"CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
            print("OpenCV with CUDA: PASS")
        else:
            print("OpenCV without CUDA: WARNING")
            
    except Exception as e:
        print(f"OpenCV: FAIL - {e}")

def test_libs():
    print("\nTest Libraries")
    packages = ['matplotlib', 'pandas', 'scipy', 'sklearn', 'PIL', 'seaborn', 'tqdm']
    
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'OK')
            print(f"{pkg}: {version}")
        except ImportError:
            print(f"{pkg}: FAIL")

def main():
    print("\nPython Tests Starting")
    import sys
    print(f"Python: {sys.version.split()[0]}")
    
    tests = [test_numpy, test_onnxruntime, test_opencv, test_libs]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"{test.__name__}: FAIL - {e}")
    
    print("\nPython Tests Complete")
    print("Jupyter: http://localhost:8888 (token: nvidia)")

if __name__ == "__main__":
    main()