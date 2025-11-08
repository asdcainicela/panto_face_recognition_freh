#include <iostream>

// Check C++ version
int main() {
    std::cout << "=== C++ Compiler Version Check ===" << std::endl;
    
    #if __cplusplus >= 202002L
        std::cout << " C++20 detected (__cplusplus = " << __cplusplus << ")" << std::endl;
    #elif __cplusplus >= 201703L
        std::cout << " C++17 detected (__cplusplus = " << __cplusplus << ")" << std::endl;
    #elif __cplusplus >= 201402L
        std::cout << " C++14 detected (__cplusplus = " << __cplusplus << ")" << std::endl;
    #else
        std::cout << " C++11 or older (__cplusplus = " << __cplusplus << ")" << std::endl;
    #endif
    
    std::cout << "\n=== Compiler Info ===" << std::endl;
    #ifdef __GNUC__
        std::cout << "Compiler: GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__ << std::endl;
    #endif
    
    #ifdef __clang__
        std::cout << "Compiler: Clang " << __clang_major__ << "." << __clang_minor__ << std::endl;
    #endif
    
    std::cout << "\n=== C++20 Features Test ===" << std::endl;
    
    // Test concepts (C++20)
    #ifdef __cpp_concepts
        std::cout << " Concepts supported" << std::endl;
    #else
        std::cout << " Concepts NOT supported" << std::endl;
    #endif
    
    // Test ranges (C++20)
    #ifdef __cpp_lib_ranges
        std::cout << " Ranges supported" << std::endl;
    #else
        std::cout << " Ranges NOT supported" << std::endl;
    #endif
    
    // Test coroutines (C++20)
    #ifdef __cpp_impl_coroutine
        std::cout << " Coroutines supported" << std::endl;
    #else
        std::cout << " Coroutines NOT supported" << std::endl;
    #endif
    
    // Test jthread (C++20)
    #ifdef __cpp_lib_jthread
        std::cout << " std::jthread supported" << std::endl;
    #else
        std::cout << " std::jthread NOT supported" << std::endl;
    #endif
    
    // Test barrier (C++20)
    #ifdef __cpp_lib_barrier
        std::cout << " std::barrier supported" << std::endl;
    #else
        std::cout << " std::barrier NOT supported" << std::endl;
    #endif
    
    // Test semaphore (C++20)
    #ifdef __cpp_lib_semaphore
        std::cout << " std::semaphore supported" << std::endl;
    #else
        std::cout << " std::semaphore NOT supported" << std::endl;
    #endif
    
    // Test format (C++20)
    #ifdef __cpp_lib_format
        std::cout << " std::format supported" << std::endl;
    #else
        std::cout << " std::format NOT supported (use fmt library)" << std::endl;
    #endif
    
    return 0;
}