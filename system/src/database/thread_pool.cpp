// ============= src/thread_pool.cpp =============
#include "database/thread_pool.hpp"  // ⭐ CAMBIO
#include <spdlog/spdlog.h>

ThreadPool::ThreadPool(size_t num_threads) {
    spdlog::info("🔧 Inicializando ThreadPool con {} threads", num_threads);
    
    for (size_t i = 0; i < num_threads; ++i) {
        workers.emplace_back(&ThreadPool::worker_thread, this);
    }
}

ThreadPool::~ThreadPool() {
    stop();
}

void ThreadPool::worker_thread() {
    while (true) {
        Task task;
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            
            condition.wait(lock, [this] {
                return stop_flag || !tasks.empty();
            });
            
            if (stop_flag && tasks.empty()) {
                return;
            }
            
            task = tasks.top();
            tasks.pop();
        }
        
        active_count++;
        
        try {
            task.func();
        } catch (const std::exception& e) {
            spdlog::error("Task exception: {}", e.what());
        }
        
        active_count--;
    }
}

void ThreadPool::submit_high_priority(std::function<void()> task) {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if (stop_flag) {
            throw std::runtime_error("ThreadPool is stopped");
        }
        tasks.push({task, 10});
    }
    condition.notify_one();
}

void ThreadPool::submit_low_priority(std::function<void()> task) {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if (stop_flag) {
            throw std::runtime_error("ThreadPool is stopped");
        }
        tasks.push({task, -10});
    }
    condition.notify_one();
}

size_t ThreadPool::pending_tasks() const {
    std::lock_guard<std::mutex> lock(queue_mutex);
    return tasks.size();
}

void ThreadPool::wait_all() {
    while (pending_tasks() > 0 || active_count > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void ThreadPool::stop() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop_flag = true;
    }
    
    condition.notify_all();
    
    for (auto& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    
    workers.clear();
}