#!/usr/bin/env python3


import psutil
import threading
import time
from collections import deque
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ProcessStats:
    """Container for process statistics"""
    cpu_1min: float = 0.0
    cpu_5min: float = 0.0
    cpu_10min: float = 0.0
    cpu_15min: float = 0.0
    ram_1min: float = 0.0
    ram_5min: float = 0.0
    ram_10min: float = 0.0
    ram_15min: float = 0.0


class ProcessMonitor:

    
    def __init__(self, pid: int, interval: int = 5):
        """
        Initialize process monitor.
        
        Args:
            pid: Process ID to monitor
            interval: Sampling interval in seconds (default: 5)
        """
        self.pid = pid
        self.interval = interval
        self.p = psutil.Process(pid)
        
        # Calculate sample counts for different time periods
        self.samples_1 = 1 * 60 // interval
        self.samples_5 = 5 * 60 // interval
        self.samples_10 = 10 * 60 // interval
        self.samples_15 = 15 * 60 // interval
        
        # Initialize history queues
        self.cpu_history = deque(maxlen=self.samples_15)
        self.ram_history = deque(maxlen=self.samples_15)
        
        # Initialize statistics
        self.stats = ProcessStats()
        
        # Threading
        self._stop_flag = False
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def avg_usage(self, data: deque, n: int) -> float:
        """Calculate average usage from the last n samples."""
        if len(data) < n:
            return sum(data) / len(data) if data else 0.0
        else:
            return sum(list(data)[-n:]) / n
    
    def _monitor_loop(self):
        """Main monitoring loop running in background thread."""
        while not self._stop_flag:
            try:
                # Get current usage
                cpu_percent = self.p.cpu_percent(interval=None) / psutil.cpu_count()
                ram_percent = self.p.memory_percent()
                
                # Add to history
                self.cpu_history.append(cpu_percent)
                self.ram_history.append(ram_percent)
                
                # Update rolling averages
                self.stats.cpu_1min = self.avg_usage(self.cpu_history, self.samples_1)
                self.stats.cpu_5min = self.avg_usage(self.cpu_history, self.samples_5)
                self.stats.cpu_10min = self.avg_usage(self.cpu_history, self.samples_10)
                self.stats.cpu_15min = self.avg_usage(self.cpu_history, self.samples_15)
                
                self.stats.ram_1min = self.avg_usage(self.ram_history, self.samples_1)
                self.stats.ram_5min = self.avg_usage(self.ram_history, self.samples_5)
                self.stats.ram_10min = self.avg_usage(self.ram_history, self.samples_10)
                self.stats.ram_15min = self.avg_usage(self.ram_history, self.samples_15)
                
                time.sleep(self.interval)
                
            except psutil.NoSuchProcess:
                # Process has terminated
                break
            except Exception as e:
                # Log error and continue
                print(f"Process monitor error: {e}")
                break
    
    def get_stats(self) -> ProcessStats:
        """Get current process statistics."""
        return self.stats
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current CPU and RAM usage."""
        try:
            return {
                'cpu_percent': self.p.cpu_percent(interval=None) / psutil.cpu_count(),
                'ram_percent': self.p.memory_percent(),
                'ram_mb': self.p.memory_info().rss / 1024 / 1024
            }
        except psutil.NoSuchProcess:
            return {'cpu_percent': 0.0, 'ram_percent': 0.0, 'ram_mb': 0.0}
    
    def get_process_info(self) -> Dict[str, any]:
        """Get detailed process information."""
        try:
            return {
                'pid': self.pid,
                'name': self.p.name(),
                'status': self.p.status(),
                'create_time': self.p.create_time(),
                'num_threads': self.p.num_threads(),
                'memory_info': self.p.memory_info()._asdict(),
                'cpu_times': self.p.cpu_times()._asdict()
            }
        except psutil.NoSuchProcess:
            return {'pid': self.pid, 'name': 'Unknown', 'status': 'terminated'}
    
    def is_alive(self) -> bool:
        """Check if the monitored process is still alive."""
        try:
            return self.p.is_running()
        except psutil.NoSuchProcess:
            return False
    
    def stop(self):
        """Stop monitoring and cleanup resources."""
        self._stop_flag = True
        if self.thread.is_alive():
            self.thread.join(timeout=2)
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop()


class SystemMonitor:
    """
    System-wide monitoring utility.
    """
    
    @staticmethod
    def get_system_stats() -> Dict[str, float]:
        """Get system-wide statistics."""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'ram_percent': psutil.virtual_memory().percent,
            'ram_available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
            'ram_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'disk_percent': psutil.disk_usage('/').percent,
            'disk_free_gb': psutil.disk_usage('/').free / 1024 / 1024 / 1024
        }
    
    @staticmethod
    def get_top_processes(n: int = 10) -> List[Dict[str, any]]:
        """Get top processes by CPU usage."""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Sort by CPU usage and return top n
        processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
        return processes[:n]


# Example usage
if __name__ == "__main__":
    import os
    
    # Monitor current process
    monitor = ProcessMonitor(os.getpid())
    
    try:
        print("Monitoring current process... Press Ctrl+C to stop")
        while True:
            stats = monitor.get_stats()
            current = monitor.get_current_usage()
            
            print(f"\rCPU: {current['cpu_percent']:.1f}% | "
                  f"RAM: {current['ram_percent']:.1f}% | "
                  f"CPU 5min avg: {stats.cpu_5min:.1f}%", end="")
            
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping monitor...")
        monitor.stop() 