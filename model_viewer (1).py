import psutil
import time
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import csv
import os
from pynvml import *

class OllamaMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.start_time = datetime.now()
        self.data = {
            "t": [],
            "cpu": [],
            "gpu": [],
            "ram": [],
            "oram": [],
            "gmem": [],
            "gmem_percent": [],
            "cpu_power": [],
            "gpu_power": [],
            "total_power": []
        }
        self.proc = None
        self.gpu_available = False
        self.gpu_handle = None
        self.gpu_total_memory = 0

        # Try to initialize GPU with error handling
        try:
            nvmlInit()
            self.gpu_handle = nvmlDeviceGetHandleByIndex(0)
            mem_info = nvmlDeviceGetMemoryInfo(self.gpu_handle)
            self.gpu_total_memory = mem_info.total / (1024 * 1024)  # Convert to MB
            self.gpu_available = True
            print("✓ GPU monitoring enabled")
        except NVMLError_GpuIsLost:
            print("⚠ Warning: GPU is lost or driver crashed. Running in CPU-only mode.")
            print("  → Try restarting your computer or resetting the GPU driver")
        except NVMLError_DriverNotLoaded:
            print("⚠ Warning: NVIDIA driver not loaded. Running in CPU-only mode.")
        except Exception as e:
            print(f"⚠ Warning: Could not initialize GPU monitoring: {e}")
            print("  Running in CPU-only mode.")

    def get_ollama_proc(self):
        for p in psutil.process_iter(['pid', 'name']):
            try:
                if p.info['name'] and 'ollama' in p.info['name'].lower():
                    return p
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return None

    def get_stats(self):
        # CPU, RAM
        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory().percent

        # GPU
        gpu = 0
        if self.gpu_available:
            try:
                util = nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu = util.gpu
            except:
                gpu = 0

        # Ollama process
        if not self.proc or not self.proc.is_running():
            self.proc = self.get_ollama_proc()

        oram = 0
        if self.proc:
            try:
                oram = self.proc.memory_info().rss / (1024 * 1024)
            except:
                pass

        gmem = 0
        gmem_percent = 0
        if self.gpu_available:
            try:
                procs = nvmlDeviceGetComputeRunningProcesses(self.gpu_handle)
                for p in procs:
                    if self.proc and p.pid == self.proc.pid:
                        gmem = p.usedGpuMemory / (1024 * 1024)
                        if self.gpu_total_memory > 0:
                            gmem_percent = (gmem / self.gpu_total_memory) * 100
            except:
                pass

        # --- Power estimation (basic proxy) ---
        # NOTE: psutil doesn't give power directly.
        # We estimate CPU power as ~ (CPU% / 100) * TDP_est (default 65W)
        cpu_power = (cpu / 100.0) * 65
        gpu_power = 0.0
        if self.gpu_available:
            try:
                gpu_power = nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0  # mW → W
            except:
                gpu_power = 0.0
        total_power = cpu_power + gpu_power

        return cpu, gpu, ram, oram, gmem, gmem_percent, cpu_power, gpu_power, total_power

    def monitor(self, duration=60):
        start = time.time()
        print("\nMonitoring Ollama Compute & Energy (Ctrl+C to stop)\n")

        try:
            while (time.time() - start) < duration:
                now = time.time() - start
                cpu, gpu, ram, oram, gmem, gmem_pct, cpu_p, gpu_p, total_p = self.get_stats()

                for k, v in zip(
                    ["t", "cpu", "gpu", "ram", "oram", "gmem", "gmem_percent", "cpu_power", "gpu_power", "total_power"],
                    [now, cpu, gpu, ram, oram, gmem, gmem_pct, cpu_p, gpu_p, total_p]
                ):
                    self.data[k].append(v)

                print(
                    f"\r{now:5.1f}s | CPU {cpu:5.1f}% | GPU {gpu:3.0f}% | "
                    f"Ollama RAM {oram:6.0f} MB | GPU VRAM {gmem:6.0f} MB ({gmem_pct:4.1f}%) | "
                    f"Power {total_p:5.1f} W",
                    end=""
                )
                time.sleep(self.interval)

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            if self.gpu_available:
                try:
                    nvmlShutdown()
                except:
                    pass
            self.plot()
            self.save_csv()

    def plot(self):
        d = self.data
        fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        fig.suptitle("Ollama Compute & Energy Monitoring", fontsize=14)

        # [1] Compute Usage
        ax[0].plot(d["t"], d["cpu"], label="CPU Usage", color="blue")
        ax[0].plot(d["t"], d["gpu"], label="GPU Usage", color="green")
        ax[0].set_ylabel("Usage (%)")
        ax[0].legend()
        ax[0].grid(alpha=0.3)

        # [2] Power
        ax[1].plot(d["t"], d["cpu_power"], label="CPU Power (est.)", color="red", alpha=0.8)
        ax[1].plot(d["t"], d["gpu_power"], label="GPU Power", color="orange", alpha=0.8)
        ax[1].plot(d["t"], d["total_power"], label="Total Power", color="black", linestyle="--")
        ax[1].set_ylabel("Power (Watts)")
        ax[1].legend()
        ax[1].grid(alpha=0.3)

        # [3] Memory
        ax[2].plot(d["t"], d["ram"], label="System RAM %", color="purple")
        ax[2].plot(d["t"], d["oram"], label="Ollama Memory (MB)", color="darkgreen", linestyle="--")
        ax[2].set_xlabel("Time (seconds)")
        ax[2].set_ylabel("System RAM Usage (%)")
        ax[2].legend()
        ax[2].grid(alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        timestamp = self.start_time.strftime('%Y%m%d_%H%M%S')
        filename = f"ollama_monitor_{timestamp}.png"
        
        # Save to same directory as script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filename)
        
        plt.savefig(filepath)
        print(f"\nSaved plot → {filepath}")

    def save_csv(self):
        timestamp = self.start_time.strftime('%Y%m%d_%H%M%S')
        filename = f"ollama_monitor_{timestamp}.csv"
        
        # Save to same directory as script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filename)
        
        with open(filepath, "w", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write metadata header
            writer.writerow(["# Ollama Monitoring Session"])
            writer.writerow(["# Start Time:", self.start_time.strftime('%Y-%m-%d %H:%M:%S')])
            writer.writerow(["# Duration:", f"{self.data['t'][-1]:.1f} seconds" if self.data['t'] else "0 seconds"])
            writer.writerow(["# Interval:", f"{self.interval} seconds"])
            writer.writerow(["# GPU Available:", "Yes" if self.gpu_available else "No"])
            if self.gpu_available and self.gpu_total_memory > 0:
                writer.writerow(["# Total GPU Memory:", f"{self.gpu_total_memory:.0f} MB"])
            writer.writerow([])
            
            # Write data headers
            headers = [
                "Time (s)", "CPU (%)", "GPU (%)",
                "System RAM (%)", "Ollama RAM (MB)", 
                "Ollama GPU VRAM (MB)", "Ollama GPU VRAM (%)",
                "CPU Power (W)", "GPU Power (W)", "Total Power (W)"
            ]
            writer.writerow(headers)
            
            # Write data rows
            for i in range(len(self.data["t"])):
                writer.writerow([
                    round(self.data["t"][i], 2),
                    round(self.data["cpu"][i], 2),
                    round(self.data["gpu"][i], 2),
                    round(self.data["ram"][i], 2),
                    round(self.data["oram"][i], 2),
                    round(self.data["gmem"][i], 2),
                    round(self.data["gmem_percent"][i], 2),
                    round(self.data["cpu_power"][i], 2),
                    round(self.data["gpu_power"][i], 2),
                    round(self.data["total_power"][i], 2)
                ])
            
            # Write summary statistics
            writer.writerow([])
            writer.writerow(["# Summary Statistics"])
            if self.data["t"]:
                writer.writerow(["# Avg CPU Usage:", f"{sum(self.data['cpu'])/len(self.data['cpu']):.2f}%"])
                writer.writerow(["# Avg GPU Usage:", f"{sum(self.data['gpu'])/len(self.data['gpu']):.2f}%"])
                writer.writerow(["# Avg Total Power:", f"{sum(self.data['total_power'])/len(self.data['total_power']):.2f} W"])
                writer.writerow(["# Total Energy:", f"{sum(self.data['total_power']) * self.interval / 3600:.4f} Wh"])
                writer.writerow(["# Peak Power:", f"{max(self.data['total_power']):.2f} W"])
                writer.writerow(["# Peak GPU VRAM:", f"{max(self.data['gmem']):.2f} MB"])

        print(f"Saved CSV → {filepath}")
    

if __name__ == "__main__":
    duration = int(input("Duration (seconds) [60]: ") or 60)
    OllamaMonitor(interval=1.0).monitor(duration)