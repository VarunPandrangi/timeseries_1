import subprocess
import sys
import os
import time
from datetime import timedelta

# Define the scripts in execution order
scripts = [
    "src/load_opsd.py",
    "src/decompose_acf_pacf.py",
    "src/forecast.py",
    "src/anomaly.py",
    "src/anomaly_ml.py",
    "src/live_loop.py"
]

def run_script(script_path):
    print(f"\n{'='*60}")
    print(f"Running {script_path}...")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        # Run the script and wait for it to finish
        # We use sys.executable to ensure we use the same python interpreter
        result = subprocess.run(
            [sys.executable, script_path], 
            check=True, 
            capture_output=False  # Let output stream to console
        )
        duration = time.time() - start_time
        print(f"\n>>> {script_path} completed in {str(timedelta(seconds=int(duration)))}")
        return duration
    except subprocess.CalledProcessError as e:
        print(f"\n!!! Error running {script_path} !!!")
        # print(e) # subprocess.run already prints stderr if capture_output=False
        sys.exit(1)

if __name__ == "__main__":
    print("Starting OPSD PowerDesk Pipeline...")
    total_start = time.time()
    
    # Ensure we are in the project root
    if not os.path.exists("config.yaml"):
        print("Error: config.yaml not found. Please run from project root.")
        sys.exit(1)

    timings = {}
    for script in scripts:
        if os.path.exists(script):
            duration = run_script(script)
            timings[script] = duration
        else:
            print(f"Warning: Script {script} not found. Skipping.")

    total_duration = time.time() - total_start
    
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    for script, duration in timings.items():
        print(f"{script:<30} : {str(timedelta(seconds=int(duration)))}")
    print(f"{'-'*60}")
    print(f"{'Total Time':<30} : {str(timedelta(seconds=int(total_duration)))}")
    print(f"{'='*60}")
