import subprocess
import sys
import time

def run_script(script_name):
    start_time = time.time()
    print(f"Running {script_name} ...")
    try:
        subprocess.run([sys.executable, script_name], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"{script_name} failed with code {e.returncode}") from e
    
    elapsed = time.time() - start_time
    print(f"Finished {script_name} in {elapsed:.2f} seconds")

def main():
    run_script("cross_prompting.py")
    run_script("cross_prompting_fixed.py")
    run_script("compute_se.py")
    run_script("verification_labeling.py")
    
if __name__ == "__main__":
    main()