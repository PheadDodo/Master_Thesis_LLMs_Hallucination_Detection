import subprocess
import sys
import time


PIPELINE_STAGES = [
    "cross_prompting.py",
    "cross_prompting_fixed.py",
    "compute_se.py",
    "verification_labeling.py",
]


def run_script(
    script_name,
    runner=subprocess.run,
    executable=sys.executable,
    timer=time.time,
):
    start_time = timer()
    print(f"Running {script_name} ...")
    try:
        runner([executable, script_name], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"{script_name} failed with code {e.returncode}") from e
    
    elapsed = timer() - start_time
    print(f"Finished {script_name} in {elapsed:.2f} seconds")


def run_pipeline(run_fn=run_script):
    for script_name in PIPELINE_STAGES:
        run_fn(script_name)


def main():
    run_pipeline()
    
if __name__ == "__main__":
    main()
