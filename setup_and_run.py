#!/usr/bin/env python3
"""
setup_and_run.py
One-command bootstrap:  python setup_and_run.py
Steps:
  1. Install dependencies
  2. Generate sample dataset
  3. Train the ML model
  4. Start the Flask server
"""

import subprocess, sys, os

BASE = os.path.dirname(os.path.abspath(__file__))


def run(cmd, **kwargs):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"  Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result


def main():
    print("=" * 55)
    print("  YouTube Trending Analyser — Setup & Run")
    print("=" * 55)

    os.chdir(BASE)

    # 1. Install deps
    print("\n  Step 1/3: Installing dependencies…")
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"])
    print("  Dependencies installed.")

    # 2. Generate data
    print("\n  Step 2/3: Generating sample dataset…")
    run([sys.executable, "generate_data.py"])

    # 3. Train model
    print("\n  Step 3/3: Training ML model…")
    run([sys.executable, "ml/train_model.py"])

    # 4. Start server
    print("\n  Starting Flask server at http://127.0.0.1:5000")
    print("     Open http://127.0.0.1:5000 in your browser")
    print("     Press Ctrl+C to stop\n")
    os.execv(sys.executable, [sys.executable, "backend/app.py"])


if __name__ == "__main__":
    main()
