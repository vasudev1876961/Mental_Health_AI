"""
Master One-Click Launcher for Privacy-Preserving Multimodal Mental Health AI Platform (Zenith Phase 10).

Usage:
    python launch_all.py
"""

import sys
import subprocess


def show_menu():
    print("\n==================================================================")
    print("  PRIVACY-PRESERVING MULTIMODAL MENTAL HEALTH AI PLATFORM (PHASE 10)")
    print("==================================================================")
    print("  [1] Launch Interactive Streamlit Web Dashboard (8 Tabs)")
    print("  [2] Launch FastAPI REST & WebSocket Edge API Server (v2.4.0)")
    print("  [3] Run Quick Sample Terminal Predictor Demo")
    print("  [4] Run Automated Unit Test Suite (~100 Tests across 21 Modules)")
    print("  [5] Run Complete Automated Research Experiments (E1 to E35)")
    print("  [6] Generate 300 DPI Paper Figures (Fig 1-22) & Academic Drafts (MD + LaTeX)")
    print("  [7] Exit")
    print("==================================================================\n")


def main():
    while True:
        show_menu()
        choice = input("Select an option (1-7): ").strip()

        if choice == "1":
            print("\nLaunching Streamlit Dashboard at http://localhost:8501 ...")
            subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"])
        elif choice == "2":
            print("\nLaunching FastAPI API Server at http://127.0.0.1:8000 ...")
            subprocess.run([sys.executable, "-m", "uvicorn", "src.api.server:app", "--reload", "--port", "8000"])
        elif choice == "3":
            print("\nRunning Quick Sample Terminal Demo ...")
            subprocess.run([sys.executable, "demo_predict_sample.py"])
        elif choice == "4":
            print("\nRunning Automated Unit Test Suite ...")
            subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
        elif choice == "5":
            print("\nRunning Research Experiments E1 to E35 ...")
            subprocess.run([sys.executable, "experiments/run_experiments.py", "--mode", "fast"])
        elif choice == "6":
            print("\nGenerating Figures & Manuscripts ...")
            subprocess.run([sys.executable, "experiments/generate_paper_plots.py"])
            subprocess.run([sys.executable, "experiments/export_paper_report.py"])
            subprocess.run([sys.executable, "experiments/build_latex_paper.py"])
        elif choice == "7":
            print("\nExiting. Thank you!")
            break
        else:
            print("\nInvalid choice. Please select a number between 1 and 7.")


if __name__ == "__main__":
    main()
