"""
=============================================================================
SADM-SEC  |  main.py
=============================================================================
Master Entry Point — runs all phases in sequence and generates all outputs.

Usage:
    python3 main.py [--phase {all|math|noise|sim|track|viz}]

    --phase all    (default) Run all phases
    --phase math   Phase 1: self-test spatial_logic.py
    --phase noise  Phase 2: Module 6 noise analysis (BECE304L alignment)
    --phase sim    Phase 3: static simulation (Bob 30°, Eve -45°)
    --phase track  Phase 4: moving target tracking demo
    --phase viz    Phase 5: render all plots to PNG

=============================================================================
"""

import sys
import argparse
import time

BANNER = """
=================================================================
  SADM-SEC  |  Spatial Aware Directional Modulation - Secure
  Physical Layer Security System  |  8-Antenna ULA
=================================================================
"""


def run_phase_math():
    print("\n" + ">>>" + " PHASE 1: Mathematical Engine Self-Test")
    import runpy
    runpy.run_path("spatial_logic.py", run_name="__main__")


def run_phase_noise():
    print("\n" + ">>>" + " PHASE 2: Module 6 Noise Analysis  (BECE304L)")
    from noise_analysis import print_noise_budget
    print_noise_budget(bob_angle_deg=30.0, eve_angle_deg=-45.0,
                       snr_signal_db=20.0, snr_noise_db=10.0)


def run_phase_sim(use_ml=True):
    print("\n" + ">>>" + f" PHASE 3: Static Simulation {'(ML Mode)' if use_ml else ''}")
    from virtual_channel import run_simulation
    run_simulation(n_blocks=8, bob_angle=30.0, moving=False, use_ml=use_ml)

def run_phase_track(use_ml=True):
    print("\n" + ">>>" + f" PHASE 4: Moving Target {'(ML Mode)' if use_ml else ''}")
    from virtual_channel import run_simulation
    run_simulation(n_blocks=20, moving=True, use_ml=use_ml)

def run_phase_viz(use_ml=True):
    print("\n" + ">>>" + " PHASE 5: Visualisation Render")
    from visualization import render_all
    path = render_all(use_ml=use_ml)
    return path

def run_all(use_ml=True):  # Added use_ml here
    run_phase_math()
    time.sleep(0.3)
    run_phase_noise()
    time.sleep(0.3)
    # Pass the ML flag down to the simulation phases
    run_phase_sim(use_ml=use_ml)
    time.sleep(0.3)
    run_phase_track(use_ml=use_ml)
    time.sleep(0.3)
    path = run_phase_viz(use_ml=use_ml)
    print(f"\n[OK]  All phases complete.")
    print(f"    Dashboard saved to: {path}")

def main():
    print(BANNER)

    parser = argparse.ArgumentParser(
        description="SADM-SEC - Physical Layer Security System")
    parser.add_argument(
        "--phase",
        choices=["all", "math", "noise", "sim", "track", "viz"],
        default="all",
        help="Which phase to run (default: all)"
    )
    # This now defaults to TRUE. You use --no-ml to switch to math.
    parser.add_argument(
        "--no-ml",
        action="store_false",
        dest="use_ml",
        help="Disable Machine Learning and use Root-MUSIC instead"
    )
    parser.set_defaults(use_ml=True) 
    args = parser.parse_args()

    if args.phase == "all":
        run_all(use_ml=args.use_ml)
    elif args.phase == "math":
        run_phase_math()
    elif args.phase == "noise":
        run_phase_noise()
    elif args.phase == "sim":
        run_phase_sim(use_ml=args.use_ml)
    elif args.phase == "track":
        run_phase_track(use_ml=args.use_ml)
    elif args.phase == "viz":
        run_phase_viz(use_ml=args.use_ml)


if __name__ == "__main__":
    main()
