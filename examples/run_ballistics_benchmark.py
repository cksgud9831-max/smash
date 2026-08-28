"""Ballistics Solver Benchmark: Gravity-only (Analytic) vs Drag-included (RK4).

Compares computing latency (seconds) and physical accuracy (elevation/azimuth difference, 
and actual miss distance if drag is ignored) at various ranges (100m to 500m).
Use this to decide when to enable nonlinear forces on Jetson Orin Nano.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aiming_engine.config import load_config, ProjectileConfig
from aiming_engine.forces import GravityForce, DragForce
from aiming_engine.newton_solver import NewtonSolver
from aiming_engine.projectile_model import ProjectileModel
from aiming_engine.target_state import TargetState
from aiming_engine.hit_equation import HitEquation
from aiming_engine.types import Vector3


def main() -> None:
    # 1. Load configuration
    config = load_config()
    solver = NewtonSolver(config.solver)

    # Prepare real/synthetic projectile projectile parameters
    # Make sure we use a non-zero mass and diameter for drag simulation
    mass = config.projectile.mass_kg if config.projectile.mass_kg > 0.0 else 0.0095  # e.g., 9g 9mm bullet
    diameter = config.projectile.diameter_m if config.projectile.diameter_m > 0.0 else 0.009  # 9mm
    
    drag_enabled_config = ProjectileConfig(
        muzzle_velocity=config.projectile.muzzle_velocity,
        forces=["gravity", "drag"],
        gravity=config.projectile.gravity,
        integrator_step=config.projectile.integrator_step,
        drag_model="G1",
        mass_kg=mass,
        diameter_m=diameter,
        air_density_kg_m3=config.projectile.air_density_kg_m3,
        speed_of_sound_mps=config.projectile.speed_of_sound_mps,
    )

    # 2. Build Models
    # Model A: Gravity only (triggers analytic shortcut in projectile_model.py -> O(1))
    model_gravity = ProjectileModel(
        config.projectile, 
        forces=[GravityForce(config.projectile.gravity)]
    )

    # Model B: Gravity + Drag (triggers RK4 numerical integration -> O(N))
    model_drag = ProjectileModel(
        drag_enabled_config,
        forces=[
            GravityForce(drag_enabled_config.gravity),
            DragForce(
                drag_model=drag_enabled_config.drag_model,
                mass_kg=drag_enabled_config.mass_kg,
                diameter_m=drag_enabled_config.diameter_m,
                air_density_kg_m3=drag_enabled_config.air_density_kg_m3,
                speed_of_sound_mps=drag_enabled_config.speed_of_sound_mps,
            )
        ]
    )

    ranges = [100.0, 200.0, 300.0, 400.0, 500.0]
    launch_point = Vector3(0.0, 0.0, 0.0)
    ref_time = 0.0

    print("=" * 85)
    print(f"{'Target Range (m)':^18} | {'Gravity Only (Analytic)':^25} | {'With Air Drag (RK4)':^25} | {'Difference'}")
    print("-" * 85)

    num_iterations = 200  # for latency test

    for r in ranges:
        # Create a stationary target at distance 'r' along the X axis
        target_pos = Vector3(r, 0.0, 0.0)
        target_state = TargetState(config.target_state)
        target_state.update(target_pos, ref_time)

        # Build equations
        eq_gravity = HitEquation(target_state, model_gravity, launch_point, ref_time)
        eq_drag = HitEquation(target_state, model_drag, launch_point, ref_time)

        # Initial guess: x0 = [time_of_flight, azimuth, elevation]
        t_guess = r / config.projectile.muzzle_velocity
        x0 = np.array([t_guess, 0.0, 0.0])

        # --- Benchmark Model A (Gravity Only) ---
        # 1. Warm up
        res_grav = solver.solve(eq_gravity, x0)
        
        # 2. Timing
        t_start = time.perf_counter()
        for _ in range(num_iterations):
            solver.solve(eq_gravity, x0)
        t_grav_ms = ((time.perf_counter() - t_start) / num_iterations) * 1000.0

        # --- Benchmark Model B (Gravity + Drag) ---
        # 1. Warm up
        res_drag = solver.solve(eq_drag, x0)

        # 2. Timing
        t_start = time.perf_counter()
        for _ in range(num_iterations):
            solver.solve(eq_drag, x0)
        t_drag_ms = ((time.perf_counter() - t_start) / num_iterations) * 1000.0

        # Output analysis
        if res_grav.converged and res_drag.converged:
            tof_g, az_g, el_g = res_grav.x
            tof_d, az_d, el_d = res_drag.x

            el_diff_mrad = (el_d - el_g) * 1000.0  # radians to mrad

            # Calculate Miss Distance (if you shoot at a Drag world using Gravity-only angles)
            # Find where the bullet actually hits when fired with (az_g, el_g) under Drag model at ToF = tof_g
            actual_pos, _ = model_drag.state_at(tof_g, launch_point, az_g, el_g)
            miss_distance_m = float(np.linalg.norm(actual_pos.to_array() - target_pos.to_array()))

            print(
                f"{r:>15.1f}m | "
                f"El: {el_g:>6.3f} rad ({t_grav_ms:>5.3f}ms) | "
                f"El: {el_d:>6.3f} rad ({t_drag_ms:>5.3f}ms) | "
                f"El Diff: {el_diff_mrad:>+.2f} mrad (Miss: {miss_distance_m * 100:.1f}cm)"
            )
        else:
            print(f"{r:>15.1f}m | Solvers failed to converge.")

    print("=" * 85)
    print("\n[Analysis Guide for Jetson Orin Nano]")
    print("1. Latency Impact:")
    print("   - Gravity Only (Analytic) uses an O(1) formula, running in sub-millisecond times.")
    print("   - With Air Drag (RK4) runs numerical integration loops, taking significantly longer.")
    print("2. Accuracy Threshold:")
    print("   - Observe the 'Miss' column. If you only apply Gravity-Only calculations,")
    print("     the bullet will drop too much at longer ranges due to drag.")
    print("   - At 100m, the miss distance is usually tiny, meaning Gravity-Only is fully sufficient.")
    print("   - At 300m+, the miss distance becomes massive (often decimeters to meters),")
    print("     making Air Drag calculations necessary despite the CPU latency cost.")


if __name__ == "__main__":
    main()
