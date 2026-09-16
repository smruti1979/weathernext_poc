import argparse
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from topology import generate_poc_topologies
from pipeline import DeepMindWeatherPOC

# =====================================================================
# VISUALIZATION HELPER BLOCK (ARCHITECTURAL METRIC & ERROR TRACKER)
# =====================================================================
def plot_architectural_metrics(horizons, variance_history, node_trajectories, feature_labels, mse_history):
    """
    Renders diagnostic tracking subplots to analyze error compounding, 
    structural stability, and multi-step accuracy divergence.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("DeepMind Weather Hierarchy POC - Architectural Diagnostic Dashboard", fontsize=14, fontweight='bold')

    # Chart 1: Structural Latent Variance / Divergence over Time
    ax1.plot(horizons, variance_history, marker='o', color='#1f77b4', linewidth=2, label='Spatial Variance')
    ax1.set_title("System-Wide Latent Feature Variance", fontsize=11, pad=10)
    ax1.set_xlabel("Forecast Horizon Step (Hours)", fontsize=10)
    ax1.set_ylabel("Mean Variance (σ²)", fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xticks(horizons)
    ax1.legend(loc='upper left')

    # Chart 2: Autoregressive Node Trajectories
    colors = ['#e377c2', '#2ca02c', '#d62728']
    for i, feature_name in enumerate(feature_labels):
        ax2.plot(horizons, node_trajectories[:, i], marker='s', color=colors[i], linewidth=2, label=feature_name)
    ax2.set_title("Atmospheric Vector Path (Station Node 0)", fontsize=11, pad=10)
    ax2.set_xlabel("Forecast Horizon Step (Hours)", fontsize=10)
    ax2.set_ylabel("Normalized Magnitude", fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_xticks(horizons)
    ax2.legend(loc='lower left')

    # Chart 3: Accuracy Evaluation Block (MSE Tracking over Time)
    ax3.plot(horizons, mse_history, marker='D', color='#ff7f0e', linewidth=2, linestyle='-', label='Model Mean Squared Error')
    ax3.set_title("Model Accuracy Degradation (vs Ground Truth)", fontsize=11, pad=10)
    ax3.set_xlabel("Forecast Horizon Step (Hours)", fontsize=10)
    ax3.set_ylabel("MSE Evaluation Loss", fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.set_xticks(horizons)
    ax3.legend(loc='upper left')

    plt.tight_layout()
    
    output_filename = "weather_poc_diagnostics.png"
    plt.savefig(output_filename, dpi=150)
    print(f"\n[Visualization Saved] Updated dashboard charts generated successfully: '{output_filename}'")
    plt.show()


# =====================================================================
# SIMULATION ENGINE WITH METRIC AGGREGATION & CLI COUPLING
# =====================================================================
def run_autoregressive_simulation(grid_nodes, mesh_nodes, horizon_hours):
    print("=============================================================")
    print("   INITIALIZING DEEPMIND WEATHER ARCHITECTURE POC ENGINE     ")
    print(f"   CONFIG: Grid Nodes: {grid_nodes} | Mesh Nodes: {mesh_nodes} | Horizon: {horizon_hours}h")
    print("=============================================================\n")
    
    # 1. Fetch system topologies using CLI parameters dynamically
    data = generate_poc_topologies(num_grid_nodes=grid_nodes, num_mesh_nodes=mesh_nodes, feature_dim=3)
    
    # 2. Instantiate our structural model configuration
    model = DeepMindWeatherPOC(feature_dim=3, latent_dim=16)
    model.eval() # Set model to evaluation mode for inference
    
    current_grid_state = data["grid_x"]
    
    # --- SIMULATED GROUND TRUTH PATHWAY ---
    ground_truth_trajectory = [current_grid_state.clone()]
    simulated_true_state = current_grid_state.clone()
    for _ in range(horizon_hours):
        simulated_true_state = simulated_true_state + (torch.sin(simulated_true_state * 0.1) * 0.05) + (torch.randn_like(simulated_true_state) * 0.02)
        ground_truth_trajectory.append(simulated_true_state)
    
    # 3. Initialize tracking buffers to capture data points for the visualizer
    horizons = list(range(1, horizon_hours + 1))
    variance_history = []
    node_0_trajectories = []
    mse_history = []
    feature_labels = ["Temperature", "Wind Speed", "Surface Pressure"]
    
    print(f"Initial State Shape at t(0): {current_grid_state.shape}")
    print("Starting multi-step autoregressive forecast tracking...\n")
    
    # 4. Autoregressive Forecast & Evaluation Loop
    with torch.no_grad():
        for hour in horizons:
            
            # Predict the next step based on the current system state
            predicted_next_state = model(
                grid_x=current_grid_state, mesh_x=data["mesh_x"],
                g2m_idx=data["g2m_idx"], g2m_dist=data["g2m_dist"],
                m2m_idx=data["m2m_idx"], m2m_dist=data["m2m_dist"],
                m2g_idx=data["m2g_idx"], m2g_dist=data["m2g_dist"]
            )
            
            # --- EVALUATION METRICS HARVESTING ---
            target_ground_truth = ground_truth_trajectory[hour]
            
            # Compute Mean Squared Error (MSE) across all dynamic regional station grid cells
            mse_loss = torch.mean((predicted_next_state - target_ground_truth) ** 2).item()
            mse_history.append(mse_loss)
            
            # Track spatial structural variance to monitor model stability
            step_variance = torch.var(predicted_next_state).item()
            variance_history.append(step_variance)
            
            # Capture the physical feature outputs for localized station node 0
            sample_features = predicted_next_state.numpy()[0] # Isolate the first dimension row array
            node_0_trajectories.append(predicted_next_state.numpy()[0])
            
            # FIXED PRINT STATEMENT: Explicitly index sample_features [0] for Temperature scalar print mapping
            print(f"[Hour t+{hour:02d}] Pass Complete | Node 0 Temp: {sample_features[0]:.4f} | Horizon Evaluation MSE: {mse_loss:.4f}")
            
            # AUTOREGRESSIVE COUPLING STEP: Feed predictions forward as input for the next step
            current_grid_state = predicted_next_state

    print("\n=============================================================")
    print("   ARCHITECTURAL VALIDATION SUCCESSFUL: COMPUTATION COMPLETE  ")
    print("=============================================================")

    # 5. Execute Visualization Engine Block using collected metric arrays
    node_0_trajectories = np.array(node_0_trajectories)
    plot_architectural_metrics(horizons, variance_history, node_0_trajectories, feature_labels, mse_history)


# =====================================================================
# COMMAND LINE INTERFACE IMPLEMENTATION BLOCK
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLI Interface Engine for DeepMind Weather Next Spatial POC Framework.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configure interactive arguments with safe low-memory laptop default fallbacks
    parser.add_argument("-g", "--grid-nodes", type=int, default=100, 
                        help="Number of local ground monitoring station tracking grid coordinates.")
    parser.add_argument("-m", "--mesh-nodes", type=int, default=16, 
                        help="Number of continuous 3D spherical mesh latent nodes floating above grid.")
    parser.add_argument("-t", "--horizon", type=int, default=6, 
                        help="Autoregressive forecast horizon step timeline length in hours.")
    
    args = parser.parse_args()
    
    # Run simulation with validated runtime inputs
    run_autoregressive_simulation(
        grid_nodes=args.grid_nodes, 
        mesh_nodes=args.mesh_nodes, 
        horizon_hours=args.horizon
    )