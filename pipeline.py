import torch
import torch.nn as nn
from layers import WeatherMessageRouter

class DeepMindWeatherPOC(nn.Module):
    """
    Hierarchical System Framework.
    Decouples raw coordinate inputs from structural atmospheric processing.
    """
    def __init__(self, feature_dim=3, latent_dim=16):
        super(DeepMindWeatherPOC, self).__init__()
        
        # Initial projection to keep runtime memory footprints light
        self.grid_embed = nn.Linear(feature_dim, latent_dim)
        
        # Functional Processing Blocks
        self.grid_to_mesh = WeatherMessageRouter(latent_dim, latent_dim, latent_dim)
        self.mesh_to_mesh = WeatherMessageRouter(latent_dim, latent_dim, latent_dim)
        self.mesh_to_grid = WeatherMessageRouter(latent_dim, latent_dim, latent_dim)
        
        # Final output layer targeting physical metrics
        self.forecast_head = nn.Linear(latent_dim, feature_dim)

    def forward(self, grid_x, mesh_x, g2m_idx, g2m_dist, m2m_idx, m2m_dist, m2g_idx, m2g_dist):
        # Phase 1: Embed raw local conditions into a high-dimensional space
        h_grid = torch.relu(self.grid_embed(grid_x))
        
        # Phase 2: Route local values up to the global 3D spherical mesh
        h_mesh = self.grid_to_mesh(h_grid, mesh_x, g2m_idx, g2m_dist)
        
        # Phase 3: Execute step calculations within the global mesh
        h_mesh = self.mesh_to_mesh(h_mesh, h_mesh, m2m_idx, m2m_dist)
        
        # Phase 4: Decode global developments back down to localized regional outputs
        h_grid_out = self.mesh_to_grid(h_mesh, h_grid, m2g_idx, m2g_dist)
        
        # Phase 5: Apply a residual skip connection to predict changes (deltas) over time
        return grid_x + self.forecast_head(h_grid_out)