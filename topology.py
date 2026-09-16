import torch

def generate_poc_topologies(num_grid_nodes=100, num_mesh_nodes=16, feature_dim=3):
    """
    Generates data arrays representing the physical environment configurations.
    """
    # 1. Weather properties at time t: [Temperature, Wind Speed, Pressure]
    grid_features = torch.randn(num_grid_nodes, feature_dim) 
    
    # 2. Structural internal tracking space variables
    mesh_latent = torch.zeros(num_mesh_nodes, 16) 
    
    # 3. Simulate spatial mapping indices [2, num_connections]
    # Grid-to-Mesh links
    g2m_src = torch.randint(0, num_grid_nodes, (300,))
    g2m_tgt = torch.randint(0, num_mesh_nodes, (300,))
    g2m_idx = torch.stack([g2m_src, g2m_tgt], dim=0)
    g2m_dist = torch.rand(300, 1) # Spherical physical distance metric
    
    # Mesh-to-Mesh internal messaging links
    m2m_idx = torch.randint(0, num_mesh_nodes, (2, 64))   
    m2m_dist = torch.rand(64, 1)
    
    # Mesh-to-Grid return mapping links
    m2g_src = torch.randint(0, num_mesh_nodes, (300,))
    m2g_tgt = torch.randint(0, num_grid_nodes, (300,))
    m2g_idx = torch.stack([m2g_src, m2g_tgt], dim=0)
    m2g_dist = torch.rand(300, 1)
    
    return {
        "grid_x": grid_features, 
        "mesh_x": mesh_latent, 
        "g2m_idx": g2m_idx, 
        "g2m_dist": g2m_dist, 
        "m2m_idx": m2m_idx, 
        "m2m_dist": m2m_dist, 
        "m2g_idx": m2g_idx, 
        "m2g_dist": m2g_dist
    }