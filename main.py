import torch

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance

from plot import *

# PyTorch device boilerplate
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def optimize_mesh(
    source_mesh: Meshes,
    target_mesh: Meshes,
    deformation_offsets: torch.Tensor,
    training_optimizer: torch.optim.Optimizer,
    num_vertex_samples: int = 100,
    num_iterations: int = 50
) -> tuple[Meshes, list]:
    """
    Optimize source_mesh to fit target_mesh

    Args:
        source_mesh: The source mesh to optimize
        target_mesh: The target mesh to use as labels
        deformation_offsets: The tensor to store deformation predictions
        training_optimizer: PyTorch optimizer instance
        num_vertex_samples: Number of vertex samples to take per training step
        num_iterations: Number of training steps to take
    """
    losses = []
    for _ in range(num_iterations):
        training_optimizer.zero_grad()
        
        # Use `deformation_offsets` to apply learned deformations
        predicted_mesh = source_mesh.offset_verts(deformation_offsets)
        
        # Sample points from the surface of meshes
        predicted_samples = sample_points_from_meshes(predicted_mesh, num_vertex_samples)
        target_samples = sample_points_from_meshes(target_mesh, num_vertex_samples)

        # Apply the loss function
        loss, _ = chamfer_distance(target_samples, predicted_samples)
        losses.append(loss.detach().cpu().item())
        loss.backward()
        training_optimizer.step()
    return predicted_mesh, losses

# Load meshes
source_mesh_path = './data/sphere1.obj'
target_mesh_path = './data/cube1.obj'
source_verts, source_faces, source_aux = load_obj(source_mesh_path, load_textures=False)
target_verts, target_faces, target_aux = load_obj(target_mesh_path, load_textures=False)
source_mesh = Meshes(verts=[source_verts.to(device)], faces=[source_faces.verts_idx.to(device)])
target_mesh = Meshes(verts=[target_verts.to(device)], faces=[target_faces.verts_idx.to(device)])

# Set hyperparameters
num_iterations = 100         # how many training passes to run before ending
num_vertex_samples = 200     # how many points to sample from the surface of vertices
learning_rate = 1.0          # learning rate
momentum = 0.9               # momentum for optimizer

# Train
deformation_offsets = torch.full(source_mesh.verts_packed().shape, 0.0, device=device,
                                  requires_grad=True)
optimizer = torch.optim.SGD([deformation_offsets], lr=learning_rate, momentum=momentum)
optimized_mesh, losses = optimize_mesh(
    source_mesh,
    target_mesh,
    deformation_offsets,
    optimizer,
    num_vertex_samples=num_vertex_samples,
    num_iterations=num_iterations
)

# Display outputs
new_mesh = optimized_mesh.verts_packed().clone().detach().cpu().squeeze().numpy()
scatter3d(new_mesh[:, 0], new_mesh[:, 1], new_mesh[:, 2], title=f'Optimized Mesh')
plot_loss(losses)
