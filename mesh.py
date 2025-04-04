import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import time
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
import random
from scipy.spatial import KDTree, Delaunay
import os

from models import UNetResNet_low, UNetResNet

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Preprocess image: transform to grayscale and 256x256 resolution
def preprocess_image(image_tensor, img_size):
    
    # If the input is a NumPy array, convert it to a PyTorch tensor
    if isinstance(image_tensor, np.ndarray):
        image_tensor = torch.tensor(image_tensor, dtype=torch.float32)

    # If the image has 3 dimensions (C x H x W), squeeze the channel dimension
    if len(image_tensor.shape) == 3 and image_tensor.shape[0] == 1:
        image_tensor = image_tensor.squeeze(0)

    # Resize the image to the target size
    image_resized = torch.nn.functional.interpolate(
        image_tensor.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
        size=img_size,
        mode="bilinear",
        align_corners=False
    ).squeeze(0).squeeze(0)  # Remove batch and channel dimensions

    # Normalize the image to [0, 1]
    image_resized = image_resized / 255.0 if image_resized.max() > 1 else image_resized

    # Add batch and channel dimensions for model input
    return image_resized.unsqueeze(0).unsqueeze(0)

# Postprocess segmentation output: get classes from model's output
def postprocess_mask(output_tensor):
    output_tensor = torch.softmax(output_tensor, dim=1)
    _, predicted_classes = torch.max(output_tensor, 1)
    return predicted_classes.squeeze(0).cpu().numpy()

# Extract contours of the fluid volume from segmentation mask
def extract_fluid_contours(mask, fluid_classes=[0, 1], fill_gaps=True, kernel_size=(5, 5)):  
    binary_mask = np.isin(mask, fluid_classes).astype(np.uint8)  # Convert to binary mask

    # If fill_gaps is True, apply dilation or closing to fill gaps between areas
    if fill_gaps:
        kernel = np.ones(kernel_size, np.uint8) 
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)  

    # Find contours in the filled binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 1. Filter contours by area (remove small noise)
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 10] 

    # 2. Sort contours by area (largest first)
    filtered_contours.sort(key=cv2.contourArea, reverse=True)
    return filtered_contours

# Helper function to check winding order (Counter-Clockwise)
def is_counter_clockwise(contour):
    signed_area = 0.0
    for i in range(len(contour)):
        x1, y1 = contour[i]
        x2, y2 = contour[(i + 1) % len(contour)]  # Wrap around to the first point
        signed_area += (x2 - x1) * (y2 + y1)

    return signed_area > 0  # Positive for CCW

def estimate_glass_width_image(glass_pixel_width, distance_cm, image_width_px=256, focal_length_mm=25, sensor_width_mm=15.5):
    # Compute focal length in pixels
    focal_length_px = (focal_length_mm * image_width_px) / sensor_width_mm

    # Compute real glass width in cm
    real_glass_width_cm = (distance_cm * glass_pixel_width) / focal_length_px * 4/3

    return real_glass_width_cm


def get_radius(mask, fluid_classes=[0, 1], kernel_size=(5, 5)):
    binary_mask = np.isin(mask, fluid_classes).astype(np.uint8)
    kernel = np.ones(kernel_size, np.uint8) 
    binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)  
    y, x = np.where(binary_mask)

    min_y = np.max(y)
    indices_min_y = np.where(y == min_y)[0]
    average_x = np.mean(x[indices_min_y])

    radius = 0
    R_dict = {}

    unique_y = np.unique(y)
    for yi in unique_y:
        x_vals = x[y == yi]  
        R = max(abs(np.max(x_vals) - average_x), abs(np.min(x_vals) - average_x))
        R_dict[yi] = R
        if R > radius:
            radius = R

    return radius, R_dict


def extract_volume(contours, R_dict):
    """
    Extracts a 3D volume from a list of 2D contours by adding height (z).
    Returns vertices and faces.
    """
    vertices = []

    for contour in contours:
        x = contour[:, 0, 0]
        y = contour[:, 0, 1]

        # Get the radius R associated with the closest y value from R_dict
        R_values = []
        for py in y:
            # Find the closest y in R_dict
            closest_y = min(R_dict.keys(), key=lambda y_val: abs(y_val - py))
            R_values.append(R_dict[closest_y])

        for px, py, R in zip(x, y, R_values):
            z_max = R * np.cos((px - np.mean(x)) * np.pi / (2 * R))  # Depth estimate

            vertices.append((px, py, z_max))   # Upper part
            vertices.append((px, py, -z_max))  # Lower part (mirrored)

    vertices = np.array(vertices)

    # Remove duplicate vertices
    unique_vertices = []
    seen = set()
    for vertex in vertices:
        if tuple(vertex) not in seen:
            seen.add(tuple(vertex))
            unique_vertices.append(vertex)
    vertices = np.array(unique_vertices)

    # Compute faces using Convex Hull
    hull = ConvexHull(vertices)
    faces = hull.simplices  # Triangular faces

    return vertices, faces

def create_3d_mesh(vertices, faces, Npoints, mesh_size):
    # Compute bounding box of the volume
    min_bounds = np.min(vertices, axis=0)
    max_bounds = np.max(vertices, axis=0)
    
    # Create Delaunay triangulation to check if points are inside the volume
    hull = Delaunay(vertices)
    
    # Initialize KDTree for efficient nearest neighbor search
    existing_tree = KDTree(vertices)  # Ensure points are inside the volume
    
    generated_points = list(vertices)  # Start with input vertices
    attempts = 0
    
    while len(generated_points) < Npoints + len(vertices) and attempts < 10 * Npoints:
        # Generate a random point inside the bounding box
        random_point = np.random.uniform(min_bounds, max_bounds)
        
        # Check if the point is inside the convex hull
        if hull.find_simplex(random_point) < 0:
            attempts += 1
            continue
        
        # Check distance constraint
        tree = KDTree(generated_points)
        if tree.query(random_point)[0] < mesh_size:
            attempts += 1
            continue
        
        # If the point is valid, add it
        generated_points.append(random_point)
        attempts += 1
    
    generated_points = np.array(generated_points)

    return generated_points

def plot_3d_mesh(vertices, faces, points):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Plot surface
    ax.add_collection3d(Poly3DCollection(vertices[faces], alpha=0.3, facecolor="cyan", edgecolor="k"))

    # Plot points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="red", alpha=0.6, s=3)

    # Labels and view
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=30, azim=120)  

    plt.show()
    
def plot_results(input_img, output_mask):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(input_img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask")
    plt.imshow(output_mask, cmap="jet") 
    plt.axis("off")

    plt.show()
    
    
"""
------------------- MAIN -------------------
"""
# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetResNet_low(n_classes=3).to(device)
model.load_state_dict(torch.load("checkpoints/model_epoch_100.pth", map_location=device), strict=False)
model.eval()

data = torch.load("data/dataset.pt")
test_images, test_masks = data["test"]
Nx = test_images[0].shape[-2]
Ny = test_images[0].shape[-1]
input_tensor = preprocess_image(test_images[6], img_size=(Nx, Ny)).to(device)

# Inference
start_time = time.time()
with torch.no_grad():
    output_tensor = model(input_tensor)
output_mask = postprocess_mask(output_tensor)
end_time = time.time()
execution_time = end_time - start_time
print(f"Total execution time: {execution_time:.4f} seconds")

# plot results
plot_results(input_tensor[0][0], output_mask)

# Extract fluid contours and generate mesh
fluid_contours = extract_fluid_contours(output_mask)

# Using random points
radius, R_dict = get_radius(output_mask)
diameter_cm = estimate_glass_width_image(radius*2, 40)
print(f"Diameter of the glass (cm): ", diameter_cm)
vertices, faces = extract_volume(fluid_contours, R_dict)
points = create_3d_mesh(vertices, faces, 500, 2)
print(f"Number of points in the second option: ", points.shape)
plot_3d_mesh(vertices, faces, points)

