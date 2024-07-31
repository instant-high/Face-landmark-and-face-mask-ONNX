import torch
import torch.nn as nn
import numpy as np

class KeypointToMask(nn.Module):
    def __init__(self, image_height, image_width):
        super(KeypointToMask, self).__init__()
        self.image_height = image_height
        self.image_width = image_width

    def forward(self, dst_points):

        jawline_indices = list(range(0, 33))
        additional_indices = [46, 45, 44, 35, 34, 33]  # Specific additional keypoints [45, 34]
        
        # Combine the jawline and additional keypoints
        all_indices = jawline_indices + additional_indices
        segment1 = dst_points[all_indices]

        segments = [segment1]

        # Create an empty mask
        bmask = torch.zeros((self.image_height, self.image_width), dtype=torch.float32)

        # Generate a grid of coordinates
        grid_y, grid_x = torch.meshgrid(torch.arange(self.image_height), torch.arange(self.image_width), indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=-1).float()

        def is_point_in_polygon(grid, points):
            num_points = points.shape[0]
            j = num_points - 1
            inside = torch.zeros(grid.shape[:-1], dtype=torch.float32)
            for i in range(num_points):
                xi, yi = points[i]
                xj, yj = points[j]
                # Check if the grid point is between yi and yj in the y-axis
                intersect = ((yi > grid[..., 1]) != (yj > grid[..., 1])) & \
                            (grid[..., 0] < (xj - xi) * (grid[..., 1] - yi) / (yj - yi) + xi)
                inside = inside + intersect
                j = i
            return inside % 2  # Even-Odd Rule to determine if inside

        # Fill the mask for each segment
        for segment in segments:
            if segment.shape[0] > 2:  # Need at least 3 points to form a polygon
                points = segment.reshape((-1, 2)).float()  # Ensure points are in (N, 2) shape
                mask_segment = is_point_in_polygon(grid, points)
                bmask = torch.clamp(bmask + mask_segment, 0, 1)  # Accumulate and clamp to binary

        bmask = bmask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

        return bmask

# Example usage
image_height, image_width = 256, 256
dst_points = np.random.rand(98, 2) * 256  # Replace with actual keypoints
dst_points = torch.from_numpy(dst_points).float()  # Ensure the dtype is float for PyTorch

model = KeypointToMask(image_height, image_width)
bmask = model(dst_points)
print(bmask.shape)  # Should be (1, 1, image_height, image_width)

# Define dummy input for ONNX export
dummy_input = torch.rand(98, 2, dtype=torch.float)

# Export the model to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "98kp_to_mask.onnx",
    input_names=['keypoints'],
    output_names=['facemask'],
    dynamic_axes={
        'dst_points': {0: 'num_keypoints'},
        'bmask': {2: 'height', 3: 'width'}
    },
    opset_version=11
)

print("ONNX model exported")

