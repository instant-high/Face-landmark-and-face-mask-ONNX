import os
import onnx
from onnx import version_converter, helper
from onnxsim import simplify

# File paths
INIT_MODEL = 'kps_student.onnx'
NEXT_MODEL = '98kp_to_mask.onnx'

# Load models
model1 = onnx.load(INIT_MODEL)
model2 = onnx.load(NEXT_MODEL)

# Add prefixes to avoid naming conflicts
model1 = onnx.compose.add_prefix(model1, prefix='kp_')
#model2 = onnx.compose.add_prefix(model2, prefix='next_')

# Inspect model outputs
def inspect_model(model):
    print("Model Inputs:")
    for input in model.graph.input:
        print(f"Name: {input.name}, Shape: {input.type.tensor_type.shape}")

    print("Model Outputs:")
    for output in model.graph.output:
        shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
        print(f"Name: {output.name}, Shape: {shape}")

print("Inspecting Model 1")
inspect_model(model1)
print("\nInspecting Model 2")
inspect_model(model2)

# Set IR and Opset versions to the higher version
higher_ir_version = max(model1.ir_version, model2.ir_version)
higher_opset_version = max(model1.opset_import[0].version, model2.opset_import[0].version)

model1.ir_version = higher_ir_version
model2.ir_version = higher_ir_version

model1 = version_converter.convert_version(model1, higher_opset_version)
model2 = version_converter.convert_version(model2, higher_opset_version)

# Define the input-output mapping for merging models
##('init_output', 'next_keypoints')  # Mapping the output from model1 to input of model2
io_map = [
    ('kp_output', 'keypoints')  # Mapping the output from model1 to input of model2
]

# Merge models
combined_model = onnx.compose.merge_models(
    model1, model2,
    io_map=io_map
)

# Function to add missing outputs
def add_missing_outputs(model, outputs):
    for output in outputs:
        output_name = output.name
        if output_name not in [o.name for o in model.graph.output]:
            # Create TensorValueInfo
            elem_type = output.type.tensor_type.elem_type
            shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
            new_output = helper.make_tensor_value_info(output_name, elem_type, shape)
            model.graph.output.append(new_output)

# Collect all outputs from both models
model1_outputs = [o for o in model1.graph.output]
model2_outputs = [o for o in model2.graph.output]

# Add missing outputs from model1 to the combined model
add_missing_outputs(combined_model, model1_outputs)

# Simplify the combined model
combined_model_simplified, check = simplify(combined_model)

# Inspect the combined model to check if all expected outputs are present
print("\nInspecting Combined Model")
inspect_model(combined_model_simplified)

# Save the combined model
name, _ = os.path.splitext(INIT_MODEL)
new_name = name + '_' + NEXT_MODEL
onnx.save(combined_model_simplified, new_name)

print("Merging and simplification done")
