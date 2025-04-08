# import onnx
# import numpy as np
# from onnx import numpy_helper
# import torchvision

# def analyze_onnx_model(model_path):
#     # Load ONNX model
#     model = onnx.load(model_path)
    
#     print("\n=== ONNX Model Architecture ===")
    
#     # Print Input Information
#     print("\nModel Inputs:")
#     # print("-" * 50)
#     for input in model.graph.input:
#         print(f"Input Name: {input.name}")
#         try:
#             shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
#             print(f"Input Shape: {shape}")
#         except:
#             print("Shape not defined")
    
#     # Print Layer Information
#     print("\nModel Layers:")
#     # print("-" * 50)
#     for idx, node in enumerate(model.graph.node):
#         print(f"\nLayer {idx + 1}:")
#         print(f"Operation Type: {node.op_type}")
#         print(f"Input Names: {node.input}")
#         print(f"Output Names: {node.output}")
#         if node.attribute:
#             print("Attributes:")
#             for attr in node.attribute:
#                 print(f"  - {attr.name}")
    
#     # Print Weights Information
#     print("\nModel Weights:")
#     # print("-" * 50)
#     for weight in model.graph.initializer:
#         # Convert weight to numpy array
#         weight_array = numpy_helper.to_array(weight)
        
#         print(f"\nWeight Name: {weight.name}")
#         print(f"Shape: {weight_array.shape}")
#         print(f"Data Type: {weight_array.dtype}")
#         # print(f"Min Value: {np.min(weight_array)}")
#         # print(f"Max Value: {np.max(weight_array)}")
#         # print(f"Mean Value: {np.mean(weight_array)}")
#         # print("First few values:", weight_array.flatten()[:5])
    
#     # Print Output Information
#     print("\nModel Outputs:")
#     print("-" * 50)
#     for output in model.graph.output:
#         print(f"Output Name: {output.name}")
#         try:
#             shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
#             print(f"Output Shape: {shape}")
#         except:
#             print("Shape not defined")
    
#     # Print Model Statistics
#     print("\nModel Statistics:")
#     # print("-" * 50)
#     total_params = sum(np.prod(numpy_helper.to_array(weight).shape) 
#                       for weight in model.graph.initializer)
#     print(f"Total Parameters: {total_params:,}")
#     print(f"Number of Layers: {len(model.graph.node)}")

# # Use the function
# # try:
# #     path = r"C:\Users\Rohit Francis\Desktop\Codes\Pathor Wake Word\EfficientWord-Net\eff_word_net\models\resnet_50_arc\slim_93%_accuracy_72.7390%.onnx"

# #     # analyze_onnx_model(path)  # Replace with your ONNX model path
# # except Exception as e:
# #     print(f"Error analyzing model: {e}")