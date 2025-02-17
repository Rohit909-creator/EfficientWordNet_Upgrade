import torch
import torch.nn as nn
import onnxruntime
import numpy as np
from torchvision.models import resnet50

class Bottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, downsample=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class Layer4(nn.Module):
    def __init__(self):
        super(Layer4, self).__init__()
        self.bottleneck1 = Bottleneck(1024, 512, 2048, stride=2, downsample=True)
        self.bottleneck2 = Bottleneck(2048, 512, 2048, stride=1, downsample=False)
        self.bottleneck3 = Bottleneck(2048, 512, 2048, stride=1, downsample=False)
    
    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        return x

# Example usage:
# layer4 = Layer4()
# x = torch.randn(1, 1024, 56, 56)  # Adjust size as needed
# output = layer4(x)
# print(output.shape)



class ONNXtoTorchModel(nn.Module):
    def __init__(self, onnx_path):
        super().__init__()
        # Load ONNX model
        self.session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.layer = Layer4()
        
    def forward(self, x):
        # Convert PyTorch tensor to numpy for ONNX Runtime
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
            
        # Run inference
        outputs = self.session.run(None, {self.input_name: x})
        out = torch.from_numpy(outputs[0])
        out = self.layer(out)
        # print("outputs", outputs)
        # Convert back to PyTorch tensor
        return out

if __name__ == "__main__":
    # Load model
    path = r"C:\Users\Rohit Francis\Desktop\Codes\Pathor Wake Word\EfficientWord-Net\eff_word_net\models\resnet_50_arc\slim_93%_accuracy_72.7390%.onnx"

    model = ONNXtoTorchModel(path)
    
    # Create random input based on model's input shape
    random_input = torch.randn(model.input_shape)
    print(model.named_modules)
    # Run prediction
    print(f"Input shape: {random_input.shape}")
    output = model(random_input)
    print(f"Output shape: {output.shape}")
    
    # model2 = resnet50()
    # print(model2.named_modules)
    
    # layer4 = Layer4()
    # x = torch.randn(1, 1024, 56, 56)  # Adjust size as needed
    # output = layer4(x)
    # print(layer4)
    # print(output.shape)