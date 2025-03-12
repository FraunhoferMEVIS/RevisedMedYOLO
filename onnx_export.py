import argparse
import torch
from models3D.model import attempt_load, Model


class ExportModel(Model):
    def forward(self, x):
        x = x.to(torch.float16)
        output, _ = super().forward(x)
        output = output.to(torch.float32)
        return output


def main():
    parser = argparse.ArgumentParser(description='Export 3D YOLO model to ONNX')
    parser.add_argument('--weights', type=str, required=True, help='Path to the model weights file (.pt)')
    parser.add_argument('--output', type=str, required=True, help='Path to save the ONNX model')
    parser.add_argument('--imgsz', type=int, default=350, help='Input image size (default: 350)')
    parser.add_argument('--input-channels', type=int, default=1, help='Number of input channels (default: 1)')
    args = parser.parse_args()
    
    device = torch.device('cuda:0')
    model = attempt_load(args.weights, map_location=device)
    model.eval()
    model.half()
    
    # Hack to override the forward method
    model.__class__ = ExportModel
    
    dummy_input = torch.randn(1, args.input_channels, args.imgsz, args.imgsz, args.imgsz,
                              dtype=torch.float32, device=device)
    
    torch.onnx.export(
        model, 
        dummy_input, 
        args.output, 
        input_names=['input'], 
        output_names=['output'],
        opset_version=20
    )
    print(f"Model has been exported to {args.output}")

if __name__ == '__main__':
    main()