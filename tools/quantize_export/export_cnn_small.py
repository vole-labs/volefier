# Small CIFAR-shaped residual CNN (Conv/BN/ReLU/MaxPool/Add/GlobalAvgPool/Gemm), random weights, for
# the ml-float graph test:  python3 tools/quantize_export/export_cnn_small.py
#   test_ml-float_onnx_infer 1 12345 model/cnn_small.onnx  /  ... 2 12345 model/cnn_small.onnx
import torch, torch.nn as nn
torch.manual_seed(0)
class Block(nn.Module):
    def __init__(s, c):
        super().__init__(); s.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False); s.b1 = nn.BatchNorm2d(c)
        s.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False); s.b2 = nn.BatchNorm2d(c)
    def forward(s, x): return torch.relu(s.b2(s.c2(torch.relu(s.b1(s.c1(x))))) + x)
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.stem = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        s.l1 = nn.Sequential(Block(16), Block(16)); s.up = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        s.l2 = nn.Sequential(Block(32), Block(32)); s.pool = nn.AdaptiveAvgPool2d(1); s.fc = nn.Linear(32, 10)
    def forward(s, x): return s.fc(torch.flatten(s.pool(s.l2(s.up(s.l1(s.stem(x))))), 1))
m = Net()
for mod in m.modules():   # non-trivial BN statistics
    if isinstance(mod, nn.BatchNorm2d):
        mod.running_mean.uniform_(-0.2, 0.2); mod.running_var.uniform_(0.5, 2.0); mod.weight.data.uniform_(0.5, 1.5); mod.bias.data.uniform_(-0.1, 0.1)
m.eval()
torch.onnx.export(m, torch.randn(1, 3, 32, 32), "model/cnn_small.onnx", opset_version=13, input_names=["input"], output_names=["logits"], dynamo=False)
print("wrote model/cnn_small.onnx")
