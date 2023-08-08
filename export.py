import torch
from models import Wav2Lip
from models import SyncNet_color as SyncNet

def torch2Onnx(model, f="test.onnx", dynamic=False, simplify=False):
    """
    pytorch转onnx
    """
    # 输入placeholder
    dtype = torch.float32 
    device = torch.device('cpu')  
    image = torch.randn(1, 6, 256, 256, dtype=dtype, device=device)
    audio = torch.randn(1, 1, 80, 16, dtype=dtype, device=device)
    dummy_output = model(audio, image)
    inputs = (audio, image)
    input_names = ["audio", "images"]
    print(dummy_output.shape)

    # Export to ONNX format
    torch.onnx.export(
        model,
        inputs,
        f,
        input_names=input_names,
        output_names=["outputs"],
        # verbose=False,
        opset_version=12,
        dynamic_axes={
            "images": {0: "batch", 2: "h", 3: "w"},
            "ids": {0: "batch"},
            "outputs": {0: "batch", 2: "h", 3: "w"},
        }
        if dynamic
        else None,
    )
    if simplify:
        import onnx
        import onnxsim

        model_onnx = onnx.load(f)  # load onnx model

        print(f"simplifying with onnxsim {onnxsim.__version__}...")
        model_onnx, check = onnxsim.simplify(model_onnx)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_onnx, f)


# torch.Size([10, 1, 80, 16]) torch.Size([10, 6, 96, 96])
def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = torch.load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace("module.", "")] = v
    model.load_state_dict(new_s)
    model.eval()
    return model

if __name__ == "__main__":
    # model = load_model("checkpoints/wav2lip.pth")
    # model = Wav2Lip()
    # model.eval()
    # torch2Onnx(model)

    image = torch.randn(1, 15, 128, 256, dtype=torch.float32)
    audio = torch.randn(1, 1, 80, 16, dtype=torch.float32)
    model = SyncNet()
    model.eval()
    model(audio, image)
