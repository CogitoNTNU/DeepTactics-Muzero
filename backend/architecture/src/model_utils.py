import torch

def save_model(model, filename="model.pth"):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

def load_model(model, filename="model.pth"):
    model.load_state_dict(torch.load(filename))
    model.eval()
    print(f"Model loaded from {filename}")
