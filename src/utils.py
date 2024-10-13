import torch

def predict(model, image, device, return_probs=False):
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        if return_probs:
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            return probabilities
        else:
            _, predicted = torch.max(outputs, 1)
            return predicted.item()
