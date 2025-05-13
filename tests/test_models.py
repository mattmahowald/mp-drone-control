import torch
from mp_drone_control.models.mobilenet import LandmarkClassifier


def test_landmark_classifier_forward_pass():
    model = LandmarkClassifier(input_dim=63, num_classes=10)
    model.eval()

    dummy_input = torch.randn(8, 63)  # batch of 8 landmark vectors
    output = model(dummy_input)

    assert output.shape == (8, 10), f"Expected output shape (8, 10), got {output.shape}"
    print("✅ test_landmark_classifier_forward_pass passed.")
