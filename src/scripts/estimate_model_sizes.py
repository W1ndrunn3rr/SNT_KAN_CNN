import torch
import torchvision
from src.models.snt_cnn import UniversalCNN
from tabulate import tabulate


def estimate_model_sizes():
    """
    Estimates the number of parameters and file size for each model architecture
    in UniversalCNN.
    """
    model_types = [
        "KAN_FAST",
        "resnet50",
        "vgg16",
        "densenet121",
        "mobilenet_v2",
        "efficientnet_b0",
        "vit_b_16",
        "diatnet",
        "features_extractor",
    ]

    num_classes = 64

    results = []

    for model_type in model_types:
        try:
            # Instantiate the model
            model_wrapper = UniversalCNN(model_type=model_type, num_classes=num_classes)
            model = model_wrapper.model

            # Calculate total parameters
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # Estimate size in kB (assuming float32, 4 bytes per parameter)
            size_kb = (total_params * 4) / 1024

            results.append([model_type, f"{total_params:,}", f"{size_kb:,.2f}"])

        except (ValueError, ImportError) as e:
            results.append([model_type, "Błąd", str(e)])

    # Print results in a table
    headers = ["Architektura", "Liczba wag (parametrów)", "Szacowany rozmiar (kB)"]
    print(tabulate(results, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    estimate_model_sizes()
