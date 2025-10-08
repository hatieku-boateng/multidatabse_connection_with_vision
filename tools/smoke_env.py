"""
Lightweight environment smoke test for CI/Cloud.

Validates that PyTorch + torchvision (CPU) install correctly and a minimal
forward pass runs on CPU. Keeps it tiny to run fast in GitHub Actions and
Streamlit Community Cloud build steps.
"""

import sys


def main() -> int:
    try:
        import torch  # type: ignore
        from torchvision.models import resnet18  # type: ignore
    except Exception as exc:
        print(f"Import failed: {exc}")
        return 2

    try:
        device = torch.device("cpu")
        model = resnet18(weights=None).to(device).eval()
        with torch.no_grad():
            x = torch.zeros(1, 3, 224, 224, device=device)
            _ = model(x)
    except Exception as exc:
        print(f"CPU forward failed: {exc}")
        return 3

    print("OK: torch", torch.__version__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

