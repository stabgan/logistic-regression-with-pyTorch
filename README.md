# Logistic Regression with PyTorch

Multiclass logistic regression trained on the MNIST handwritten-digit dataset using PyTorch.

## What It Does

Trains a single fully-connected layer (`nn.Linear`) to classify 28×28 grayscale digit images into 10 classes (0–9). Despite its simplicity, the model reaches ~82 % test accuracy after 30 000 gradient steps, making it a clean baseline before moving to deeper architectures.

## Architecture

```
Input (784) ──▶ Linear (784 → 10) ──▶ CrossEntropyLoss
```

- Flattened 28×28 images → 784-dim input
- Single linear layer → 10 logits
- SGD optimizer, learning rate 1e-4, batch size 80

## Dataset

[MNIST](http://yann.lecun.com/exdb/mnist/) — 60 000 training / 10 000 test images of handwritten digits. Downloaded automatically by `torchvision` on first run.

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python 3.8+ | Language |
| 🔥 PyTorch | Model, training loop, autograd |
| 🖼 torchvision | MNIST dataset & transforms |

## Getting Started

```bash
# Install dependencies
pip install torch torchvision

# Train the model
python logistic-regression.py
```

Training logs print every 1 000 steps with current loss and test accuracy. A CUDA GPU is used automatically when available.

## ⚠️ Known Issues

- No learning-rate scheduler — accuracy plateaus with longer training.
- No data augmentation (not typical for logistic regression, but limits ceiling).
- Model checkpoint saving is not implemented.

## License

MIT
