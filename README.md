# 🔢 Logistic Regression with PyTorch

A clean implementation of logistic regression using PyTorch, trained and evaluated on the MNIST handwritten digit dataset. The model classifies 28×28 grayscale digit images into 10 classes (0–9).

## 📖 Description

This project demonstrates how to build a simple logistic regression classifier from scratch using PyTorch's `nn.Module`. It covers the full pipeline: loading data, defining a model, training with SGD, and evaluating accuracy on a held-out test set. A great starting point for learning PyTorch fundamentals.

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python 3 | Programming language |
| 🔥 PyTorch | Deep learning framework |
| 🖼️ torchvision | MNIST dataset & transforms |

## 📦 Dependencies

- `torch`
- `torchvision`

Install with:

```bash
pip install torch torchvision
```

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/stabgan/logistic-regression-with-pyTorch.git
cd logistic-regression-with-pyTorch
```

2. Install dependencies:

```bash
pip install torch torchvision
```

3. Run the training script:

```bash
python logistic-regression.py
```

The script will automatically download the MNIST dataset on first run. Training progress (loss and accuracy) is printed every 1000 iterations. GPU is used automatically if available.

## ⚠️ Known Issues

- The MNIST images referenced in the original README (from `saedsayad.com` and `ibb.co`) are external links and may break over time.
- No model checkpointing — the trained model is not saved to disk.
- No command-line arguments for hyperparameters (batch size, learning rate, epochs are hardcoded).

## 📄 License

See [LICENSE](LICENSE) for details.
