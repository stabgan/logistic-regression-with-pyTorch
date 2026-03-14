"""Logistic Regression on MNIST using PyTorch."""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets


class LogisticRegressionModel(nn.Module):
    """Simple logistic regression classifier."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def train(model, device, train_loader, test_loader, num_epochs, learning_rate):
    """Train the model and periodically evaluate on the test set."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    step = 0
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images = images.view(-1, 28 * 28).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            step += 1

            if step % 1000 == 0:
                accuracy = evaluate(model, device, test_loader)
                print(
                    f"Epoch: {epoch + 1}/{num_epochs} | "
                    f"Step: {step} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Test Accuracy: {accuracy:.2f}%"
                )


def evaluate(model, device, test_loader):
    """Evaluate model accuracy on a data loader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 28 * 28).to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100.0 * correct / total


def main():
    # ── Hyperparameters ──────────────────────────────────────────
    batch_size = 80
    n_iters = 30_000
    learning_rate = 1e-4
    input_dim = 28 * 28
    output_dim = 10

    # ── Device ───────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Dataset ──────────────────────────────────────────────────
    train_dataset = dsets.MNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=True
    )
    test_dataset = dsets.MNIST(
        root="./data", train=False, transform=transforms.ToTensor()
    )

    num_epochs = int(n_iters / (len(train_dataset) / batch_size))

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    # ── Model ────────────────────────────────────────────────────
    model = LogisticRegressionModel(input_dim, output_dim).to(device)

    # ── Train ────────────────────────────────────────────────────
    train(model, device, train_loader, test_loader, num_epochs, learning_rate)

    # ── Final evaluation ─────────────────────────────────────────
    final_accuracy = evaluate(model, device, test_loader)
    print(f"\nFinal Test Accuracy: {final_accuracy:.2f}%")


if __name__ == "__main__":
    main()
