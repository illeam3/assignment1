import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

set_seed(42)

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train(model, loader, criterion, optimizer, device):
    model.train()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    return accuracy


def fgsm_untargeted(model, x, label, eps):
    x_adv = x.clone().detach().to(device)
    label = label.to(device)
    x_adv.requires_grad_(True)

    outputs = model(x_adv)
    loss = nn.CrossEntropyLoss()(outputs, label)

    model.zero_grad()
    loss.backward()

    x_adv = x_adv + eps * x_adv.grad.sign()
    x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()


def fgsm_targeted(model, x, target, eps):
    x_adv = x.clone().detach().to(device)
    target = target.to(device)
    x_adv.requires_grad_(True)

    outputs = model(x_adv)
    loss = nn.CrossEntropyLoss()(outputs, target)

    model.zero_grad()
    loss.backward()

    x_adv = x_adv - eps * x_adv.grad.sign()
    x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()


def pgd_untargeted(model, x, label, eps, eps_step, k):
    x_orig = x.clone().detach().to(device)
    label = label.to(device)
    x_adv = x_orig.clone().detach()

    for _ in range(k):
        x_adv.requires_grad_(True)

        outputs = model(x_adv)
        loss = nn.CrossEntropyLoss()(outputs, label)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + eps_step * x_adv.grad.sign()

            # 원본 이미지 기준 eps-ball 안으로 projection
            x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)

            # 이미지 범위 [0, 1] 유지
            x_adv = torch.clamp(x_adv, 0, 1)

        x_adv = x_adv.detach()

    return x_adv


def pgd_targeted(model, x, target, eps, eps_step, k):
    x_orig = x.clone().detach().to(device)
    target = target.to(device)
    x_adv = x_orig.clone().detach()

    for _ in range(k):
        x_adv.requires_grad_(True)

        outputs = model(x_adv)
        loss = nn.CrossEntropyLoss()(outputs, target)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv - eps_step * x_adv.grad.sign()

            # 원본 이미지 기준 eps-ball 안으로 projection
            x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)

            # 이미지 범위 [0, 1] 유지
            x_adv = torch.clamp(x_adv, 0, 1)

        x_adv = x_adv.detach()

    return x_adv


def make_target_labels(labels):
    return (labels + 1) % 10


def attack_success_rate_untargeted_fgsm(model, loader, eps, max_samples=100):
    model.eval()
    success = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        remaining = max_samples - total
        if remaining <= 0:
            break

        images = images[:remaining]
        labels = labels[:remaining]

        x_adv = fgsm_untargeted(model, images, labels, eps)

        with torch.no_grad():
            outputs_adv = model(x_adv)
            preds_adv = outputs_adv.argmax(dim=1)

        success += (preds_adv != labels).sum().item()
        total += labels.size(0)

    return 100 * success / total


def attack_success_rate_targeted_fgsm(model, loader, eps, max_samples=100):
    model.eval()
    success = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        remaining = max_samples - total
        if remaining <= 0:
            break

        images = images[:remaining]
        labels = labels[:remaining]
        targets = make_target_labels(labels)

        x_adv = fgsm_targeted(model, images, targets, eps)

        with torch.no_grad():
            outputs_adv = model(x_adv)
            preds_adv = outputs_adv.argmax(dim=1)

        success += (preds_adv == targets).sum().item()
        total += labels.size(0)

    return 100 * success / total


def attack_success_rate_untargeted_pgd(model, loader, eps, eps_step, k, max_samples=100):
    model.eval()
    success = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        remaining = max_samples - total
        if remaining <= 0:
            break

        images = images[:remaining]
        labels = labels[:remaining]

        x_adv = pgd_untargeted(model, images, labels, eps, eps_step, k)

        with torch.no_grad():
            outputs_adv = model(x_adv)
            preds_adv = outputs_adv.argmax(dim=1)

        success += (preds_adv != labels).sum().item()
        total += labels.size(0)

    return 100 * success / total


def attack_success_rate_targeted_pgd(model, loader, eps, eps_step, k, max_samples=100):
    model.eval()
    success = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        remaining = max_samples - total
        if remaining <= 0:
            break

        images = images[:remaining]
        labels = labels[:remaining]
        targets = make_target_labels(labels)

        x_adv = pgd_targeted(model, images, targets, eps, eps_step, k)

        with torch.no_grad():
            outputs_adv = model(x_adv)
            preds_adv = outputs_adv.argmax(dim=1)

        success += (preds_adv == targets).sum().item()
        total += labels.size(0)

    return 100 * success / total


model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 3

for epoch in range(epochs):
    train(model, train_loader, criterion, optimizer, device)
    acc = evaluate(model, test_loader, device)
    print(f"Epoch {epoch+1}/{epochs}, Test Accuracy: {acc:.2f}%")


eps = 0.1
eps_step = 0.01
k = 10

untargeted_fgsm_rate = attack_success_rate_untargeted_fgsm(model, test_loader, eps, max_samples=100)
targeted_fgsm_rate = attack_success_rate_targeted_fgsm(model, test_loader, eps, max_samples=100)

untargeted_pgd_rate = attack_success_rate_untargeted_pgd(
    model, test_loader, eps, eps_step, k, max_samples=100
)
targeted_pgd_rate = attack_success_rate_targeted_pgd(
    model, test_loader, eps, eps_step, k, max_samples=100
)

print(f"Untargeted FGSM Success Rate (eps={eps}): {untargeted_fgsm_rate:.2f}%")
print(f"Targeted FGSM Success Rate (eps={eps}): {targeted_fgsm_rate:.2f}%")
print(f"Untargeted PGD Success Rate (eps={eps}, step={eps_step}, k={k}): {untargeted_pgd_rate:.2f}%")
print(f"Targeted PGD Success Rate (eps={eps}, step={eps_step}, k={k}): {targeted_pgd_rate:.2f}%")