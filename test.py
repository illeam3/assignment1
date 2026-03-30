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
os.makedirs("results", exist_ok=True)

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
            x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
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
            x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
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


def save_visualizations(model, loader, attack_fn, attack_name, eps, num_samples=5, eps_step=None, k=None):
    model.eval()
    saved = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            clean_outputs = model(images)
            clean_preds = clean_outputs.argmax(dim=1)

        if attack_name in ["fgsm_targeted", "pgd_targeted"]:
            targets = make_target_labels(labels)
            if attack_name == "fgsm_targeted":
                adv_images = attack_fn(model, images, targets, eps)
            else:
                adv_images = attack_fn(model, images, targets, eps, eps_step, k)
        else:
            if attack_name == "fgsm_untargeted":
                adv_images = attack_fn(model, images, labels, eps)
            else:
                adv_images = attack_fn(model, images, labels, eps, eps_step, k)

        with torch.no_grad():
            adv_outputs = model(adv_images)
            adv_preds = adv_outputs.argmax(dim=1)

        for i in range(images.size(0)):
            if saved >= num_samples:
                return

            original = images[i].detach().cpu().squeeze().numpy()
            adversarial = adv_images[i].detach().cpu().squeeze().numpy()
            perturbation = adversarial - original

            fig, axes = plt.subplots(1, 3, figsize=(9, 3))

            axes[0].imshow(original, cmap="gray")
            axes[0].set_title(f"Original\nPred: {clean_preds[i].item()}")
            axes[0].axis("off")

            axes[1].imshow(adversarial, cmap="gray")
            axes[1].set_title(f"Adversarial\nPred: {adv_preds[i].item()}")
            axes[1].axis("off")

            axes[2].imshow(perturbation, cmap="gray")
            axes[2].set_title("Perturbation")
            axes[2].axis("off")

            plt.tight_layout()
            save_path = f"results/{attack_name}_{saved+1}.png"
            plt.savefig(save_path)
            plt.close(fig)

            saved += 1


model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 3

for epoch in range(epochs):
    train(model, train_loader, criterion, optimizer, device)
    acc = evaluate(model, test_loader, device)
    print(f"Epoch {epoch+1}/{epochs}, Test Accuracy: {acc:.2f}%")

eps_list = [0.05, 0.1, 0.2, 0.3]
k = 10

print("\n=== Attack Success Rates on MNIST ===")
print("eps\tFGSM-U\tFGSM-T\tPGD-U\tPGD-T")

for eps in eps_list:
    eps_step = eps / 10

    fgsm_u = attack_success_rate_untargeted_fgsm(model, test_loader, eps, max_samples=100)
    fgsm_t = attack_success_rate_targeted_fgsm(model, test_loader, eps, max_samples=100)
    pgd_u = attack_success_rate_untargeted_pgd(model, test_loader, eps, eps_step, k, max_samples=100)
    pgd_t = attack_success_rate_targeted_pgd(model, test_loader, eps, eps_step, k, max_samples=100)

    print(f"{eps:.2f}\t{fgsm_u:.2f}%\t{fgsm_t:.2f}%\t{pgd_u:.2f}%\t{pgd_t:.2f}%")

# 시각화 저장용 eps 설정
vis_eps = 0.3
vis_eps_step = vis_eps / 10
vis_k = 10

save_visualizations(model, test_loader, fgsm_untargeted, "fgsm_untargeted", vis_eps, num_samples=5)
save_visualizations(model, test_loader, fgsm_targeted, "fgsm_targeted", vis_eps, num_samples=5)
save_visualizations(model, test_loader, pgd_untargeted, "pgd_untargeted", vis_eps, num_samples=5, eps_step=vis_eps_step, k=vis_k)
save_visualizations(model, test_loader, pgd_targeted, "pgd_targeted", vis_eps, num_samples=5, eps_step=vis_eps_step, k=vis_k)

print("\nSaved visualization images to results/")