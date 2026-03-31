import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# 실행할 때마다 결과가 너무 달라지지 않도록 시드를 고정한다.
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# GPU가 있으면 GPU를 쓰고, 없으면 CPU로 실행한다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

set_seed(42)

# 결과 이미지 저장 폴더가 없으면 만든다.
os.makedirs("results", exist_ok=True)


# MNIST는 흑백 이미지라서 텐서 변환만 해도 충분하다.
def get_mnist_loaders(batch_size=64):
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# CIFAR-10은 일반적으로 간단한 augmentation을 같이 쓴다.
def get_cifar10_loaders(batch_size=128):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])

    test_transform = transforms.ToTensor()

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# MNIST용 간단한 CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # classifier
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


# CIFAR-10은 입력 크기가 작아서 ResNet18의 첫 부분을 조금 바꿔서 쓴다.
def get_cifar10_model():
    model = models.resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


# 한 epoch 동안 학습을 수행한다.
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


# clean test set에서 분류 정확도를 계산한다.
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

    return 100 * correct / total


# untargeted FGSM:
# 정답 클래스의 loss를 키워서 예측이 정답에서 멀어지게 만든다.
def fgsm_untargeted(model, x, label, eps):
    x_adv = x.clone().detach().to(device)
    label = label.to(device)

    # 입력 기준 gradient를 구해야 하므로 requires_grad를 켠다.
    x_adv.requires_grad_(True)

    outputs = model(x_adv)
    loss = nn.CrossEntropyLoss()(outputs, label)

    model.zero_grad()
    loss.backward()

    # loss를 증가시키는 방향으로 한 번 이동한다.
    x_adv = x_adv + eps * x_adv.grad.sign()

    # 이미지 값 범위를 유지한다.
    x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()


# targeted FGSM:
# 목표 클래스의 loss를 줄이는 방향으로 이동해서 특정 클래스로 유도한다.
def fgsm_targeted(model, x, target, eps):
    x_adv = x.clone().detach().to(device)
    target = target.to(device)

    x_adv.requires_grad_(True)

    outputs = model(x_adv)
    loss = nn.CrossEntropyLoss()(outputs, target)

    model.zero_grad()
    loss.backward()

    # targeted는 목표 클래스 쪽으로 가야 하므로 부호가 반대다.
    x_adv = x_adv - eps * x_adv.grad.sign()
    x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()


# untargeted PGD:
# 작은 step을 여러 번 반복하고, 매번 원본 주변 eps-ball 안으로 다시 projection한다.
def pgd_untargeted(model, x, label, eps, eps_step, k):
    x_orig = x.clone().detach().to(device)
    label = label.to(device)

    # 처음에는 원본 이미지에서 시작한다.
    x_adv = x_orig.clone().detach()

    for _ in range(k):
        x_adv.requires_grad_(True)

        outputs = model(x_adv)
        loss = nn.CrossEntropyLoss()(outputs, label)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            # untargeted이므로 loss를 키우는 방향으로 이동한다.
            x_adv = x_adv + eps_step * x_adv.grad.sign()

            # 원본 기준 eps 범위를 넘지 않도록 자른다.
            x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)

            # 픽셀 범위를 유지한다.
            x_adv = torch.clamp(x_adv, 0, 1)

        x_adv = x_adv.detach()

    return x_adv


# targeted PGD:
# 목표 클래스 방향으로 여러 번 이동하면서 점진적으로 target class로 유도한다.
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
            # targeted는 목표 클래스 loss를 줄이는 방향으로 이동한다.
            x_adv = x_adv - eps_step * x_adv.grad.sign()

            x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
            x_adv = torch.clamp(x_adv, 0, 1)

        x_adv = x_adv.detach()

    return x_adv


# targeted attack에서는 간단하게 (현재 라벨 + 1) % 10 을 목표 클래스로 쓴다.
def make_target_labels(labels, num_classes=10):
    return (labels + 1) % num_classes


# untargeted FGSM 성공률:
# 공격 후 예측이 정답과 달라지면 성공으로 본다.
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
            preds_adv = model(x_adv).argmax(dim=1)

        success += (preds_adv != labels).sum().item()
        total += labels.size(0)

    return 100 * success / total


# targeted FGSM 성공률:
# 공격 후 예측이 목표 클래스와 같으면 성공으로 본다.
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
            preds_adv = model(x_adv).argmax(dim=1)

        success += (preds_adv == targets).sum().item()
        total += labels.size(0)

    return 100 * success / total


# untargeted PGD 성공률
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
            preds_adv = model(x_adv).argmax(dim=1)

        success += (preds_adv != labels).sum().item()
        total += labels.size(0)

    return 100 * success / total


# targeted PGD 성공률
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
            preds_adv = model(x_adv).argmax(dim=1)

        success += (preds_adv == targets).sum().item()
        total += labels.size(0)

    return 100 * success / total


# 시각화를 위해 텐서를 numpy 이미지 형태로 바꾼다.
# MNIST는 (H, W), CIFAR-10은 (H, W, C) 형태로 맞춘다.
def tensor_to_image(tensor):
    x = tensor.detach().cpu()
    if x.dim() == 3:
        x = x.permute(1, 2, 0).numpy()
    else:
        x = x.squeeze().numpy()
    return x


# 원본 / 공격 이미지 / perturbation을 나란히 저장한다.
def save_visualizations(model, loader, attack_fn, attack_type, file_prefix, eps, num_samples=5, eps_step=None, k=None):
    model.eval()
    saved = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # 공격 전 예측
        with torch.no_grad():
            clean_preds = model(images).argmax(dim=1)

        # 공격 종류에 따라 필요한 인자가 조금 다르다.
        if attack_type == "fgsm_targeted":
            targets = make_target_labels(labels)
            adv_images = attack_fn(model, images, targets, eps)

        elif attack_type == "pgd_targeted":
            targets = make_target_labels(labels)
            adv_images = attack_fn(model, images, targets, eps, eps_step, k)

        elif attack_type == "fgsm_untargeted":
            adv_images = attack_fn(model, images, labels, eps)

        elif attack_type == "pgd_untargeted":
            adv_images = attack_fn(model, images, labels, eps, eps_step, k)

        else:
            raise ValueError(f"Unknown attack_type: {attack_type}")

        # 공격 후 예측
        with torch.no_grad():
            adv_preds = model(adv_images).argmax(dim=1)

        for i in range(images.size(0)):
            if saved >= num_samples:
                return

            original = tensor_to_image(images[i])
            adversarial = tensor_to_image(adv_images[i])
            perturbation = adversarial - original

            fig, axes = plt.subplots(1, 3, figsize=(9, 3))

            if original.ndim == 2:
                # MNIST
                axes[0].imshow(original, cmap="gray")
                axes[1].imshow(adversarial, cmap="gray")
                axes[2].imshow(perturbation, cmap="gray")
            else:
                # CIFAR-10
                axes[0].imshow(np.clip(original, 0, 1))
                axes[1].imshow(np.clip(adversarial, 0, 1))

                # perturbation은 그대로 두면 잘 안 보여서 0~1로 다시 맞춘다.
                p = perturbation
                p = (p - p.min()) / (p.max() - p.min() + 1e-8)
                axes[2].imshow(p)

            axes[0].set_title(f"Original\nPred: {clean_preds[i].item()}")
            axes[1].set_title(f"Adversarial\nPred: {adv_preds[i].item()}")
            axes[2].set_title("Perturbation")

            for ax in axes:
                ax.axis("off")

            plt.tight_layout()
            plt.savefig(f"results/{file_prefix}_{saved + 1}.png")
            plt.close(fig)

            saved += 1


# eps 값별로 FGSM / PGD 성공률 표를 출력한다.
def run_attack_table(model, loader, dataset_name, eps_list, k=10, max_samples=100):
    print(f"\n=== Attack Success Rates on {dataset_name} ===")
    print("eps\tFGSM-U\tFGSM-T\tPGD-U\tPGD-T")

    for eps in eps_list:
        eps_step = eps / 10

        fgsm_u = attack_success_rate_untargeted_fgsm(model, loader, eps, max_samples=max_samples)
        fgsm_t = attack_success_rate_targeted_fgsm(model, loader, eps, max_samples=max_samples)
        pgd_u = attack_success_rate_untargeted_pgd(model, loader, eps, eps_step, k, max_samples=max_samples)
        pgd_t = attack_success_rate_targeted_pgd(model, loader, eps, eps_step, k, max_samples=max_samples)

        print(f"{eps:.2f}\t{fgsm_u:.2f}%\t{fgsm_t:.2f}%\t{pgd_u:.2f}%\t{pgd_t:.2f}%")


# ---------------- MNIST ----------------
mnist_train_loader, mnist_test_loader = get_mnist_loaders()

mnist_model = SimpleCNN().to(device)
mnist_criterion = nn.CrossEntropyLoss()
mnist_optimizer = optim.Adam(mnist_model.parameters(), lr=0.001)
mnist_epochs = 3
mnist_ckpt = "mnist_model.pth"

# 저장된 모델이 있으면 다시 학습하지 않고 불러온다.
if os.path.exists(mnist_ckpt):
    mnist_model.load_state_dict(torch.load(mnist_ckpt, map_location=device))
    mnist_acc = evaluate(mnist_model, mnist_test_loader, device)
    print(f"\nLoaded MNIST model, Test Accuracy: {mnist_acc:.2f}%")
else:
    print("\nTraining MNIST model...")
    for epoch in range(mnist_epochs):
        train(mnist_model, mnist_train_loader, mnist_criterion, mnist_optimizer, device)
        acc = evaluate(mnist_model, mnist_test_loader, device)
        print(f"MNIST Epoch {epoch + 1}/{mnist_epochs}, Test Accuracy: {acc:.2f}%")

    # 다음 실행 때 재사용할 수 있도록 저장한다.
    torch.save(mnist_model.state_dict(), mnist_ckpt)

eps_list = [0.05, 0.1, 0.2, 0.3]
run_attack_table(mnist_model, mnist_test_loader, "MNIST", eps_list, k=10, max_samples=100)

vis_eps = 0.3
vis_eps_step = vis_eps / 10
vis_k = 10

save_visualizations(
    mnist_model, mnist_test_loader,
    fgsm_untargeted, "fgsm_untargeted", "mnist_fgsm_untargeted",
    vis_eps, num_samples=5
)

save_visualizations(
    mnist_model, mnist_test_loader,
    fgsm_targeted, "fgsm_targeted", "mnist_fgsm_targeted",
    vis_eps, num_samples=5
)

save_visualizations(
    mnist_model, mnist_test_loader,
    pgd_untargeted, "pgd_untargeted", "mnist_pgd_untargeted",
    vis_eps, num_samples=5, eps_step=vis_eps_step, k=vis_k
)

save_visualizations(
    mnist_model, mnist_test_loader,
    pgd_targeted, "pgd_targeted", "mnist_pgd_targeted",
    vis_eps, num_samples=5, eps_step=vis_eps_step, k=vis_k
)

print("Saved MNIST visualization images to results/")


# ---------------- CIFAR-10 ----------------
cifar_train_loader, cifar_test_loader = get_cifar10_loaders()

cifar_model = get_cifar10_model().to(device)
cifar_criterion = nn.CrossEntropyLoss()
cifar_optimizer = optim.Adam(cifar_model.parameters(), lr=0.001)
cifar_epochs = 10
cifar_ckpt = "cifar10_model.pth"

# 저장된 CIFAR-10 모델이 있으면 불러오고, 없으면 학습한다.
if os.path.exists(cifar_ckpt):
    cifar_model.load_state_dict(torch.load(cifar_ckpt, map_location=device))
    cifar_acc = evaluate(cifar_model, cifar_test_loader, device)
    print(f"\nLoaded CIFAR-10 model, Test Accuracy: {cifar_acc:.2f}%")
else:
    print("\nTraining CIFAR-10 model...")
    for epoch in range(cifar_epochs):
        train(cifar_model, cifar_train_loader, cifar_criterion, cifar_optimizer, device)
        acc = evaluate(cifar_model, cifar_test_loader, device)
        print(f"CIFAR-10 Epoch {epoch + 1}/{cifar_epochs}, Test Accuracy: {acc:.2f}%")

    torch.save(cifar_model.state_dict(), cifar_ckpt)

run_attack_table(cifar_model, cifar_test_loader, "CIFAR-10", eps_list, k=10, max_samples=100)

save_visualizations(
    cifar_model, cifar_test_loader,
    fgsm_untargeted, "fgsm_untargeted", "cifar10_fgsm_untargeted",
    vis_eps, num_samples=5
)

save_visualizations(
    cifar_model, cifar_test_loader,
    fgsm_targeted, "fgsm_targeted", "cifar10_fgsm_targeted",
    vis_eps, num_samples=5
)

save_visualizations(
    cifar_model, cifar_test_loader,
    pgd_untargeted, "pgd_untargeted", "cifar10_pgd_untargeted",
    vis_eps, num_samples=5, eps_step=vis_eps_step, k=vis_k
)

save_visualizations(
    cifar_model, cifar_test_loader,
    pgd_targeted, "pgd_targeted", "cifar10_pgd_targeted",
    vis_eps, num_samples=5, eps_step=vis_eps_step, k=vis_k
)

print("Saved CIFAR-10 visualization images to results/")