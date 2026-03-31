# Adversarial Attack Assignment

이 프로젝트는 MNIST와 CIFAR-10 데이터셋에 대해 adversarial attack을 구현하고 성능을 비교한 코드이다.

구현한 공격은 다음과 같다.

- Untargeted FGSM
- Targeted FGSM
- Untargeted PGD
- Targeted PGD

## Environment

- Python 3.11
- PyTorch
- torchvision
- matplotlib
- numpy

필요한 패키지는 `requirements.txt`에 정리되어 있다.

## Files

- `test.py` : 전체 실험을 실행하는 메인 코드
- `requirements.txt` : 실행에 필요한 패키지 목록
- `results/` : adversarial example 시각화 이미지 저장 폴더
- `mnist_model.pth` : 학습된 MNIST 모델
- `cifar10_model.pth` : 학습된 CIFAR-10 모델
- `report.pdf` : 실험 결과를 정리한 보고서

## How to Run

아래 명령어를 실행하면 전체 실험이 수행된다.

```bash
python test.py
```

## What the code does

test.py는 다음 작업을 순서대로 수행한다.

1. MNIST 모델을 학습하거나 저장된 모델을 불러온다.
2. MNIST clean accuracy를 측정한다.
3. MNIST에 대해 FGSM / PGD 공격 성공률을 계산한다.
4. MNIST adversarial example 시각화를 저장한다.
5. CIFAR-10 모델을 학습하거나 저장된 모델을 불러온다.
6. CIFAR-10 clean accuracy를 측정한다.
7. CIFAR-10에 대해 FGSM / PGD 공격 성공률을 계산한다.
8. CIFAR-10 adversarial example 시각화를 저장한다.

## Notes

MNIST는 직접 구현한 CNN을 사용했다.
CIFAR-10은 ResNet18 구조를 사용했다.
공격 성공률은 테스트 샘플 100개 기준으로 계산했다.
시각화 결과는 results/ 폴더에 저장된다.
이 프로젝트는 chat GPT를 참조했다. 