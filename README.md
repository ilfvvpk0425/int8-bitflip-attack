# Bit-Flip Attack on an 8-bit Neural Network

Rowhammer 같은 하드웨어 결함 주입으로 메모리의 비트가 뒤집히면 신경망이 어떻게 무너지는지 재현하는 실험.
CIFAR-10으로 학습한 ResNet-20의 가중치를 int8로 양자화한 뒤, **무작위 비트 플립**과 **골라서 뒤집는 비트 플립**(Rakin et al., *Bit-Flip Attack*, ICCV 2019의 Progressive Bit Search를 단순화)을 비교한다.

## 실행 (RTX 4060 Ti 기준)

```bash
python -m venv venv && venv\Scripts\activate        # macOS/Linux: source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install matplotlib
python train.py      # CIFAR-10 자동 다운로드 후 40 epoch 학습, 약 10~15분, 정확도 90% 안팎 목표
python attack.py     # 무작위/표적 비트 플립 실험, results.json과 bitflip.png 생성
```

## 기대 결과 (실험 전 가설)

- 무작위 비트 플립은 수백~수천 개를 뒤집어도 정확도가 크게 떨어지지 않는다.
- 골라서 뒤집으면 수십 개 안쪽으로 정확도가 무작위 추측 수준(10%)까지 떨어진다.
- 뒤집히는 비트는 대부분 부호 비트(bit 7)일 것이다. 작은 가중치의 부호 비트를 뒤집으면 값이 -128 근처로 크게 바뀌기 때문이다.

실제 숫자는 `results.json`, 어느 층의 어느 비트가 뒤집혔는지는 `flips` 기록에서 확인한다.

## 이후 확장 아이디어 (시간 날 때)

1. 방어: 가중치 범위 제한(clipping), 이진화, 체크섬으로 탐지했을 때 필요한 비트 수가 얼마나 늘어나는지 비교
2. 층별 취약성: 어느 층이 가장 약한지 분석
3. 공격자가 가진 데이터 양(128장 → 16장)에 따른 공격 성공률 변화

## 보고서 한 쪽 구성

목표 → 위협 모델(공격자는 메모리 비트를 뒤집을 수 있고, 테스트 데이터 일부를 가짐) → 방법 → 결과 그래프 → 무엇을 배웠나 → 방어 아이디어
