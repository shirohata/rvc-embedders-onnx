---
license: apache-2.0
language:
  - ko
library_name: transformers
pipeline_tag: automatic-speech-recognition
tags:
  - speech
  - audio
---

# hubert-base-korean

## Model Details

Hubert(Hidden-Unit BERT)는 Facebook에서 제안한 Speech Representation Learning 모델입니다.
Hubert는 기존의 음성 인식 모델과 달리, 음성 신호를 raw waveform에서 바로 학습하는 self-supervised learning 방식을 사용합니다.

이 연구는 구글의 TPU Research Cloud(TRC)를 통해 지원받은 Cloud TPU로 학습되었습니다.

### Model Description

<table>
   <tr>
      <td colspan="2"></td>
      <td>Base</td>
      <td>Large</td>
   </tr>
   <tr>
      <td rowspan="3">CNN Encoder</td>
      <td>strides</td>
      <td colspan="2">5, 2, 2, 2, 2, 2, 2</td>
   </tr>
   <tr>
      <td>kernel width</td>
      <td colspan="2">10, 3, 3, 3, 3, 2, 2</td>
   </tr>
   <tr>
      <td>channel</td>
      <td colspan="2">512</td>
   </tr>
   <tr>
      <td rowspan="4">Transformer Encoder</td>
      <td>Layer</td>
      <td>12</td>
      <td>24</td>
   </tr>
   <tr>
      <td>embedding dim</td>
      <td>768</td>
      <td>1024</td>
   </tr>
   <tr>
      <td>inner FFN dim</td>
      <td>3072</td>
      <td>4096</td>
   </tr>
   <tr>
      <td>attention heads</td>
      <td>8</td>
      <td>16</td>
   </tr>
   <tr>
      <td>Projection</td>
      <td>dim</td>
      <td>256</td>
      <td>768</td>
   </tr>
   <tr>
      <td colspan="2">Params</td>
      <td>95M</td>
      <td>317M </td>
   </tr>
</table>

## How to Get Started with the Model

### Pytorch

```py
import torch
from transformers import HubertModel

model = HubertModel.from_pretrained("team-lucid/hubert-base-korean")

wav = torch.ones(1, 16000)
outputs = model(wav)
print(f"Input:   {wav.shape}")  # [1, 16000]
print(f"Output:  {outputs.last_hidden_state.shape}")  # [1, 49, 768]
```

### JAX/Flax

```py
import jax.numpy as jnp
from transformers import FlaxAutoModel

model = FlaxAutoModel.from_pretrained("team-lucid/hubert-base-korean", trust_remote_code=True)

wav = jnp.ones((1, 16000))
outputs = model(wav)
print(f"Input:   {wav.shape}")  # [1, 16000]
print(f"Output:  {outputs.last_hidden_state.shape}")  # [1, 49, 768]
```

## Training Details

### Training Data

해당 모델은 과학기술정보통신부의 재원으로 한국지능정보사회진흥원의 지원을 받아
구축된 [자유대화 음성(일반남여)](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=109), [다화자 음성합성 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=542), [방송 콘텐츠 대화체 음성인식 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=463)
에서 약 4,000시간을 추출해 학습되었습니다.

### Training Procedure

[원 논문](https://arxiv.org/pdf/2106.07447.pdf)과 동일하게 MFCC 기반으로 Base 모델을 학습한 다음, 500 cluster로 k-means를 수행해 다시 Base와
Large 모델을 학습했습니다.

#### Training Hyperparameters

| Hyperparameter      | Base    |   Large |
|:--------------------|---------|--------:|
| Warmup Steps        | 32,000  |  32,000 |
| Learning Rates      | 5e-4    |  1.5e-3 |
| Batch Size          | 128     |     128 |
| Weight Decay        | 0.01    |    0.01 |
| Max Steps           | 400,000 | 400,000 |
| Learning Rate Decay | 0.1     |     0.1 |
| \\(Adam\beta_1\\)   | 0.9     |     0.9 |
| \\(Adam\beta_2\\)   | 0.99    |    0.99 |