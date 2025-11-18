# 표준어 vs 비표준어 이진 분류 가이드

## 📖 개요

HabitLink의 방언 분석 모듈은 **이진 분류(Binary Classification)**로 작동합니다:
- ✅ **표준어** (서울/수도권)
- ⚠️ **비표준어** (지역 방언)

간단하고 실용적인 판별을 위한 AI 모델입니다!

---

## 🎯 변경 사항 (Issue #10)

### 기존 계획 (다중 분류)
- ❌ 경상도, 전라도, 충청도 등 세부 구분
- ❌ 복잡한 다중 클래스 모델

### 현재 구현 (이진 분류)
- ✅ 표준어 vs 비표준어만 구분
- ✅ 간단하고 handy한 판별
- ✅ Google Colab에서 바로 학습 가능

---

## 🚀 빠른 시작 (3단계)

### 1️⃣ 데이터 준비

```
/your/dataset/
├── standard/      # 표준어 → 0
│   ├── audio_001.wav
│   └── ...
├── gyeongsang/    # 경상도 → 1 (비표준어)
│   ├── audio_001.wav
│   └── ...
└── gangwon/       # 강원도 → 1 (비표준어)
    ├── audio_001.wav
    └── ...
```

**💡 포인트**: 
- `standard` 폴더만 0으로 레이블됨
- 나머지 모든 폴더는 자동으로 1(비표준어)로 통합

### 2️⃣ Google Colab에서 학습

1. **Colab 열기**: https://colab.research.google.com
2. **GPU 설정**: 런타임 > 런타임 유형 변경 > T4 GPU
3. **노트북 업로드**: `notebooks/dialect_model_training.ipynb`
4. **Drive 마운트**: 첫 번째 셀 실행 시 자동
5. **데이터 경로 수정**:
   ```python
   DATASET_PATH = "/content/drive/MyDrive/dialect_dataset"
   ```
6. **모든 셀 실행**: Ctrl+F9

**⏱️ 학습 시간**: 
- T4 GPU: 30분 ~ 1시간
- 데이터 100개씩: 약 40분

### 3️⃣ 모델 다운로드 & 사용

**Colab에서 다운로드:**
```python
from google.colab import files
import shutil

shutil.make_archive('dialect_model', 'zip', final_model_path)
files.download('dialect_model.zip')
```

**로컬에 배치:**
```
HabitLink/
└── models/
    └── dialect_binary_classifier/
        └── final_model/
            ├── config.json
            ├── pytorch_model.bin
            └── ...
```

**실행:**
```bash
python main.py
# 7. 방언 분석 선택
```

---

## 📊 결과 예시

```
--- 🗣️ 방언 분석 요약 (표준어 vs 비표준어) ---

📊 이진 분류 분석 중...

📊 확률 분포:
  표준어      [████████████████████████████████████████████░░░░░] 85.23%
  비표준어    [███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 14.77%

✨ 판정: ✅ 표준어 (신뢰도: 85.23%)
   → 표준어 발음을 사용하고 있습니다.
```

또는:

```
📊 확률 분포:
  표준어      [███████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 22.31%
  비표준어    [███████████████████████████████████████████░░░░░░░] 77.69%

✨ 판정: ⚠️ 비표준어 (신뢰도: 77.69%)
   → 방언 특성이 감지되었습니다.
```

---

## 💻 프로그래밍 사용법

```python
from src.dialect_analyzer import DialectAnalyzer

# 모델 로드
analyzer = DialectAnalyzer("models/dialect_binary_classifier/final_model")

# 분석
result = analyzer.get_classification("audio.wav")

print(result)
# {
#   'is_standard': True,
#   'confidence': 0.8523,
#   'probabilities': {
#     'standard': 0.8523,
#     'non_standard': 0.1477
#   }
# }

# 간단한 판정
if result['is_standard']:
    print("✅ 표준어입니다!")
else:
    print("⚠️ 비표준어(방언)가 감지되었습니다.")
```

---

## 🔧 Google Colab 사용 팁

### Drive 마운트
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 데이터 업로드 방법

**방법 1: Google Drive 업로드**
1. Drive 웹에서 `MyDrive/dialect_dataset/` 폴더 생성
2. 표준어, 방언 폴더 업로드
3. 노트북에서 경로 지정

**방법 2: Colab에서 직접 업로드**
```python
from google.colab import files
uploaded = files.upload()  # 파일 선택
```

### GPU 확인
```python
import torch
print(torch.cuda.is_available())  # True여야 함
print(torch.cuda.get_device_name(0))  # Tesla T4
```

### 학습 중간 저장
Colab은 12시간 제한이 있으므로:
- 체크포인트가 자동으로 Drive에 저장됨
- 중단되면 마지막 체크포인트부터 재개 가능

---

## 📈 성능 개선 팁

### 1. 데이터 균형 맞추기
```
표준어: 150개
비표준어 (경상도 + 강원도): 150개
→ 50:50 비율 유지
```

### 2. 데이터 증강 (선택사항)
```python
# 속도 변경
librosa.effects.time_stretch(audio, rate=0.9)  # 느리게
librosa.effects.time_stretch(audio, rate=1.1)  # 빠르게

# 피치 변경
librosa.effects.pitch_shift(audio, sr, n_steps=2)
```

### 3. Hyperparameter 튜닝
```python
BATCH_SIZE = 16  # → 32 (메모리 충분하면)
NUM_EPOCHS = 10   # → 15 (더 학습)
LEARNING_RATE = 3e-5  # → 2e-5 (더 세밀하게)
```

---

## 🐛 문제 해결

### Colab에서 모델이 안 보임
```python
# 경로 확인
!ls /content/drive/MyDrive/models/dialect_binary_classifier/final_model/
```

### CUDA out of memory
```python
BATCH_SIZE = 8  # → 4로 줄이기
gradient_accumulation_steps = 4  # 추가
```

### 데이터 경로 오류
```python
# 절대 경로 확인
import os
print(os.listdir("/content/drive/MyDrive/"))
```

### 모델 다운로드 안 됨
```python
# Drive에서 직접 다운로드
# 또는 zip으로 압축 후 다운로드
import shutil
shutil.make_archive('/content/dialect_model', 'zip', 
                    '/content/drive/MyDrive/models/dialect_binary_classifier/final_model')
```

---

## ❓ FAQ

**Q: 왜 이진 분류인가요?**  
A: 실용성! 대부분의 경우 "표준어인지 아닌지"만 알면 충분합니다. 세부 방언 구분은 복잡하고 데이터도 많이 필요합니다.

**Q: Colab 무료로 충분한가요?**  
A: 네! T4 GPU면 충분합니다. 학습 시간 1시간 이내.

**Q: 데이터가 50개씩밖에 없어요.**  
A: 최소 권장은 100개씩이지만, 50개로도 시도해볼 수 있습니다. Data augmentation으로 늘리세요.

**Q: 로컬에서 학습하면 안 되나요?**  
A: GPU가 있다면 가능하지만, Colab이 더 안정적입니다.

**Q: 모델 크기는?**  
A: 약 1.2GB (Wav2Vec2-large 기반)

**Q: 정확도가 낮아요 (60%).**  
A: 
1. 데이터를 더 모으세요 (각 200개 이상)
2. 오디오 품질을 확인하세요 (배경 소음 제거)
3. 에포크를 늘려보세요 (15~20)

---

## 📚 참고 자료

- **Wav2Vec2 논문**: https://arxiv.org/abs/2006.11477
- **Hugging Face 모델**: https://huggingface.co/kresnik/wav2vec2-large-xlsr-korean
- **Google Colab**: https://colab.research.google.com

---

## ✅ 체크리스트

학습 전:
- [ ] 데이터 폴더 구조 확인 (`standard/`, 방언폴더들)
- [ ] 각 폴더당 최소 100개 오디오
- [ ] Google Drive에 업로드 완료
- [ ] Colab GPU 설정 (T4)

학습 후:
- [ ] 모델 파일 생성 확인 (`final_model/`)
- [ ] 테스트 정확도 >75%
- [ ] 모델 다운로드 완료
- [ ] HabitLink에 배치 완료

---

**🎉 준비 완료! 이제 Colab에서 학습을 시작하세요!**

궁금한 점이 있으면 GitHub Issues에 올려주세요!

