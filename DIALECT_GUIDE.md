# 방언 분석 모듈 가이드

## 📖 개요

HabitLink의 방언 분석 모듈은 AI 기반 음성 분류를 통해 한국어 방언을 자동으로 탐지합니다. Wav2Vec2 사전 학습 모델을 파인튜닝하여 경상도, 전라도, 충청도 등 다양한 방언을 구분할 수 있습니다.

---

## 🎯 기능

- **실시간 방언 탐지**: 세션 종료 시 전체 음성을 분석하여 방언 확률 분포 제공
- **높은 정확도**: 사전 학습된 Wav2Vec2 모델 기반
- **다양한 방언 지원**: 표준어, 경상도, 전라도, 충청도, 강원도, 제주도 등
- **신뢰도 표시**: 각 방언에 대한 확률 점수 제공
- **PDF 리포트 통합**: 분석 결과가 자동으로 리포트에 포함

---

## 📋 사전 준비

### 1. 필요한 라이브러리 설치

방언 분석 모듈을 사용하려면 추가 라이브러리가 필요합니다:

```bash
pip install transformers torch datasets accelerate evaluate scikit-learn
```

또는 `requirements.txt`에서 해당 줄의 주석을 제거하고:

```bash
pip install -r requirements.txt
```

### 2. GPU 설정 (선택사항)

- **GPU 사용 권장**: 모델 학습 시 GPU를 사용하면 10~100배 빠릅니다
- **CPU만으로도 가능**: 추론은 CPU로도 충분히 빠릅니다 (1-2초)
- **GPU 확인**: 
  ```bash
  python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
  ```

---

## 🔧 모델 학습 단계

### Step 1: 데이터 준비

SSD에 저장된 방언 오디오 데이터를 다음과 같은 구조로 정리하세요:

```
/your/ssd/path/dialect_dataset/
├── standard/          # 표준어 (서울/수도권)
│   ├── audio_001.wav
│   ├── audio_002.wav
│   └── ...
├── gyeongsang/        # 경상도 방언
│   ├── audio_001.wav
│   ├── audio_002.wav
│   └── ...
├── jeolla/            # 전라도 방언
│   ├── audio_001.wav
│   └── ...
├── chungcheong/       # 충청도 방언
│   ├── audio_001.wav
│   └── ...
└── gangwon/           # 강원도 방언 (선택사항)
    ├── audio_001.wav
    └── ...
```

**데이터 요구사항:**
- ✅ 형식: WAV 또는 MP3
- ✅ 샘플링 레이트: 16kHz 권장 (자동 리샘플링됨)
- ✅ 길이: 2~30초 권장
- ✅ 각 방언당: **최소 100개 이상 샘플** 권장
  - 50개 미만: 학습 어려움
  - 100-300개: 기본 성능
  - 300개 이상: 좋은 성능
  - 1000개 이상: 최고 성능

**데이터 수집 팁:**
1. **공개 데이터셋**:
   - AI Hub (한국어 방언 발화 데이터)
   - YouTube 영상에서 추출
   - 방송 인터뷰 오디오

2. **자체 수집**:
   - 가족/지인의 방언 녹음
   - 각 방언의 다양한 화자 포함
   - 남성/여성 균형 있게

3. **데이터 품질**:
   - 배경 소음 최소화
   - 명확한 발음
   - 자연스러운 대화

### Step 2: Jupyter Notebook 실행

1. **Jupyter Notebook 실행**:
   ```bash
   cd notebooks
   jupyter notebook dialect_model_training.ipynb
   ```

2. **설정 수정 (Step 2 셀)**:
   ```python
   # 데이터셋 경로를 본인의 SSD 경로로 수정
   DATASET_PATH = "/Volumes/MySSD/dialect_dataset"  # 여기를 수정!
   
   # 학습 설정 (필요시 조정)
   BATCH_SIZE = 8  # GPU 메모리에 따라 조정
   NUM_EPOCHS = 10
   LEARNING_RATE = 3e-5
   ```

3. **셀 순서대로 실행**:
   - Step 1: 라이브러리 임포트 ✅
   - Step 2: 설정 확인 ✅
   - Step 3-7: 데이터 로드 및 전처리 ✅
   - Step 8: **학습 시작** (시간이 오래 걸림!) ⏰
   - Step 9-11: 평가 및 저장 ✅

4. **학습 시간**:
   - GPU (RTX 3090): 1-3시간
   - GPU (GTX 1060): 3-6시간
   - CPU: 수 시간 ~ 수일 (권장하지 않음)

### Step 3: 모델 확인

학습이 완료되면 다음 경로에 모델이 저장됩니다:

```
models/dialect_classifier/final_model/
├── config.json
├── preprocessor_config.json
├── pytorch_model.bin
└── ...
```

---

## 🚀 사용 방법

### 1. HabitLink 애플리케이션에서 사용

1. **메인 프로그램 실행**:
   ```bash
   python main.py
   ```

2. **분석 모듈 선택**:
   ```
   사용할 분석 모듈을 선택하세요:
   1. 특정 반복 단어 검출
   2. 비속어 검출
   3. 발화 속도 분석
   4. 문법 분석
   5. 맥락 분석
   6. 말더듬 분석
   7. 방언 분석 (AI 모델)  ← 선택!
   
   선택: 7
   ```

3. **음성 녹음 및 분석**:
   - 평소처럼 말하기
   - 세션 종료 시 자동으로 방언 분석 실행

4. **결과 확인**:
   ```
   --- 🗣️ 방언 분석 요약 ---
   
   방언 확률 분포:
     경상도 방언              [████████████████████████░░] 85.23%
     표준어 (서울/수도권)     [████░░░░░░░░░░░░░░░░░░░░░░] 10.45%
     전라도 방언              [██░░░░░░░░░░░░░░░░░░░░░░░░] 3.12%
   
   ✨ 주요 방언: 경상도 방언 (신뢰도: 85.23%)
   ```

### 2. 프로그래밍 방식으로 사용

```python
from src.dialect_analyzer import DialectAnalyzer

# 모델 로드
analyzer = DialectAnalyzer("models/dialect_classifier/final_model")

# 단일 오디오 파일 분석
result = analyzer.analyze("path/to/audio.wav")
print(result)
# {'gyeongsang': 0.85, 'standard': 0.10, 'jeolla': 0.03, ...}

# 가장 높은 확률의 방언
top = analyzer.get_top_dialect("path/to/audio.wav")
print(f"{top['dialect']}: {top['confidence']:.2%}")
# gyeongsang: 85.00%

# 한국어 이름으로 변환
dialect_kr = analyzer.get_dialect_name_korean(top['dialect'])
print(dialect_kr)
# 경상도 방언
```

---

## 🎨 PDF 리포트

방언 분석 결과는 자동으로 PDF 리포트에 포함됩니다:

```
📄 HabitLink 세션 리포트
...

[방언 분석]
- 주요 방언: 경상도 방언
- 신뢰도: 85.23%
- 확률 분포:
  • 경상도 방언: 85.23%
  • 표준어: 10.45%
  • 전라도 방언: 3.12%
  • 충청도 방언: 1.20%
```

---

## ⚙️ 고급 설정

### 모델 개선

더 나은 성능을 원한다면:

1. **더 많은 데이터 수집**
   - 각 방언당 500개 이상 권장

2. **학습 설정 조정** (notebook Step 2):
   ```python
   BATCH_SIZE = 16  # GPU 메모리가 충분하면 증가
   NUM_EPOCHS = 15  # 더 많은 에포크
   LEARNING_RATE = 2e-5  # 학습률 조정
   ```

3. **Data Augmentation** (고급):
   - 배경 소음 추가
   - 속도 변경
   - 피치 변경

### GPU 메모리 부족 시

`CUDA out of memory` 에러가 발생하면:

```python
BATCH_SIZE = 4  # 배치 크기 감소
gradient_accumulation_steps = 4  # Gradient accumulation 증가
```

### 다른 베이스 모델 사용

```python
# 옵션 1: 한국어 특화 (기본, 추천)
BASE_MODEL = "kresnik/wav2vec2-large-xlsr-korean"

# 옵션 2: 다국어 (한국어 데이터 적을 때)
BASE_MODEL = "facebook/wav2vec2-large-xlsr-53"

# 옵션 3: 더 큰 모델 (성능 향상, GPU 필요)
BASE_MODEL = "kresnik/wav2vec2-large-xlsr-korean-v2"
```

---

## 🐛 문제 해결

### 모델이 로드되지 않음

```
⚠️ 방언 분석 모델을 찾을 수 없습니다
```

**해결책**: 
1. 모델 경로 확인: `models/dialect_classifier/final_model/`
2. 노트북 재실행: `notebooks/dialect_model_training.ipynb`

### 라이브러리 오류

```
⚠️ Transformers 라이브러리가 설치되지 않았습니다
```

**해결책**:
```bash
pip install transformers torch datasets
```

### GPU 인식 안 됨

```python
import torch
print(torch.cuda.is_available())  # False
```

**해결책**:
1. CUDA 드라이버 설치 확인
2. PyTorch GPU 버전 설치:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### 학습이 너무 느림

**해결책**:
- GPU 사용 (필수)
- 배치 크기 증가
- Mixed precision 활성화됨 확인 (`fp16=True`)

---

## 📊 성능 평가

### 좋은 모델 기준

- **Accuracy > 80%**: 좋음
- **Accuracy > 90%**: 매우 좋음
- **Accuracy > 95%**: 우수

### 성능이 낮을 때

1. **데이터 부족**: 각 방언당 최소 100개 이상
2. **데이터 불균형**: 모든 방언의 샘플 수를 비슷하게
3. **데이터 품질**: 소음이 적고 명확한 녹음
4. **학습 부족**: 에포크 수 증가 (15-20)

---

## 💡 팁

### 데이터 수집
- **다양성 확보**: 다양한 화자, 나이, 성별
- **자연스러운 발화**: 스크립트 읽기보다 자연스러운 대화
- **적절한 길이**: 너무 짧거나 길지 않게 (5-15초 권장)

### 모델 학습
- **Early Stopping**: 과적합 방지
- **Validation Set**: 반드시 분리
- **Cross Validation**: 데이터가 적을 때 유용

### 실전 사용
- **충분한 음성 길이**: 최소 5초 이상 녹음
- **명확한 발음**: 배경 소음 최소화
- **자연스러운 발화**: 방언 특징이 잘 드러나게

---

## 📚 참고 자료

### 논문
- Wav2Vec 2.0: [arxiv.org/abs/2006.11477](https://arxiv.org/abs/2006.11477)
- XLSR-53: [arxiv.org/abs/2111.09296](https://arxiv.org/abs/2111.09296)

### 한국어 음성 데이터셋
- AI Hub: [aihub.or.kr](https://aihub.or.kr)
- 한국어 방언 발화 데이터
- 자유대화 음성 데이터

### Hugging Face 모델
- [kresnik/wav2vec2-large-xlsr-korean](https://huggingface.co/kresnik/wav2vec2-large-xlsr-korean)
- [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53)

---

## 🤝 기여

더 나은 방언 분석을 위해:
1. 데이터 공유
2. 모델 개선 아이디어
3. 버그 리포트

GitHub Issue: [github.com/runammm/HabitLink/issues](https://github.com/runammm/HabitLink/issues)

---

## ❓ FAQ

**Q: 모델 학습에 얼마나 걸리나요?**  
A: GPU 사용 시 1-3시간, CPU는 권장하지 않습니다 (수일 소요).

**Q: 데이터가 얼마나 필요한가요?**  
A: 각 방언당 최소 100개, 권장 300개 이상.

**Q: 제주도 방언도 지원하나요?**  
A: 학습 데이터만 있으면 가능합니다!

**Q: 실시간 분석이 가능한가요?**  
A: 현재는 세션 종료 시 분석하지만, 실시간 분석도 가능합니다 (향후 업데이트).

**Q: 모델 정확도를 높이려면?**  
A: 더 많은 양질의 데이터, 더 긴 학습 시간, 데이터 증강 기법 적용.

---

## 📝 체크리스트

모델 학습 전:
- [ ] GPU 사용 가능 확인
- [ ] 라이브러리 설치 완료
- [ ] 데이터 폴더 구조 올바르게 정리
- [ ] 각 방언당 100개 이상 샘플 확보
- [ ] 충분한 저장 공간 확보 (10GB+)

모델 학습 후:
- [ ] 모델 파일 생성 확인
- [ ] 테스트 정확도 확인 (>80%)
- [ ] 샘플 추론 테스트 성공
- [ ] HabitLink에서 모듈 활성화 가능

---

**Happy Training! 🎉**

더 나은 한국어 방언 인식을 위해 함께 노력해요!

