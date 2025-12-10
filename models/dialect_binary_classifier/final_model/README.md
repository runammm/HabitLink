# ✅ 방언 모델 배치 완료

## 📦 모델 파일 현황

이 폴더에는 **Wav2Vec2 기반 표준어/비표준어 이진 분류 모델**이 배치되어 있습니다.

```
final_model/
├── config.json                  ✅ 모델 설정
├── model.safetensors           ✅ 모델 가중치 (1.2 GB)
├── preprocessor_config.json    ✅ 전처리 설정
├── vocab.json                  ✅ 어휘 사전
├── tokenizer_config.json       ✅ 토크나이저 설정
├── special_tokens_map.json     ✅ 특수 토큰
├── added_tokens.json           ✅ 추가 토큰
└── training_args.bin           ✅ 학습 파라미터
```

---

## 🎯 모델 정보

| 항목 | 내용 |
|------|------|
| **베이스 모델** | Wav2Vec2-large |
| **태스크** | Audio Classification (Binary) |
| **클래스** | 0: 표준어, 1: 비표준어 |
| **입력** | 16kHz 오디오 |
| **출력** | 표준어/비표준어 확률 |
| **모델 크기** | ~1.2 GB |

---

## 🧪 테스트 방법

### 1. 모델 로드 테스트

프로젝트 루트에서 다음 명령어 실행:

```bash
python test_dialect_model.py
```

**예상 출력:**

```
🧪 방언 검출 시스템 테스트
============================================================

📦 의존성 확인
============================================================
✅ Hugging Face Transformers: 4.x.x
✅ PyTorch: 2.x.x
✅ NumPy: 1.x.x
✅ SoundFile: 0.x.x

📚 방언 어휘 사전 확인
============================================================
✅ 방언 어휘 사전: 150개 단어

지역별 분포:
  • 경상도: 45개
  • 전라도: 38개
  • 충청도: 28개
  • 강원도: 22개
  • 제주도: 17개

🤖 모델 파일 확인
============================================================
✅ 모델 설정: config.json
✅ 전처리 설정: preprocessor_config.json
✅ 모델 가중치: model.safetensors (1204.5 MB)

🔄 모델 로딩 테스트
============================================================
방언 분석 모델을 로드하는 중...
✅ 방언 분석 모델 로드 완료!

모델 로드 상태: ✅ 성공
방언 어휘 사전: 150개 단어
억양 분석 가능: ✅
어휘 분석 가능: ✅

✅ 방언 검출 시스템이 정상적으로 작동합니다!

🎉 모든 테스트 통과! 방언 검출 시스템을 사용할 수 있습니다.
```

### 2. 프로그램 실행

```bash
python main.py
```

분석 모듈 선택 시 `7`번 (방언 분석)을 선택하세요.

---

## 📊 사용 방법

### Python 코드에서 직접 사용

```python
from src.dialect_analyzer import DialectAnalyzer

# 모델 초기화
analyzer = DialectAnalyzer('models/dialect_binary_classifier/final_model')

# 오디오 파일 분석
result = analyzer.get_classification('audio.wav')

print(result)
# {
#   'is_standard': False,
#   'confidence': 0.7769,
#   'probabilities': {
#     'standard': 0.2231,
#     'non_standard': 0.7769
#   }
# }

# 판정
if result['is_standard']:
    print("✅ 표준어입니다")
else:
    print("⚠️ 비표준어(방언)가 감지되었습니다")
```

### 실시간 분석 (10초 단위)

```python
import numpy as np

# 10초 오디오 세그먼트 (16kHz)
audio_segment = np.random.randn(16000 * 10)  # 실제로는 녹음된 오디오
text = "오늘 날씨가 참 좋네, 가가리도 보이고"  # STT 결과

# 실시간 분석 (억양 + 어휘)
result = analyzer.analyze_segment_realtime(
    audio_array=audio_segment,
    sample_rate=16000,
    text=text,
    timestamp=0.0
)

if result['feedback_trigger']:
    print("🔔 방언 감지!")
    
    if result['vocabulary_analysis']:
        for word in result['vocabulary_analysis']['detected_words']:
            print(f"  • '{word['word']}' ({word['region']}도)")
    
    if result['acoustic_analysis']:
        print(f"  • 억양 분석: {result['combined_verdict']}")
```

---

## ⚙️ 모델 설정

### CPU vs GPU

기본적으로 CPU 모드로 실행됩니다. GPU를 사용하려면:

```python
# dialect_analyzer.py의 __init__ 메서드에서
self.classifier = pipeline(
    "audio-classification",
    model=model_path,
    device=0  # -1: CPU, 0: GPU
)
```

### 메모리 최적화

메모리가 부족한 경우:

1. **배치 크기 줄이기**: 한 번에 처리하는 오디오 수 감소
2. **작은 모델 사용**: Wav2Vec2-base로 교체
3. **다른 프로그램 종료**: 최소 4GB RAM 권장

---

## 🔧 문제 해결

### "모델을 로드할 수 없습니다"

**원인**: transformers 또는 torch 미설치

**해결방법**:
```bash
pip install transformers torch
```

### "CUDA out of memory"

**원인**: GPU 메모리 부족

**해결방법**: CPU 모드 사용 (`device=-1`)

### 정확도가 낮아요

**원인**: 학습 데이터 부족 또는 품질 문제

**해결방법**:
1. 더 많은 데이터로 재학습 (각 클래스당 200개 이상 권장)
2. 데이터 품질 확인 (배경 소음, 음질)
3. Data augmentation 적용

---

## 📚 추가 정보

- **학습 방법**: `DIALECT_BINARY_GUIDE.md` 참고
- **통합 가이드**: `DIALECT_INTEGRATION_GUIDE.md` 참고
- **빠른 시작**: `DIALECT_QUICKSTART.md` 참고

---

## 🎉 완료!

모델이 제대로 배치되어 있으며 사용할 준비가 되었습니다!

테스트 실행:
```bash
python test_dialect_model.py
```

프로그램 실행:
```bash
python main.py
```
