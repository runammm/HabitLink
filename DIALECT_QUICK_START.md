# 방언 분석 빠른 시작 가이드 🚀

Issue #10의 방언 탐지 모듈이 성공적으로 구현되었습니다!

## ✅ 완료된 작업

1. ✅ **Jupyter Notebook 생성** (`notebooks/dialect_model_training.ipynb`)
   - 데이터 로드, 전처리, 모델 학습, 평가까지 전체 파이프라인 포함
   
2. ✅ **DialectAnalyzer 클래스 구현** (`src/dialect_analyzer.py`)
   - 모델 로드 및 추론 기능
   - 한국어 방언명 변환 기능
   
3. ✅ **HabitLink 통합** (`src/session.py`)
   - 분석 모듈 선택 메뉴에 추가
   - 세션 종료 시 자동 분석
   - PDF 리포트에 결과 포함
   
4. ✅ **의존성 업데이트** (`requirements.txt`)
   - transformers, torch, datasets 등 추가 (주석 처리됨)
   
5. ✅ **사용자 가이드 작성** (`DIALECT_GUIDE.md`)
   - 데이터 준비부터 모델 학습까지 상세 설명
   
6. ✅ **README 업데이트**
   - 방언 분석 기능 추가

---

## 🎯 지금 당장 해야 할 일

### 1️⃣ 데이터 준비

SSD에 저장된 방언 데이터를 다음과 같이 정리:

```
/your/ssd/path/dialect_dataset/
├── standard/          # 표준어
│   ├── audio_001.wav
│   └── ...
├── gyeongsang/        # 경상도
├── jeolla/            # 전라도
└── chungcheong/       # 충청도
```

**필요한 것:**
- ✅ 각 방언당 최소 100개 이상 오디오 파일
- ✅ WAV 또는 MP3 형식
- ✅ 2~30초 길이 권장

### 2️⃣ 필수 라이브러리 설치

```bash
pip install transformers torch datasets accelerate evaluate scikit-learn
```

### 3️⃣ 모델 학습

1. Jupyter Notebook 실행:
   ```bash
   cd notebooks
   jupyter notebook dialect_model_training.ipynb
   ```

2. Step 2 셀에서 데이터 경로 수정:
   ```python
   DATASET_PATH = "/your/ssd/path/dialect_dataset"  # 여기를 수정!
   ```

3. 모든 셀 순서대로 실행 (Shift+Enter)

4. 학습 완료까지 대기 (GPU: 1-3시간, CPU: 권장하지 않음)

### 4️⃣ 사용하기

```bash
python main.py
```

분석 모듈 선택 시 **"7. 방언 분석"** 선택!

---

## 📊 예상 출력

```
--- 🗣️ 방언 분석 요약 ---

방언 확률 분포:
  경상도 방언              [████████████████████████░░] 85.23%
  표준어 (서울/수도권)     [████░░░░░░░░░░░░░░░░░░░░░░] 10.45%
  전라도 방언              [██░░░░░░░░░░░░░░░░░░░░░░░░] 3.12%

✨ 주요 방언: 경상도 방언 (신뢰도: 85.23%)
```

---

## 🐛 문제 해결

### 모델을 찾을 수 없음
```
⚠️ 방언 분석 모델을 찾을 수 없습니다
```
→ 노트북으로 모델을 먼저 학습시켜야 합니다!

### GPU 없음
→ CPU로도 추론 가능하지만, 학습은 GPU 필수

### 데이터 부족
→ 각 방언당 최소 100개 이상 필요

---

## 📚 더 자세한 정보

- **상세 가이드**: `DIALECT_GUIDE.md`
- **Issue #10**: https://github.com/runammm/HabitLink/issues/10
- **노트북**: `notebooks/dialect_model_training.ipynb`
- **소스 코드**: `src/dialect_analyzer.py`

---

## 🎉 완료!

Issue #10의 방언 탐지 모듈이 성공적으로 구현되었습니다!

**다음 단계:**
1. SSD에서 데이터 준비
2. 노트북으로 모델 학습
3. HabitLink에서 방언 분석 사용!

질문이 있으시면 GitHub Issues에 올려주세요! 🚀

