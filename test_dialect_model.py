#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
방언 검출 시스템 테스트 스크립트

모델이 제대로 로드되는지, 어휘 사전이 제대로 동작하는지 확인합니다.
"""

import os
import sys

def test_dependencies():
    """의존성 패키지 확인"""
    print("\n" + "="*60)
    print("📦 의존성 확인")
    print("="*60)
    
    required = {
        'transformers': 'Hugging Face Transformers',
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'soundfile': 'SoundFile'
    }
    
    missing = []
    
    for package, name in required.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {name}: {version}")
        except ImportError:
            print(f"❌ {name}: 미설치")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  누락된 패키지: {', '.join(missing)}")
        print(f"다음 명령어로 설치하세요:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    return True


def test_vocabulary():
    """방언 어휘 사전 확인"""
    print("\n" + "="*60)
    print("📚 방언 어휘 사전 확인")
    print("="*60)
    
    vocab_path = 'resources/dialect_vocabulary.txt'
    
    if not os.path.exists(vocab_path):
        print(f"❌ 어휘 사전을 찾을 수 없습니다: {vocab_path}")
        return False
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    
    print(f"✅ 방언 어휘 사전: {len(lines)}개 단어")
    
    # 지역별 통계
    region_counts = {}
    for line in lines:
        parts = line.split('|')
        if len(parts) >= 2:
            region = parts[1].strip()
            region_counts[region] = region_counts.get(region, 0) + 1
    
    print("\n지역별 분포:")
    for region, count in sorted(region_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {region}도: {count}개")
    
    # 샘플 표시
    print("\n샘플 (처음 5개):")
    for i, line in enumerate(lines[:5], 1):
        parts = line.split('|')
        if len(parts) == 3:
            word, region, meaning = [p.strip() for p in parts]
            print(f"  {i}. '{word}' ({region}도) → {meaning}")
    
    return True


def test_model_files():
    """모델 파일 확인"""
    print("\n" + "="*60)
    print("🤖 모델 파일 확인")
    print("="*60)
    
    model_dir = 'models/dialect_binary_classifier/final_model'
    
    if not os.path.exists(model_dir):
        print(f"❌ 모델 폴더를 찾을 수 없습니다: {model_dir}")
        return False
    
    required_files = {
        'config.json': '모델 설정',
        'preprocessor_config.json': '전처리 설정'
    }
    
    model_weight_files = [
        'model.safetensors',
        'pytorch_model.bin',
        'model.pt'
    ]
    
    # 필수 파일 확인
    all_ok = True
    for filename, description in required_files.items():
        filepath = os.path.join(model_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"✅ {description}: {filename} ({size:,} bytes)")
        else:
            print(f"❌ {description}: {filename} (없음)")
            all_ok = False
    
    # 모델 가중치 파일 확인 (하나라도 있으면 OK)
    model_file_found = False
    for filename in model_weight_files:
        filepath = os.path.join(model_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            size_mb = size / (1024 * 1024)
            print(f"✅ 모델 가중치: {filename} ({size_mb:.1f} MB)")
            model_file_found = True
            break
    
    if not model_file_found:
        print(f"❌ 모델 가중치 파일을 찾을 수 없습니다")
        print(f"   필요한 파일: {', '.join(model_weight_files)}")
        all_ok = False
    
    return all_ok


def test_model_loading():
    """모델 로딩 테스트"""
    print("\n" + "="*60)
    print("🔄 모델 로딩 테스트")
    print("="*60)
    
    try:
        from src.dialect_analyzer import DialectAnalyzer
        
        print("방언 분석기를 초기화하는 중...")
        analyzer = DialectAnalyzer('models/dialect_binary_classifier/final_model')
        
        print(f"\n모델 로드 상태: {'✅ 성공' if analyzer.model_loaded else '❌ 실패'}")
        print(f"방언 어휘 사전: {len(analyzer.dialect_vocab)}개 단어")
        print(f"억양 분석 가능: {'✅' if analyzer.model_loaded else '❌'}")
        print(f"어휘 분석 가능: {'✅' if len(analyzer.dialect_vocab) > 0 else '❌'}")
        
        if analyzer.is_available():
            print("\n✅ 방언 검출 시스템이 정상적으로 작동합니다!")
            
            # 어휘 검출 테스트
            print("\n" + "-"*60)
            print("📝 어휘 검출 테스트")
            print("-"*60)
            
            test_texts = [
                "오늘 날씨가 참 좋네요",  # 표준어
                "오늘 날씨가 참 좋네, 가가리도 보이고",  # 경상도
                "당께 내가 그랬제? 밥 묵었어?",  # 전라도
            ]
            
            for text in test_texts:
                detected = analyzer.detect_dialect_vocabulary(text)
                if detected:
                    print(f"\n'{text}'")
                    for word_info in detected:
                        print(f"  → '{word_info['word']}' ({word_info['region']}도, 뜻: {word_info['standard_meaning']})")
                else:
                    print(f"\n'{text}'")
                    print(f"  → 방언 단어 없음 (표준어)")
            
            return True
        else:
            print("\n⚠️  모델은 로드되지 않았지만 어휘 분석은 가능합니다.")
            return False
            
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 테스트 실행"""
    print("\n" + "="*60)
    print("🧪 방언 검출 시스템 테스트")
    print("="*60)
    
    results = {
        '의존성': test_dependencies(),
        '어휘 사전': test_vocabulary(),
        '모델 파일': test_model_files(),
        '모델 로딩': test_model_loading()
    }
    
    print("\n" + "="*60)
    print("📊 테스트 결과 요약")
    print("="*60)
    
    for name, result in results.items():
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 모든 테스트 통과! 방언 검출 시스템을 사용할 수 있습니다.")
        print("\n다음 명령어로 프로그램을 실행하세요:")
        print("  python main.py")
        return 0
    else:
        print("\n⚠️  일부 테스트 실패. 위의 오류 메시지를 확인하세요.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
