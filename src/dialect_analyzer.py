"""
Dialect Analyzer Module

This module uses a fine-tuned Wav2Vec2 model for binary classification:
- Standard Korean (표준어) vs Non-Standard Korean (비표준어)

It also includes vocabulary-based dialect detection for comprehensive analysis.
"""

import os
from typing import Dict, Optional, List, Tuple
import traceback
import re
from collections import defaultdict


class DialectAnalyzer:
    """
    Uses a fine-tuned speech classification model for binary classification:
    Standard Korean (표준어) vs Non-Standard Korean (비표준어)
    
    Also detects dialect vocabulary in transcribed text.
    """
    
    def __init__(self, model_path: str, vocabulary_path: Optional[str] = None):
        """
        Initializes the DialectAnalyzer by loading the fine-tuned binary classification model
        and dialect vocabulary dictionary.

        Args:
            model_path (str): The local path or Hugging Face Hub ID of the fine-tuned
                              binary classification model (standard vs non-standard).
            vocabulary_path (str, optional): Path to dialect vocabulary file. If None,
                                           uses default path.
        """
        self.model_path = model_path
        self.classifier = None
        self.model_loaded = False
        self.is_binary = True  # Binary classification mode
        
        # Dialect vocabulary dictionary
        self.dialect_vocab = {}  # {word: (region, standard_meaning)}
        self.dialect_patterns = []  # Compiled regex patterns for fast matching
        
        # Load dialect vocabulary
        if vocabulary_path is None:
            vocabulary_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                "resources", 
                "dialect_vocabulary.txt"
            )
        self._load_dialect_vocabulary(vocabulary_path)
        
        # Real-time analysis tracking
        self.realtime_results = []  # Store 10-second segment results
        self.detected_vocabulary = []  # Store detected dialect words
        
        # Try to load the model
        try:
            if not os.path.exists(model_path):
                print(f"⚠️ 방언 분석 모델을 찾을 수 없습니다: {model_path}")
                print(f"   노트북 'notebooks/dialect_model_training.ipynb'를 실행하여 모델을 먼저 학습시켜주세요.")
                return
            
            # Import transformers here to avoid dependency issues if not installed
            from transformers import pipeline
            
            # The 'audio-classification' pipeline handles all preprocessing automatically
            print(f"방언 분석 모델을 로드하는 중: {model_path}")
            self.classifier = pipeline(
                "audio-classification",
                model=model_path,
                device=-1  # Use CPU by default (can be changed to 0 for GPU)
            )
            self.model_loaded = True
            print("✅ 방언 분석 모델 로드 완료!")
            
        except ImportError as e:
            print(f"⚠️ Transformers 라이브러리가 설치되지 않았습니다.")
            print(f"   다음 명령어로 설치하세요: pip install transformers torch")
            print(f"   에러: {e}")
        except Exception as e:
            print(f"⚠️ 방언 분석 모델 로드 중 오류 발생: {e}")
            traceback.print_exc()
    
    def _load_dialect_vocabulary(self, vocabulary_path: str):
        """
        Load dialect vocabulary from file.
        
        Args:
            vocabulary_path (str): Path to vocabulary file
        """
        try:
            if not os.path.exists(vocabulary_path):
                print(f"⚠️ 방언 어휘 사전을 찾을 수 없습니다: {vocabulary_path}")
                print(f"   어휘 기반 방언 검출이 비활성화됩니다.")
                return
            
            with open(vocabulary_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse format: 방언용어|지역|표준어뜻
                    parts = line.split('|')
                    if len(parts) == 3:
                        word, region, meaning = parts
                        word = word.strip()
                        region = region.strip()
                        meaning = meaning.strip()
                        
                        self.dialect_vocab[word] = (region, meaning)
                        # Create regex pattern for whole word matching
                        # Use word boundaries to avoid partial matches
                        pattern = re.compile(r'\b' + re.escape(word) + r'\b')
                        self.dialect_patterns.append((word, pattern))
            
            print(f"✅ 방언 어휘 사전 로드 완료 ({len(self.dialect_vocab)}개 단어)")
        
        except Exception as e:
            print(f"⚠️ 방언 어휘 사전 로드 중 오류: {e}")
            traceback.print_exc()
    
    def analyze(self, audio_path: str, top_k: int = 2) -> Dict[str, float]:
        """
        Analyzes a single audio file and returns binary classification probabilities.

        Args:
            audio_path (str): The path to the audio file to be analyzed.
            top_k (int): Number of predictions to return (default: 2 for binary).

        Returns:
            Dict[str, float]: A dictionary with binary classification probabilities.
                              Example: {'standard': 0.85, 'non_standard': 0.15}
        """
        if not self.model_loaded or self.classifier is None:
            return {"error": "Model not loaded. Please train the model first."}
        
        if not os.path.exists(audio_path):
            return {"error": f"Audio file not found: {audio_path}"}
        
        try:
            # Check file size
            file_size = os.path.getsize(audio_path)
            print(f"  🔍 분석 중: {audio_path} ({file_size / (1024*1024):.1f} MB)")
            
            # Run inference
            predictions = self.classifier(audio_path, top_k=2)  # Binary classification
            
            if not predictions:
                print(f"  ⚠️ 모델이 예측을 반환하지 않았습니다")
                return {"error": "Model returned empty predictions"}
            
            print(f"  ✅ 모델 예측 완료: {len(predictions)}개 결과")
            for p in predictions:
                print(f"     • {p['label']}: {p['score']*100:.2f}%")
            
            # Format the output into a simple dictionary
            probabilities = {p['label']: round(p['score'], 4) for p in predictions}
            return probabilities
            
        except Exception as e:
            print(f"  ⚠️ 방언 분석 중 오류 발생: {e}")
            traceback.print_exc()
            return {"error": str(e)}
    
    def analyze_segment(self, audio_array, sample_rate: int = 16000, top_k: int = 5) -> Dict[str, float]:
        """
        Analyzes an audio segment (numpy array) and returns dialect probabilities.
        
        Args:
            audio_array: Numpy array containing audio data
            sample_rate (int): Sample rate of the audio (default: 16000)
            top_k (int): Number of top predictions to return
            
        Returns:
            Dict[str, float]: Dictionary of dialect labels and confidence scores
        """
        if not self.model_loaded or self.classifier is None:
            return {"error": "Model not loaded. Please train the model first."}
        
        try:
            import tempfile
            import soundfile as sf
            
            # Save audio array to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                sf.write(temp_path, audio_array, sample_rate)
            
            # Analyze the temporary file
            result = self.analyze(temp_path, top_k=top_k)
            
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except:
                pass
            
            return result
            
        except Exception as e:
            print(f"⚠️ 방언 분석 중 오류 발생: {e}")
            traceback.print_exc()
            return {"error": str(e)}
    
    def get_classification(self, audio_path: str) -> Dict[str, any]:
        """
        Get binary classification result (standard vs non-standard).
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            Dict containing 'is_standard', 'confidence', and 'probabilities' keys
        """
        probabilities = self.analyze(audio_path, top_k=2)
        
        if "error" in probabilities:
            return {
                "is_standard": None, 
                "confidence": 0.0, 
                "error": probabilities["error"],
                "probabilities": {}
            }
        
        if not probabilities:
            return {"is_standard": None, "confidence": 0.0, "probabilities": {}}
        
        # Get standard probability
        standard_prob = probabilities.get("standard", probabilities.get("LABEL_0", 0.0))
        non_standard_prob = probabilities.get("non_standard", probabilities.get("LABEL_1", 0.0))
        
        is_standard = standard_prob > non_standard_prob
        confidence = max(standard_prob, non_standard_prob)
        
        return {
            "is_standard": is_standard,
            "confidence": confidence,
            "probabilities": {
                "standard": standard_prob,
                "non_standard": non_standard_prob
            }
        }
    
    def is_available(self) -> bool:
        """
        Check if the dialect analyzer is available and ready to use.
        
        Returns:
            bool: True if model is loaded and ready, False otherwise
        """
        return self.model_loaded and self.classifier is not None
    
    def detect_dialect_vocabulary(self, text: str, timestamp: Optional[float] = None) -> List[Dict]:
        """
        Detect dialect vocabulary in transcribed text.
        
        Args:
            text (str): Transcribed text to analyze
            timestamp (float, optional): Timestamp of the text
            
        Returns:
            List[Dict]: List of detected dialect words with metadata
        """
        detected = []
        
        if not self.dialect_patterns:
            return detected
        
        # Search for dialect words in text
        for word, pattern in self.dialect_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                region, meaning = self.dialect_vocab[word]
                detected.append({
                    'word': word,
                    'region': region,
                    'standard_meaning': meaning,
                    'position': match.start(),
                    'timestamp': timestamp,
                    'context': text[max(0, match.start()-10):min(len(text), match.end()+10)]
                })
        
        # Store in detected vocabulary
        self.detected_vocabulary.extend(detected)
        
        return detected
    
    def analyze_segment_realtime(self, audio_array, sample_rate: int = 16000, 
                                  timestamp: Optional[float] = None,
                                  text: Optional[str] = None) -> Dict:
        """
        Analyze a 10-second audio segment in real-time for dialect detection.
        Combines acoustic analysis (Wav2Vec2) and vocabulary analysis.
        
        Args:
            audio_array: Numpy array containing audio data (10 seconds)
            sample_rate (int): Sample rate of the audio (default: 16000)
            timestamp (float, optional): Timestamp of the segment
            text (str, optional): Transcribed text for vocabulary analysis
            
        Returns:
            Dict: Analysis result with acoustic and vocabulary components
        """
        result = {
            'timestamp': timestamp,
            'acoustic_analysis': None,
            'vocabulary_analysis': None,
            'combined_verdict': 'standard',  # 'standard' or 'non_standard'
            'confidence': 0.0,
            'feedback_trigger': False  # Whether to give real-time feedback
        }
        
        # 1. Acoustic analysis (Wav2Vec2)
        if self.model_loaded and self.classifier is not None:
            try:
                import tempfile
                import soundfile as sf
                
                # Save audio array to temporary file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                    sf.write(temp_path, audio_array, sample_rate)
                
                # Analyze
                probabilities = self.analyze(temp_path, top_k=2)
                
                # Clean up
                try:
                    os.remove(temp_path)
                except:
                    pass
                
                if "error" not in probabilities:
                    standard_prob = probabilities.get("standard", probabilities.get("LABEL_0", 0.0))
                    non_standard_prob = probabilities.get("non_standard", probabilities.get("LABEL_1", 0.0))
                    
                    result['acoustic_analysis'] = {
                        'standard_prob': standard_prob,
                        'non_standard_prob': non_standard_prob,
                        'verdict': 'standard' if standard_prob > non_standard_prob else 'non_standard'
                    }
                    
            except Exception as e:
                print(f"⚠️ 실시간 음성 분석 오류: {e}")
        
        # 2. Vocabulary analysis
        if text:
            detected_words = self.detect_dialect_vocabulary(text, timestamp)
            if detected_words:
                result['vocabulary_analysis'] = {
                    'detected_words': detected_words,
                    'count': len(detected_words),
                    'regions': list(set(w['region'] for w in detected_words))
                }
        
        # 3. Combined verdict
        # Priority: If vocabulary detected, it's non-standard
        # Otherwise, use acoustic analysis with 70% threshold to reduce false positives
        if result['vocabulary_analysis'] and result['vocabulary_analysis']['count'] > 0:
            result['combined_verdict'] = 'non_standard'
            result['confidence'] = 0.9  # High confidence for vocabulary match
            result['feedback_trigger'] = True
        elif result['acoustic_analysis']:
            acoustic = result['acoustic_analysis']
            if acoustic['non_standard_prob'] >= 0.70:  # 70% threshold
                result['combined_verdict'] = 'non_standard'
                result['confidence'] = acoustic['non_standard_prob']
                result['feedback_trigger'] = True
            else:
                result['combined_verdict'] = 'standard'
                result['confidence'] = acoustic['standard_prob']
        
        # Store result for aggregation
        self.realtime_results.append(result)
        
        return result
    
    def get_realtime_summary(self) -> Dict:
        """
        Get summary of real-time analysis results.
        
        Returns:
            Dict: Summary statistics of real-time analysis
        """
        if not self.realtime_results:
            return {
                'total_segments': 0,
                'non_standard_count': 0,
                'non_standard_ratio': 0.0,
                'vocabulary_detections': 0,
                'avg_confidence': 0.0
            }
        
        total = len(self.realtime_results)
        non_standard_count = sum(1 for r in self.realtime_results if r['combined_verdict'] == 'non_standard')
        vocab_count = sum(
            r['vocabulary_analysis']['count'] 
            for r in self.realtime_results 
            if r['vocabulary_analysis']
        )
        
        confidences = [r['confidence'] for r in self.realtime_results if r['confidence'] > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            'total_segments': total,
            'non_standard_count': non_standard_count,
            'non_standard_ratio': non_standard_count / total if total > 0 else 0.0,
            'vocabulary_detections': vocab_count,
            'avg_confidence': avg_confidence,
            'detected_words': self.detected_vocabulary
        }
    
    def clear_realtime_data(self):
        """Clear real-time analysis data."""
        self.realtime_results.clear()
        self.detected_vocabulary.clear()
    
    def get_label_name_korean(self, label: str) -> str:
        """
        Convert English label to Korean name.
        
        Args:
            label (str): English label (e.g., 'standard', 'non_standard')
            
        Returns:
            str: Korean name (e.g., '표준어', '비표준어')
        """
        label_names = {
            'standard': '표준어',
            'non_standard': '비표준어',
            'LABEL_0': '표준어',
            'LABEL_1': '비표준어'
        }
        
        return label_names.get(label, label)

