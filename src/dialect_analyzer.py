"""
Dialect Analyzer Module

This module uses a fine-tuned Wav2Vec2 model to detect Korean dialects from audio.
"""

import os
from typing import Dict, Optional, List
import traceback


class DialectAnalyzer:
    """
    Uses a fine-tuned speech classification model to detect Korean dialects from audio.
    """
    
    def __init__(self, model_path: str):
        """
        Initializes the DialectAnalyzer by loading the fine-tuned model.

        Args:
            model_path (str): The local path or Hugging Face Hub ID of the fine-tuned
                              dialect classification model.
        """
        self.model_path = model_path
        self.classifier = None
        self.model_loaded = False
        
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
    
    def analyze(self, audio_path: str, top_k: int = 5) -> Dict[str, float]:
        """
        Analyzes a single audio file and returns the predicted dialect probabilities.

        Args:
            audio_path (str): The path to the audio file to be analyzed.
            top_k (int): Number of top predictions to return (default: 5).

        Returns:
            Dict[str, float]: A dictionary of dialect labels and their corresponding
                              confidence scores.
                              Example: {'gyeongsang': 0.85, 'standard': 0.10, 'jeolla': 0.05}
        """
        if not self.model_loaded or self.classifier is None:
            return {"error": "Model not loaded. Please train the model first."}
        
        if not os.path.exists(audio_path):
            return {"error": f"Audio file not found: {audio_path}"}
        
        try:
            # Run inference
            predictions = self.classifier(audio_path, top_k=top_k)
            
            # Format the output into a simple dictionary
            probabilities = {p['label']: round(p['score'], 4) for p in predictions}
            return probabilities
            
        except Exception as e:
            print(f"⚠️ 방언 분석 중 오류 발생: {e}")
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
    
    def get_top_dialect(self, audio_path: str) -> Dict[str, any]:
        """
        Get the most likely dialect for an audio file.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            Dict containing 'dialect' and 'confidence' keys
        """
        probabilities = self.analyze(audio_path, top_k=1)
        
        if "error" in probabilities:
            return {"dialect": None, "confidence": 0.0, "error": probabilities["error"]}
        
        if not probabilities:
            return {"dialect": None, "confidence": 0.0}
        
        # Get the dialect with highest probability
        top_dialect = max(probabilities.items(), key=lambda x: x[1])
        
        return {
            "dialect": top_dialect[0],
            "confidence": top_dialect[1]
        }
    
    def is_available(self) -> bool:
        """
        Check if the dialect analyzer is available and ready to use.
        
        Returns:
            bool: True if model is loaded and ready, False otherwise
        """
        return self.model_loaded and self.classifier is not None
    
    def get_dialect_name_korean(self, dialect_label: str) -> str:
        """
        Convert English dialect label to Korean name.
        
        Args:
            dialect_label (str): English label (e.g., 'gyeongsang')
            
        Returns:
            str: Korean name (e.g., '경상도 방언')
        """
        dialect_names = {
            'standard': '표준어 (서울/수도권)',
            'gyeongsang': '경상도 방언',
            'jeolla': '전라도 방언',
            'chungcheong': '충청도 방언',
            'gangwon': '강원도 방언',
            'jeju': '제주도 방언'
        }
        
        return dialect_names.get(dialect_label.lower(), dialect_label)

