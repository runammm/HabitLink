"""
Dialect Analyzer Module

This module uses a fine-tuned Wav2Vec2 model for binary classification:
- Standard Korean (표준어) vs Non-Standard Korean (비표준어)
"""

import os
from typing import Dict, Optional, List
import traceback


class DialectAnalyzer:
    """
    Uses a fine-tuned speech classification model for binary classification:
    Standard Korean (표준어) vs Non-Standard Korean (비표준어)
    """
    
    def __init__(self, model_path: str):
        """
        Initializes the DialectAnalyzer by loading the fine-tuned binary classification model.

        Args:
            model_path (str): The local path or Hugging Face Hub ID of the fine-tuned
                              binary classification model (standard vs non-standard).
        """
        self.model_path = model_path
        self.classifier = None
        self.model_loaded = False
        self.is_binary = True  # Binary classification mode
        
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
            # Run inference
            predictions = self.classifier(audio_path, top_k=2)  # Binary classification
            
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

