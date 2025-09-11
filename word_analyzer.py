import time
from collections import defaultdict

class WordAnalyzer:
    """
    실시간으로 텍스트에서 특정 단어의 등장을 감지하고,
    세션 종료 시 단어 사용 빈도에 대한 요약 정보를 제공하는 클래스.
    """
    def __init__(self, target_words: list[str]):
        """
        WordAnalyzer를 초기화합니다.

        Args:
            target_words (list[str]): 감지할 대상 단어의 리스트.
        """
        self.target_words = [word.lower() for word in target_words]
        self.word_counts = defaultdict(int)
        print(f"✅ WordAnalyzer 초기화 완료. 대상 단어: {', '.join(target_words)}")

    def analyze(self, text: str):
        """
        주어진 텍스트에서 대상 단어를 분석하고, 발견 시 즉시 알림을 출력합니다.

        Args:
            text (str): 분석할 텍스트.
        """
        text_lower = text.lower()
        for word in self.target_words:
            if word in text_lower:
                count = text_lower.count(word)
                self.word_counts[word] += count
                # 실시간 감지 신호
                print(f"🎯 단어 감지! -> '{word}' ({count}회)")

    def display_summary(self, total_duration_minutes: float):
        """
        분석 세션의 요약 정보를 출력합니다.

        Args:
            total_duration_minutes (float): 총 발화 시간 (분 단위).
        """
        print("\n\n📊 ===== 최종 단어 사용 빈도 분석 요약 =====")
        print(f"총 분석 시간: {total_duration_minutes:.2f}분")
        print("---------------------------------------------")
        
        if not self.word_counts:
            print("감지된 대상 단어가 없습니다.")
            print("---------------------------------------------")
            return

        print(f"{'단어':<10} | {'총 사용 횟수':<15} | {'분당 사용 빈도':<15}")
        print("---------------------------------------------")

        sorted_counts = sorted(self.word_counts.items(), key=lambda item: item[1], reverse=True)

        for word, count in sorted_counts:
            frequency = count / total_duration_minutes if total_duration_minutes > 0 else 0
            print(f"{word:<10} | {count:<15} | {frequency:<15.2f} 회/분")
        
        print("=============================================\n")

if __name__ == '__main__':
    # WordAnalyzer 클래스 테스트를 위한 예제 코드
    async def test_word_analyzer():
        target_words = ["테스트", "분석", "실시간"]
        analyzer = WordAnalyzer(target_words)
        
        sample_texts = [
            "이것은 첫 번째 테스트 문장입니다.",
            "실시간 분석 기능을 테스트하고 있습니다.",
            "분석이 잘 되는지 확인하는 테스트입니다. 실시간으로요.",
            "이 문장에는 대상 단어가 없습니다.",
            "테스트, 테스트, 한 번 더 테스트!"
        ]
        
        total_duration_seconds = 10
        print(f"\n{total_duration_seconds}초 동안 가상으로 텍스트 분석을 시작합니다...\n")

        for i, text in enumerate(sample_texts):
            print(f"[{i+1:02d}] 수신 텍스트: \"{text}\"")
            analyzer.analyze(text)
            time.sleep(1) # 1초 간격으로 텍스트가 들어오는 것을 시뮬레이션

        print("\n가상 분석이 종료되었습니다.")
        
        total_duration_minutes = total_duration_seconds / 60
        analyzer.display_summary(total_duration_minutes)

    # 비동기 테스트 함수 실행
    import asyncio
    asyncio.run(test_word_analyzer())

