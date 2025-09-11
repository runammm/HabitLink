import os
import asyncio
import tempfile
from src.audio import AudioInterface
from src.whisperlivekit_stt import stt_with_speakers, stt, WhisperLiveStreaming
from word_analyzer import WordAnalyzer

# --- 전역 변수 및 초기화 ---
audio_interface = AudioInterface(samplerate=16000, channels=1)

# --- 비동기 STT 처리 함수 ---


async def process_diarization_result(task: asyncio.Task, path: str, rec_num: int, word_analyzer: WordAnalyzer):
    """화자 분리 STT 결과를 처리하고 단어를 분석합니다."""
    try:
        result = task.result()
        if not result or not result.get('speakers'):
            print(f"❌ [{rec_num:03d}] 화자 구분 결과가 없습니다.")
            return

        print(f"\n📝 [{rec_num:03d}] ===== 화자별 발화 내용 =====")
        for i, speaker_info in enumerate(result['speakers'], 1):
            speaker, text = speaker_info['speaker'], speaker_info['text']
            print(f"    {i:2d}. {speaker}: {text}")
            word_analyzer.analyze(text)
        
        full_text = result.get('full_text', '')
        formatted_text = full_text.replace('\n', '\n    ')
        print(f"💬 [{rec_num:03d}] 전체 대화:\n    {formatted_text}")
        print("=" * 50 + "\n")

    except Exception as e:
        print(f"❌ [{rec_num:03d}] STT 처리 오류: {e!r}")
    finally:
        if os.path.exists(path):
            os.remove(path)
            print(f"🗑️ [{rec_num:03d}] 파일 정리 완료")


async def process_simple_stt_result(path: str, rec_num: int, word_analyzer: WordAnalyzer):
    """기본 STT 결과를 처리하고 단어를 분석합니다."""
    try:
        print(f"🔄 [{rec_num:03d}] STT 작업 시작...")
        result_text = await stt(path)
        if result_text:
            print(f"📝 [{rec_num:03d}] 인식 결과: {result_text}")
            word_analyzer.analyze(result_text)
        else:
            print(f"❌ [{rec_num:03d}] 음성 인식 결과가 없습니다.")
    except Exception as e:
        print(f"❌ [{rec_num:03d}] STT 처리 오류: {e!r}")

# --- 메인 사이클 함수 ---


async def run_continuous_stt_cycle(word_analyzer: WordAnalyzer, with_diarization: bool):
    """연속 음성 인식을 위한 메인 루프를 실행합니다."""
    mode_title = "화자 구분" if with_diarization else "기본"
    prefix = "voice_diarization_" if with_diarization else "voice_simple_"
    
    print(f"\n=== {mode_title} 연속 음성 인식 ===")
    print(f"- 대상 단어: {', '.join(word_analyzer.target_words)}")
    print("- Ctrl+C로 종료할 수 있습니다.\n")

    start_time = asyncio.get_event_loop().time()
    pending_tasks = set()
    recording_count = 0

    try:
        while True:
            recording_count += 1
            with tempfile.NamedTemporaryFile(prefix=f"{prefix}{recording_count:03d}_", suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            print(f"🎤 [{recording_count:03d}] 녹음 시작... (10초)")
            out_path = await audio_interface.record_to_file(10, tmp_path)
            print(f"✅ [{recording_count:03d}] 녹음 완료 -> {os.path.basename(out_path)}")

            if with_diarization:
                task = asyncio.create_task(stt_with_speakers(out_path))
                callback = lambda t: asyncio.create_task(process_diarization_result(t, out_path, recording_count, word_analyzer))
                task.add_done_callback(callback)
            else:
                task = asyncio.create_task(process_simple_stt_result(out_path, recording_count, word_analyzer))
            
            pending_tasks.add(task)
            task.add_done_callback(pending_tasks.discard)

            if len(pending_tasks) >= 3:
                print(f"⏳ STT 큐 가득참 ({len(pending_tasks)}개). 하나 완료될 때까지 대기...")
                await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
    
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 종료되었습니다.")
        if pending_tasks:
            print(f"⏳ 남은 STT 작업({len(pending_tasks)}개) 완료까지 대기 중...")
            await asyncio.wait(pending_tasks)
            print("✅ 모든 STT 작업 완료")
    
    finally:
        end_time = asyncio.get_event_loop().time()
        total_duration_minutes = (end_time - start_time) / 60
        if total_duration_minutes > 0.01:  # 아주 짧은 실행은 요약 스킵
            word_analyzer.display_summary(total_duration_minutes)

# --- 사용자 인터페이스 및 메인 실행 로직 ---


def get_target_words_from_user() -> WordAnalyzer:
    """사용자로부터 분석할 단어를 입력받아 WordAnalyzer 객체를 생성합니다."""
    words_input = input("분석할 단어를 쉼표(,)로 구분하여 입력하세요 (기본값: 예,아니오,음): ").strip()
    if not words_input:
        target_words = ["예", "아니오", "음"]
    else:
        target_words = [word.strip() for word in words_input.split(',') if word.strip()]
    return WordAnalyzer(target_words)


async def run_streaming_mode():
    """실시간 스트리밍 모드를 선택하고 실행합니다."""
    print("\n=== 실시간 스트리밍 모드 ===")
    print("1. 화자 구분 실시간 인식")
    print("2. 기본 실시간 인식 (화자 구분 없음)")
    print("==========================")
    choice = input("스트리밍 모드를 선택하세요 (1 또는 2): ").strip()
    diarization_enabled = choice == "1"
    
    streaming = WhisperLiveStreaming(model="base", language="ko", diarization=diarization_enabled)
    await streaming.start_streaming()


async def main():
    """메인 함수: 사용자 입력을 받아 적절한 모드를 실행합니다."""
    try:
        print("=== HabitLink 음성 분석 시스템 ===")
        word_analyzer = get_target_words_from_user()

        print("\n1. 연속 음성 파일 분석 (화자 구분)")
        print("2. 기본 음성 인식 (화자 구분 없음)")
        print("3. 실시간 스트리밍 분석")
        print("===================================")
        mode = input("모드를 선택하세요 (1, 2, 또는 3): ").strip()

        if mode == "1":
            await run_continuous_stt_cycle(word_analyzer, with_diarization=True)
        elif mode == "2":
            await run_continuous_stt_cycle(word_analyzer, with_diarization=False)
        elif mode == "3":
            await run_streaming_mode()
        else:
            print("잘못된 선택입니다. 기본 음성 인식 모드로 시작합니다.")
            await run_continuous_stt_cycle(word_analyzer, with_diarization=False)

    except (KeyboardInterrupt, EOFError):
        print("\n프로그램을 종료합니다.")
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    asyncio.run(main())
