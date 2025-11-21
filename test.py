# test.py
import os
from rust_core import DataCleanser

def create_test_file(filename="test_data.txt"):
    """테스트를 위한 데이터 파일을 생성합니다."""
    data = """
이것도 짧아요 # 필터링 대상 (길이 부족)
  이것은 첫 번째 고유한 문장입니다. 대문자도 섞여 있습니다.
이 문장은 필터를 통과하고 고유한 것으로 추적되어야 합니다.
이것은 첫 번째 고유한 문장입니다. 대문자도 섞여 있습니다. # 중복 라인 (첫 번째와 동일)
This is a fourth unique line that passes the minimum length filter.
THIS IS A FOURTH UNIQUE LINE THAT PASSES THE MINIMUM LENGTH FILTER. # 중복 라인 (대소문자 무시)
짧은 텍스트. # 필터링 대상
badword1이 포함된 문장은 필터링되어야 합니다. # 유해 단어 필터링
아주 긴 문장이지만 offensive_term이 들어있어서 삭제될 운명입니다. # 유해 단어 필터링
"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(data.strip())
    print(f"✅ {filename} 파일 생성 완료.")

def run_file_processor_test():
    """DataCleanser 클래스의 process_file 기능을 테스트합니다."""
    print("--- 🚀 File Processor 테스트 시작 ---")

    filename = "test_data.txt"

    # 1. 테스트 파일 준비
    create_test_file(filename)

    cleaner = DataCleanser()

    try:
        # [핵심 변경 사항]
        # Python 파일 객체(open(...))가 아니라, '파일 경로(String)'를 Rust에게 직접 넘깁니다.
        # 이렇게 하면 Rust가 직접 파일을 열어서 처리하므로 속도가 훨씬 빠르고 메모리도 적게 듭니다.
        print(f"🦀 Rust 엔진에게 파일 처리 요청: {filename}")

        processed_unique_count = cleaner.process_file(filename)

        print(f"\n✅ Rust가 처리한 고유 라인 수: {processed_unique_count}개")
        print(f"✅ Rust 내부 저장소(HashSet) 크기: {cleaner.count}개")

        # [예상 결과 분석]
        # 1. "이것도 짧아요..." -> 필터 (길이)
        # 2. "이것은 첫 번째..." -> 통과 (고유 1)
        # 3. "이 문장은..." -> 통과 (고유 2)
        # 4. "이것은 첫 번째..." -> 중복 (무시)
        # 5. "This is a fourth..." -> 통과 (고유 3)
        # 6. "THIS IS A FOURTH..." -> 중복 (무시)
        # 7. "짧은 텍스트." -> 필터 (길이)
        # 8. "badword1..." -> 필터 (유해 단어)
        # 9. "offensive_term..." -> 필터 (유해 단어)

        expected_unique = 3

        if cleaner.count == expected_unique:
             print(f"\n🏆 성공: 최종 추적 개수가 예상치({expected_unique})와 일치합니다.")
        else:
             print(f"\n❌ 실패: 예상 {expected_unique}개, 실제 {cleaner.count}개")

    except Exception as e:
        print(f"❌ 처리 중 예외 발생: {e}")
    finally:
        # 테스트 후 파일 삭제 (선택 사항)
        if os.path.exists(filename):
            os.remove(filename)
            print(f"✅ {filename} 파일 정리 완료.")

if __name__ == "__main__":
    run_file_processor_test()
