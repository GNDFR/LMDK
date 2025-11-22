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

def run_default_filter_test():
    """기본 필터링 규칙으로 DataCleanser를 테스트합니다."""
    print("--- 🚀 1. 기본 필터 테스트 ---")
    filename = "test_data.txt"
    create_test_file(filename)

    # 기본값으로 초기화 (min_length=20, 기본 유해 단어)
    cleaner = DataCleanser()

    try:
        print(f"🦀 Rust 엔진에게 파일 처리 요청: {filename}")
        processed_unique_count = cleaner.process_file(filename)

        print(f"\n✅ Rust가 처리한 고유 라인 수: {processed_unique_count}개")
        print(f"✅ Rust 내부 저장소(HashSet) 크기: {cleaner.count}개")

        expected_unique = 3
        if cleaner.count == expected_unique:
             print(f"🏆 성공: 최종 추적 개수가 예상치({expected_unique})와 일치합니다.")
        else:
             print(f"❌ 실패: 예상 {expected_unique}개, 실제 {cleaner.count}개")

    except Exception as e:
        print(f"❌ 처리 중 예외 발생: {e}")
    finally:
        # 파일은 다음 테스트에서 사용하므로 여기서는 삭제하지 않음
        pass

def run_custom_filter_test():
    """커스텀 필터링 규칙으로 DataCleanser를 테스트합니다."""
    print("\n--- 🚀 2. 커스텀 필터 테스트 ---")
    filename = "test_data.txt"
    # 파일이 이미 생성되었다고 가정

    try:
        # 테스트 2-1: 최소 길이를 32로 늘려서 테스트
        print("\n[테스트 2-1] 최소 길이를 32로 설정")
        cleaner_long = DataCleanser(min_length=32)
        count_long = cleaner_long.process_file(filename)
        # 33자, 33자, 63자 라인이 통과하므로 예상 결과는 3
        expected_long = 3

        print(f"✅ 처리된 라인 수: {count_long}개 (예상: {expected_long}개)")
        if count_long == expected_long:
            print("👍 성공!")
        else:
            print("👎 실패!")

        # 테스트 2-2: 유해 단어 목록을 직접 지정하여 테스트
        print("\n[테스트 2-2] 커스텀 유해 단어 설정")
        # 'badword1'을 허용하고, 대신 '대문자'를 유해 단어로 지정
        cleaner_custom_toxic = DataCleanser(
            min_length=20,
            toxic_keywords=["대문자", "offensive_term"]
        )
        count_custom_toxic = cleaner_custom_toxic.process_file(filename)
        expected_custom_toxic = 3

        print(f"✅ 처리된 라인 수: {count_custom_toxic}개 (예상: {expected_custom_toxic}개)")
        if count_custom_toxic == expected_custom_toxic:
            print("👍 성공!")
        else:
            print("👎 실패!")

    except Exception as e:
        print(f"❌ 처리 중 예외 발생: {e}")
    finally:
        # 모든 테스트가 끝나고 파일 정리
        if os.path.exists(filename):
            os.remove(filename)
            print(f"\n✅ {filename} 파일 정리 완료.")


if __name__ == "__main__":
    run_default_filter_test()
    run_custom_filter_test()
