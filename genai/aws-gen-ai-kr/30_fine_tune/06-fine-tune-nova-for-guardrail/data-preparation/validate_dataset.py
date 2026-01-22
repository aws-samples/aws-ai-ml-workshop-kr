#!/usr/bin/env python3
"""
패션 가드레일 데이터셋 검증 스크립트

이 스크립트는 다음을 검증합니다:
1. 필수 필드 존재 여부
2. 레이블 값 유효성
3. 분할 내 중복
4. 분할 간 중복
5. 카테고리 분포
6. 레이블 비율
"""

import json
import sys
from pathlib import Path
from collections import Counter

def validate_dataset(filepath, name):
    """단일 데이터셋 파일 검증"""
    print(f"\n{'='*60}")
    print(f"검증 중: {name}")
    print(f"{'='*60}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 파일 로드 실패: {e}")
        return None, False

    print(f"샘플 수: {len(data)}")

    # 필수 필드 검사
    required_fields = ['messages', 'teacher_response', 'label', 'category']
    missing_fields = []

    for i, item in enumerate(data):
        for field in required_fields:
            if field not in item:
                missing_fields.append((i, field))

    if missing_fields:
        print(f"❌ 누락된 필드: {len(missing_fields)}개")
        for idx, field in missing_fields[:5]:  # 처음 5개만 표시
            print(f"   - Index {idx}: '{field}' 누락")
        return None, False
    else:
        print(f"✅ 필수 필드: 모든 샘플 완전")

    # 레이블 검증
    labels = [item['label'] for item in data]
    invalid_labels = [l for l in labels if l not in ['Safe', 'Unsafe']]

    if invalid_labels:
        print(f"❌ 잘못된 레이블: {len(invalid_labels)}개")
        return None, False

    label_counts = Counter(labels)
    unsafe_count = label_counts.get('Unsafe', 0)
    safe_count = label_counts.get('Safe', 0)
    unsafe_pct = unsafe_count / len(data) * 100 if data else 0
    safe_pct = safe_count / len(data) * 100 if data else 0

    print(f"✅ 레이블 분포:")
    print(f"   - Unsafe: {unsafe_count} ({unsafe_pct:.1f}%)")
    print(f"   - Safe: {safe_count} ({safe_pct:.1f}%)")

    # 카테고리 분포
    categories = [item['category'] for item in data]
    category_counts = Counter(categories)

    print(f"✅ 카테고리: {len(category_counts)}개 고유")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"   - {cat}: {count}")

    # 내부 중복 검사
    contents = [item['messages'][0]['content'] for item in data]
    unique_contents = set(contents)

    if len(contents) != len(unique_contents):
        duplicates = len(contents) - len(unique_contents)
        print(f"❌ 내부 중복: {duplicates}개")

        # 중복 샘플 찾기
        seen = {}
        for i, content in enumerate(contents):
            if content in seen:
                print(f"   - Index {i}: '{content[:50]}...' (처음: {seen[content]})")
            else:
                seen[content] = i
        return None, False
    else:
        print(f"✅ 내부 중복: 없음")

    return set(contents), True

def main():
    """메인 검증 함수"""
    base_dir = Path(__file__).parent.parent
    dataset_dir = base_dir / 'dataset'

    print("="*60)
    print("패션 가드레일 데이터셋 검증")
    print("="*60)

    # 각 분할 검증
    files = {
        'Train': dataset_dir / 'train.json',
        'Validation': dataset_dir / 'validation.json',
        'Test': dataset_dir / 'test.json'
    }

    all_valid = True
    all_contents = {}

    for name, filepath in files.items():
        if not filepath.exists():
            print(f"\n❌ {name}: 파일 없음 ({filepath})")
            all_valid = False
            continue

        contents, valid = validate_dataset(filepath, name)
        if not valid:
            all_valid = False
        else:
            all_contents[name] = contents

    # 분할 간 중복 검사
    if len(all_contents) == 3:
        print(f"\n{'='*60}")
        print("분할 간 중복 검사")
        print(f"{'='*60}")

        train_val = all_contents['Train'] & all_contents['Validation']
        train_test = all_contents['Train'] & all_contents['Test']
        val_test = all_contents['Validation'] & all_contents['Test']

        if train_val:
            print(f"❌ Train-Validation 중복: {len(train_val)}개")
            for item in list(train_val)[:3]:
                print(f"   - '{item[:50]}...'")
            all_valid = False
        else:
            print(f"✅ Train-Validation: 중복 없음")

        if train_test:
            print(f"❌ Train-Test 중복: {len(train_test)}개")
            for item in list(train_test)[:3]:
                print(f"   - '{item[:50]}...'")
            all_valid = False
        else:
            print(f"✅ Train-Test: 중복 없음")

        if val_test:
            print(f"❌ Validation-Test 중복: {len(val_test)}개")
            for item in list(val_test)[:3]:
                print(f"   - '{item[:50]}...'")
            all_valid = False
        else:
            print(f"✅ Validation-Test: 중복 없음")

        # 총계
        total_unique = len(all_contents['Train'] | all_contents['Validation'] | all_contents['Test'])
        total_samples = len(all_contents['Train']) + len(all_contents['Validation']) + len(all_contents['Test'])

        print(f"\n{'='*60}")
        print("최종 결과")
        print(f"{'='*60}")
        print(f"전체 샘플: {total_samples}")
        print(f"고유 샘플: {total_unique}")

        if total_unique == total_samples:
            print(f"\n🎉 검증 성공: 데이터셋이 완벽합니다!")
        else:
            print(f"\n⚠️  중복 {total_samples - total_unique}개 발견")
            all_valid = False

    if all_valid:
        print(f"\n{'='*60}")
        print("✅ 모든 검증 통과")
        print(f"{'='*60}")
        return 0
    else:
        print(f"\n{'='*60}")
        print("❌ 검증 실패: 오류 수정 필요")
        print(f"{'='*60}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
