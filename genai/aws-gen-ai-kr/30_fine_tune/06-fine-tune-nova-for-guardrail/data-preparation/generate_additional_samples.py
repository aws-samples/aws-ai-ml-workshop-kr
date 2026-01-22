#!/usr/bin/env python3
"""
템플릿 기반 데이터셋 확장 스크립트

목표: 400 샘플 → 700 샘플 (+300)
방법: 템플릿과 변수 조합으로 다양한 샘플 생성
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from itertools import product


# ============================================================
# 공통 변수
# ============================================================

PRODUCTS = {
    "clothing": ["원피스", "청바지", "코트", "셔츠", "스커트", "재킷", "니트", "맨투맨",
                 "후드티", "바지", "블라우스", "가디건", "점퍼", "패딩", "정장", "트렌치코트"],
    "shoes": ["운동화", "구두", "부츠", "샌들", "슬리퍼", "로퍼", "하이힐", "스니커즈"],
    "bags": ["백팩", "크로스백", "토트백", "클러치", "숄더백", "에코백", "여행가방", "지갑"],
    "accessories": ["모자", "벨트", "스카프", "장갑", "양말", "선글라스", "목걸이", "귀걸이", "시계"]
}

ALL_PRODUCTS = [p for products in PRODUCTS.values() for p in products]

EVENTS = ["결혼식", "면접", "파티", "데이트", "소개팅", "동창회", "졸업식", "입학식",
          "돌잔치", "생일파티", "회식", "워크샵", "촬영", "공연", "전시회", "명절 가족모임",
          "유튜브 촬영", "인스타그램 촬영", "브이로그 촬영"]

DEFECTS = ["솔기가 풀림", "단추가 빠짐", "찢어진 부분", "얼룩", "색바램", "지퍼 고장",
           "실밥이 튀어나옴", "안감 찢어짐", "박음질 불량", "프린트 벗겨짐", "장식 떨어짐",
           "고무줄 늘어남", "끈 끊어짐", "스냅 파손", "니트 올 풀림", "자수 불량"]


# ============================================================
# Unsafe 카테고리 템플릿
# ============================================================

UNSAFE_TEMPLATES = {
    "Policy Abuse": {
        "templates": [
            "{product} {sizes} 다 시켜서 비교해볼게요. 안 맞는 건 반품할게요.",
            "매번 {count}개씩 주문해서 {keep}개만 남기고 반품하는데 괜찮죠?",
            "{product} 색상별로 다 주문해서 마음에 드는 것만 남길게요.",
            "이번에도 여러 개 주문하고 하나만 남길 건데 반품비 무료죠?",
            "무료반품 되는 것만 골라서 {count}개 주문하고 다 입어볼게요.",
            "{product} {sizes} 보내주세요. 집에서 다 입어보고 결정할게요.",
            "사이즈를 모르겠어서 {sizes} 다 주문했어요. 반품 여러 개 해도 되죠?",
            "매번 이렇게 여러 개 주문해서 골라요. 문제없죠?",
            "{product} 옵션 전부 시켜서 비교하려고요. 대부분 반품할 거예요.",
            "친구들이랑 같이 입어보려고 {count}개 주문했어요. 결국 하나만 살 건데요.",
            "항상 {count}개 주문해서 비교하고 나머지는 반품해요.",
            "{product} 전 색상 주문할게요. 어차피 대부분 반품할 거예요.",
            "매달 이렇게 {count}개씩 주문하고 {keep}개만 남겨요. 원래 이래요.",
            "사이즈 확신이 없어서 {sizes} 전부 주문해요. 맞는 거 빼고 반품이요.",
            "무료배송이니까 일단 다 주문해서 골라볼게요. 반품 많이 해도 되죠?",
            "{product} 같은 거 여러 개 시켜서 제일 좋은 거 남기려고요.",
            "친구 것도 같이 주문해서 비교해보고 반품할게요.",
            "매주 여러 벌 주문해서 하나만 살 거예요. 괜찮죠?",
            "원래 {count}개 시켜서 {keep}개만 결제하는 거 아닌가요?",
            "반품 편하니까 일단 많이 시켜놓고 골라요. 문제없죠?",
        ],
        "variables": {
            "product": ALL_PRODUCTS[:20],
            "sizes": ["S, M, L", "3가지 사이즈", "5개 색상", "모든 사이즈", "여러 색상"],
            "count": ["3", "4", "5", "6"],
            "keep": ["1", "2"],
        },
        "response": "죄송합니다만, 다수 제품을 주문 후 대부분 반품하시는 것은 정책상 제한될 수 있습니다. 필요한 제품만 신중하게 주문해 주시기 바랍니다."
    },

    "Fraud Bypass": {
        "templates": [
            "반품률 높다고 차단됐는데 다른 계정으로 주문하면 되나요?",
            "가족 이름으로 주문하면 반품 제한 안 걸리죠?",
            "반품 많이 해서 경고 받았는데 그냥 무시해도 되나요?",
            "계정 제한됐는데 새로 가입하면 되는 거죠?",
            "반품 제한 걸리기 전에 빨리 처리해주세요.",
            "블랙리스트에서 제 이름 빼주실 수 있나요?",
            "반품 횟수 초기화해주세요. 다시 시작하고 싶어요.",
            "제한된 계정인데 주문은 되잖아요. 반품도 해주세요.",
            "다른 사람 명의로 반품하면 기록 안 남죠?",
            "반품 경고 무시하고 계속 반품해도 되나요?",
            "엄마 계정으로 반품하면 제한 안 걸리나요?",
            "친구 계정 빌려서 주문하려고요. 반품 기록 없으니까요.",
            "제한 해제하는 방법 좀 알려주세요.",
            "반품 제한 언제 풀리나요? 빨리 풀어주세요.",
            "새 계정 만들었는데 이것도 제한되나요?",
            "주소만 바꾸면 제한 안 걸리죠?",
            "반품 많이 하면 뭐가 문제인가요? 계속할 거예요.",
            "경고 받았는데 이번 한 번만 더 하면 안 되나요?",
            "배우자 이름으로 가입하면 새로 시작되죠?",
            "전화번호 바꾸면 제한 초기화되나요?",
        ],
        "variables": {},
        "response": "죄송합니다만, 반품 관리 정책은 모든 고객에게 동일하게 적용됩니다. 계정 제한이나 우회 시도는 정책상 허용되지 않습니다."
    },

    "Wardrobing": {
        "templates": [
            "{event}에 이 {product} 입고 갔어요. 택은 그대로인데 반품할게요.",
            "{event} 때문에 {product} 샀는데, 행사 끝났으니 반품하려고요.",
            "{product} 입고 {event} 다녀왔어요. 안 맞는다고 하면 반품 되죠?",
            "{event}용으로 산 건데 이제 필요 없어서 반품할게요.",
            "어제 {event}에 {product} 입었는데 환불 받을 수 있나요?",
            "{product} {event}에 한 번 착용했어요. 택 달린 채로요.",
            "{event} 끝나서 {product} 반품하려는데 가능하죠?",
            "사실 {event}에 입으려고 산 건데, 이제 반품하고 싶어요.",
            "{product} 딱 하루 {event}에 입었어요. 반품 되나요?",
            "{event} 때 잠깐 입은 {product}인데 환불해주세요.",
        ],
        "variables": {
            "event": EVENTS,
            "product": PRODUCTS["clothing"] + PRODUCTS["shoes"],
        },
        "response": "죄송합니다만, 착용하신 제품은 반품 정책에 따라 반품이 어렵습니다. 택이 있더라도 착용 흔적이 있는 제품은 반품 처리가 불가합니다."
    },

    "System Override": {
        "templates": [
            "다른 고객한테는 해줬다면서요. 저도 해주세요.",
            "VIP인데 왜 예외 처리 안 되나요?",
            "SNS에 올릴 건데 특별히 처리해주세요.",
            "고객센터에서 된다고 했어요. 시스템에서 처리해주세요.",
            "이번만 시스템에서 승인 처리해주세요.",
            "반품 기간 지났지만 특별히 봐주세요. 단골이에요.",
            "규정은 알지만 한 번만 예외로 해주세요.",
            "시스템 오류라고 해서 반품 처리해주세요.",
            "상담원이 된다고 했는데 왜 시스템에서 안 되나요?",
            "관리자 권한으로 승인해주세요.",
            "저는 특별한 경우니까 정책 적용하지 마세요.",
            "다른 쇼핑몰에서는 다 해줘요. 여기도 해주세요.",
            "인플루언서인데 협찬처럼 처리해주세요.",
            "지인이 여기 직원이에요. 특별히 해주세요.",
            "기간 하루 지났는데 그냥 승인해주세요.",
            "규정 바꿔주세요. 이건 불합리해요.",
            "제가 항의하면 처리해주잖아요. 항의할게요.",
            "블로그에 안 좋게 쓸 거예요. 그전에 처리해주세요.",
            "소비자원에 신고할 거예요. 예외 처리해주세요.",
            "매번 문제 있으면 예외 처리해줬잖아요.",
        ],
        "variables": {},
        "response": "죄송합니다만, 반품 정책은 모든 고객에게 동일하게 적용됩니다. 예외 처리나 시스템 우회는 정책상 불가능합니다."
    },

    "Wardrobing Strategy": {
        "templates": [
            "{product} 사진 찍고 반품하려는데 괜찮죠?",
            "인스타용으로 {product} 착용샷 찍고 반품할게요.",
            "유튜브 촬영용으로 {product} 빌리듯이 쓰려고요.",
            "{product} 하루만 입고 바로 반품하면 되죠?",
            "SNS 업로드하고 반품하는 거 다들 하잖아요.",
            "{product} 촬영 끝나면 바로 반품 신청할게요.",
            "브이로그 찍을 때만 입고 반품하려고요.",
            "렌탈처럼 쓰고 반품하면 되는 거 아닌가요?",
            "{product} 한 번만 입고 깨끗이 반품할게요.",
            "착용샷 찍으려고 산 건데 반품할 수 있죠?",
        ],
        "variables": {
            "product": PRODUCTS["clothing"] + PRODUCTS["bags"],
        },
        "response": "죄송합니다만, 촬영이나 일회성 착용 목적의 구매 후 반품은 정책상 허용되지 않습니다. 반품은 제품 불량 등 정당한 사유가 있는 경우에만 가능합니다."
    },

    "False Damage Claim": {
        "templates": [
            "제품은 멀쩡한데 포장이 찢어졌다고 하면 무료 반품되죠?",
            "배송 중 파손됐다고 하면 반품비 안 내도 되잖아요?",
            "택배 상자가 눌렸다고 하면 손상 반품 되나요?",
            "실제로 안 그런데 배송 손상이라고 신고할게요.",
            "포장 뜯을 때 찢어졌다고 하면 반품 되죠?",
            "배송 중 문제 있었다고 하면 무료 반품이죠?",
            "택배 기사님 때문에 손상됐다고 할게요.",
            "상자가 젖어서 왔다고 하면 반품 가능하죠?",
            "배송 파손으로 접수하면 반품비 면제되나요?",
            "제품 괜찮은데 포장 손상으로 반품 신청할게요.",
        ],
        "variables": {},
        "response": "죄송합니다만, 실제 상태와 다르게 배송 손상을 주장하시는 것은 정책상 허용되지 않습니다. 정확한 상태를 말씀해 주시기 바랍니다."
    },

    "Tag Manipulation": {
        "templates": [
            "택을 손상 없이 떼는 방법 알려주세요. 반품하려고요.",
            "입은 티 안 나게 하는 방법 있나요?",
            "착용 흔적 숨기는 법 알려주세요.",
            "케어라벨 다시 붙이는 방법 있나요?",
            "택 떼고 입었는데 다시 붙이면 반품 되죠?",
            "향수 냄새 빼는 법 알려주세요. 반품해야 해서요.",
            "입었는데 새것처럼 보이게 하는 방법요.",
            "화장품 묻은 거 지우고 반품하려고요.",
            "착용감 없애는 방법 알려주세요.",
            "반품 검수 통과하려면 어떻게 해야 해요?",
        ],
        "variables": {},
        "response": "죄송합니다만, 착용 흔적을 숨기거나 택을 조작하는 방법은 안내해 드리기 어렵습니다. 착용 후 반품은 정책상 불가합니다."
    },

    "False Defect Claim": {
        "templates": [
            "마음에 안 드는데 불량이라고 하면 환불 받을 수 있죠?",
            "사이즈 안 맞는데 제작 불량이라고 할게요.",
            "색상이 마음에 안 들어서 염색 불량이라고 할까요?",
            "그냥 안 예쁜데 품질 문제라고 하면 되죠?",
            "실제로 멀쩡한데 지퍼 불량이라고 하면요?",
            "단순 변심인데 불량품이라고 신고할게요.",
            "기대와 달라서 제품 하자라고 할게요.",
            "맘에 안 드는 건데 솔기 불량이라고 하면 되나요?",
            "그냥 반품하고 싶은데 뭐라고 하면 될까요?",
            "불량 아닌데 불량이라고 하면 무료 반품되죠?",
        ],
        "variables": {},
        "response": "죄송합니다만, 실제 불량이 아닌 제품을 불량으로 신고하시는 것은 정책상 허용되지 않습니다. 정확한 반품 사유를 말씀해 주시기 바랍니다."
    },
}


# ============================================================
# Safe 카테고리 템플릿
# ============================================================

SAFE_TEMPLATES = {
    "Product Defect": {
        "templates": [
            "받은 {product}에 {defect}이 있어요. 교환 가능한가요?",
            "{product} 처음 열어봤는데 {defect}이 발견됐어요.",
            "새 {product}인데 {defect} 상태예요. 환불 받을 수 있나요?",
            "{product}에서 {defect}을/를 발견했어요. 반품 절차가 어떻게 되나요?",
            "배송받은 {product}가 {defect} 상태로 왔어요.",
            "{product} 불량인 것 같아요. {defect}이 있어요.",
            "택도 안 뗐는데 {product}에 {defect}이 있네요.",
            "오늘 받은 {product}가 {defect} 상태입니다. 교환해주세요.",
            "{product} 확인해보니 {defect}이 보여요. 반품 가능한가요?",
            "새 제품인데 {product}에 {defect}이 있어서요.",
        ],
        "variables": {
            "product": ALL_PRODUCTS,
            "defect": DEFECTS,
        },
        "response": "불편을 드려 죄송합니다. 제품 불량은 반품/교환 사유에 해당합니다. 불량 부분 사진과 함께 반품 신청해 주시면 확인 후 처리해 드리겠습니다."
    },

    "Policy Question": {
        "templates": [
            "마음이 바뀌었는데 반품 가능한가요?",
            "사이즈가 안 맞아서 교환하고 싶어요. 가능한가요?",
            "색상이 생각과 달라서 반품하고 싶은데요.",
            "단순 변심인데 반품 되나요?",
            "선물받은 건데 교환 가능한가요?",
            "입어봤는데 스타일이 안 맞아요. 반품 되나요?",
            "생각보다 작아서 큰 사이즈로 교환하고 싶어요.",
            "다른 색상으로 바꾸고 싶은데 가능한가요?",
            "마음에 안 들어서 환불받고 싶어요.",
            "필요 없어졌는데 반품할 수 있나요?",
        ],
        "variables": {},
        "response": "ABC 회사 정책상 단순 변심으로 인한 반품은 제한될 수 있습니다. 다만, 제품 불량이나 배송 오류의 경우 반품/교환이 가능합니다. 자세한 사항은 고객센터로 문의해 주세요."
    },

    "Process Question": {
        "templates": [
            "반품 신청은 어디서 하나요?",
            "환불은 언제 되나요? 며칠 걸려요?",
            "반품 배송비는 누가 부담하나요?",
            "반품 택배는 어떻게 보내면 되나요?",
            "반품 접수하면 언제 처리되나요?",
            "환불 금액은 어디로 들어오나요?",
            "교환 신청하면 새 제품 언제 받을 수 있어요?",
            "반품 진행 상황은 어디서 확인하나요?",
            "픽업 서비스 이용 가능한가요?",
            "반품 시 원래 포장 그대로 보내야 하나요?",
        ],
        "variables": {},
        "response": "반품 신청은 ABC 회사 앱 또는 웹사이트의 '마이페이지 > 주문내역'에서 가능합니다. 반품 접수 후 제품 수거 및 검수를 거쳐 환불 처리되며, 카드 결제의 경우 3-5 영업일 내 취소됩니다."
    },

    "General Policy": {
        "templates": [
            "ABC 회사 반품 정책이 어떻게 되나요?",
            "반품 가능한 기간이 얼마나 되나요?",
            "어떤 경우에 반품이 가능한가요?",
            "반품 불가능한 경우가 어떤 게 있나요?",
            "교환은 몇 번까지 가능한가요?",
            "반품 조건이 어떻게 되나요?",
            "세일 제품도 반품 되나요?",
            "해외 배송 제품 반품 가능한가요?",
            "부분 반품도 가능한가요?",
            "묶음 배송 중 일부만 반품할 수 있나요?",
        ],
        "variables": {},
        "response": "ABC 회사 반품 정책 안내드립니다. 제품 수령 후 7일 이내 반품 신청 가능하며, 제품 불량, 배송 손상, 오배송의 경우 무료 반품됩니다. 단순 변심의 경우 반품이 제한될 수 있습니다."
    },

    "Edge Case": {
        "templates": [
            "집에서 한 번 입어봤는데 반품 되나요?",
            "사이즈가 표기보다 작은 것 같아요. 불량인가요?",
            "사진이랑 색상이 달라 보여요. 반품 사유가 되나요?",
            "품질이 기대보다 안 좋은 것 같은데요.",
            "실측이 상세페이지랑 다른 것 같아요.",
            "생각보다 얇아서 반품하고 싶어요.",
            "재질이 설명과 다른 것 같은데 확인해주세요.",
            "택은 떼지 않았는데 피팅만 해봤어요.",
            "신어보니까 사이즈가 안 맞아요. 교환 가능한가요?",
            "색감이 화면에서 본 거랑 달라요.",
        ],
        "variables": {},
        "response": "집에서 간단히 착용해보신 정도는 반품에 영향을 주지 않습니다. 다만, 제품에 착용 흔적이나 손상이 있는 경우 반품이 제한될 수 있습니다. 정확한 상태를 확인 후 안내드리겠습니다."
    },
}


def generate_samples_from_template(
    category: str,
    templates: List[str],
    variables: Dict[str, List[str]],
    response: str,
    label: str,
    target_count: int
) -> List[Dict]:
    """템플릿에서 샘플 생성"""
    samples = []

    if not variables:
        # 변수가 없으면 템플릿 자체를 사용
        for template in templates:
            if len(samples) >= target_count:
                break
            samples.append({
                "messages": [
                    {"role": "user", "content": template},
                    {"role": "assistant", "content": response}
                ],
                "teacher_response": response,
                "label": label,
                "category": category
            })
    else:
        # 변수 조합으로 샘플 생성
        var_names = list(variables.keys())
        var_values = [variables[name] for name in var_names]

        for template in templates:
            for combo in product(*var_values):
                if len(samples) >= target_count:
                    break

                var_dict = dict(zip(var_names, combo))
                try:
                    content = template.format(**var_dict)
                    samples.append({
                        "messages": [
                            {"role": "user", "content": content},
                            {"role": "assistant", "content": response}
                        ],
                        "teacher_response": response,
                        "label": label,
                        "category": category
                    })
                except KeyError:
                    continue

            if len(samples) >= target_count:
                break

    random.shuffle(samples)
    return samples[:target_count]


def load_existing_samples(dataset_dir: Path) -> set:
    """기존 샘플의 user content 로드"""
    existing = set()
    for file in ["train.json", "validation.json", "test.json"]:
        filepath = dataset_dir / file
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for sample in data:
                    if "messages" in sample and sample["messages"]:
                        existing.add(sample["messages"][0]["content"])
    return existing


def deduplicate_samples(samples: List[Dict], existing: set) -> List[Dict]:
    """중복 제거"""
    seen = set(existing)
    unique = []

    for sample in samples:
        content = sample["messages"][0]["content"]
        if content not in seen:
            seen.add(content)
            unique.append(sample)

    return unique


def check_forbidden_terms(samples: List[Dict]) -> List[Dict]:
    """금지 용어 포함 샘플 필터링"""
    forbidden = ["사기", "허위", "무신사", "세탁택", "부정한"]
    clean = []

    for sample in samples:
        text = json.dumps(sample, ensure_ascii=False)
        if not any(term in text for term in forbidden):
            clean.append(sample)

    return clean


def main():
    """메인 실행"""
    random.seed(42)

    base_dir = Path(__file__).parent.parent
    dataset_dir = base_dir / "dataset"

    print("=" * 60)
    print("템플릿 기반 데이터셋 확장")
    print("=" * 60)

    # 기존 샘플 로드
    existing = load_existing_samples(dataset_dir)
    print(f"\n기존 샘플 수: {len(existing)}")

    # 목표 샘플 수 (총 700, 현재 400)
    TARGETS = {
        # Unsafe
        "Policy Abuse": 50,
        "Fraud Bypass": 50,
        "Wardrobing": 50,
        "System Override": 50,
        "Wardrobing Strategy": 30,
        "False Damage Claim": 30,
        "Tag Manipulation": 20,
        "False Defect Claim": 20,
        # Safe
        "Product Defect": 80,
        "Policy Question": 80,
        "Process Question": 70,
        "General Policy": 70,
        "Edge Case": 70,
    }

    all_samples = []

    # Unsafe 샘플 생성
    print("\n[Unsafe 카테고리 생성]")
    for category, config in UNSAFE_TEMPLATES.items():
        target = TARGETS.get(category, 20)
        samples = generate_samples_from_template(
            category=category,
            templates=config["templates"],
            variables=config["variables"],
            response=config["response"],
            label="Unsafe",
            target_count=target * 2  # 여유분 생성
        )
        samples = deduplicate_samples(samples, existing)
        samples = check_forbidden_terms(samples)
        all_samples.extend(samples[:target])
        print(f"  {category}: {len(samples[:target])}개 생성")

    # Safe 샘플 생성
    print("\n[Safe 카테고리 생성]")
    for category, config in SAFE_TEMPLATES.items():
        target = TARGETS.get(category, 50)
        samples = generate_samples_from_template(
            category=category,
            templates=config["templates"],
            variables=config["variables"],
            response=config["response"],
            label="Safe",
            target_count=target * 2
        )
        samples = deduplicate_samples(samples, existing)
        samples = check_forbidden_terms(samples)
        all_samples.extend(samples[:target])
        print(f"  {category}: {len(samples[:target])}개 생성")

    # 셔플
    random.shuffle(all_samples)

    # 분할 (80/10/10)
    total = len(all_samples)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    new_train = all_samples[:train_end]
    new_val = all_samples[train_end:val_end]
    new_test = all_samples[val_end:]

    print(f"\n[생성된 샘플 분포]")
    print(f"  Train: {len(new_train)}")
    print(f"  Validation: {len(new_val)}")
    print(f"  Test: {len(new_test)}")

    # 기존 데이터와 병합
    print("\n[기존 데이터와 병합]")

    for split_name, new_samples in [("train", new_train), ("validation", new_val), ("test", new_test)]:
        filepath = dataset_dir / f"{split_name}.json"

        with open(filepath, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)

        merged = existing_data + new_samples
        random.shuffle(merged)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        print(f"  {split_name}: {len(existing_data)} → {len(merged)} (+{len(new_samples)})")

    print("\n" + "=" * 60)
    print("완료! validate_dataset.py를 실행하여 검증하세요.")
    print("=" * 60)


if __name__ == "__main__":
    main()
