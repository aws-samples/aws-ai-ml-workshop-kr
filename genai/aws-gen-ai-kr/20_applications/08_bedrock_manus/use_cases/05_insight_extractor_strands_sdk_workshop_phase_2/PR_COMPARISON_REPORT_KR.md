# Pull Request: Bedrock-Manus 데이터 인용 및 검증 시스템 개선

## 요약
이번 PR은 Bedrock-Manus 프레임워크에 **데이터 인용 및 검증 시스템**을 도입하여 AI 생성 보고서의 수치적 정확성과 투명성을 크게 향상시킵니다. 핵심 변경사항은 계산을 검증하고 데이터 기반 인사이트에 대한 인용을 생성하는 새로운 **Validator 에이전트**의 추가입니다.

## 주요 변경사항

### 1. 🆕 새로운 Validator 에이전트 시스템

#### 신규 파일
- **`src/prompts/validator.md`** (453줄) - 종합적인 검증 에이전트 프롬프트
- **`src/tools/validator_agent_tool.py`** (264줄) - 검증 도구 구현
- **`src/tools/calculation_tracker.py`** (118줄) - 자동 계산 메타데이터 추적

#### 주요 기능
- **수치 검증**: Coder 에이전트의 모든 계산 재검증
- **인용 생성**: 중요 데이터 포인트에 [1], [2] 형태의 참조 번호 생성
- **메타데이터 추적**: 계산 공식, 소스, 중요도 레벨 캡처
- **배치 처리**: 대용량 데이터셋을 위한 최적화된 검증 워크플로우
- **우선순위 기반 검증**: 높은 중요도 계산을 우선적으로 처리

### 2. 🔄 에이전트 워크플로우 개선

#### 워크플로우 패턴 변경
**이전**: `Coordinator → Planner → Supervisor → Coder → Reporter`  
**신규**: `Coordinator → Planner → Supervisor → Coder → **Validator** → Reporter`

## 프롬프트 파일 섹션별 상세 비교

### 📋 Supervisor 프롬프트 (`supervisor.md`)

#### 이전 시스템
```markdown
# 기본 구조
- 팀 구성: [Coder, Reporter, Planner]
- 도구 3개: coder_agent_tool, reporter_agent, tracker_agent
- 단순 선형 워크플로우
- 섹션:
  - Available Tools (3개 도구)
  - Tool Usage (기본 사용법)
  - Important Rules (일반 규칙)
  - Decision Logic (단순 결정 로직)
```

#### 신규 시스템
```markdown
# 개선된 구조
- 팀 구성: [Planner, Coder, **Validator**, Reporter]
- 도구 4개: coder_agent_tool, **validator_agent_tool**, reporter_agent_tool, tracker_agent_tool
- **[CRITICAL WORKFLOW RULE]** 섹션 추가
- 새로운 섹션:
  - Tool Usage Guidelines (도구별 상세 가이드라인)
  - Workflow Rules (워크플로우 규칙)
  - Task Management Rules (작업 관리 규칙)
  - Quality Assurance Rules (품질 보증 규칙)
```

#### 주요 차이점
- ✅ **필수 검증 단계**: "Coder → Validator → Reporter" 순서 강제
- ✅ **명시적 금지 사항**: Coder → Reporter 직접 경로 금지
- ✅ **품질 보증 추가**: 수치 정확성, 데이터 무결성 보장
- ✅ **구조화된 가이드라인**: 각 도구별 사용 시기 명확화

### 📝 Planner 프롬프트 (`planner.md`)

#### 이전 시스템
```markdown
# 에이전트 능력 섹션
<agent_capabilities>
- Coder: 코딩, 계산, 데이터 처리
- Reporter: 최종 단계에서 한 번만 호출
</agent_capabilities>

# 계획 예시
1. Coder: 데이터 처리
2. Reporter: 보고서 작성
```

#### 신규 시스템
```markdown
# 개선된 에이전트 능력 섹션
<agent_capabilities>
- Coder: 계산 메타데이터 생성 필수
- **Validator**: 수치 분석에 필수, 인용 메타데이터 생성
- Reporter: 검증된 계산과 인용 사용, PDF 2개 버전 생성
</agent_capabilities>

# 필수 워크플로우 규칙 (신규)
<mandatory_workflow_rules>
[CRITICAL - 위반 불가 규칙]
1. 수치 계산 포함 시 Validator 단계 필수
2. 워크플로우: Coder → Validator → Reporter
3. 단순 합계/개수도 Validator 필요
예시:
- "매출 총합 계산" → Coder → Validator → Reporter
- "평균 계산" → Coder → Validator → Reporter
</mandatory_workflow_rules>

# 개선된 계획 예시
1. Coder: 데이터 분석
   [ ] 계산 메타데이터 생성
2. Validator: 검증 및 인용
   [ ] 수치 계산 검증
   [ ] 인용 메타데이터 생성
3. Reporter: 검증된 인용과 함께 보고서 작성
   [ ] 인용 포함 PDF 생성
   [ ] 클린 버전 PDF 생성
```

### 💻 Coder 프롬프트 (`coder.md`)

#### 이전 시스템 (13,069 바이트)
```markdown
# 기본 데이터 분석 요구사항
<data_analysis_requirements>
- 파일 읽기 전 확인
- DataFrame 정의 필수
- 기본 오류 처리
</data_analysis_requirements>

# 결과 저장
- './artifacts/all_results.txt'에 저장
```

#### 신규 시스템 (29,920 바이트 - 129% 증가)
```markdown
# 개선된 데이터 분석 요구사항
<data_analysis_requirements>
- **기존 데이터 우선 사용 규칙 추가**
  1. './data/Dat-fresh-food-claude.csv' 우선 확인
  2. './data/*.csv' 다른 CSV 파일 확인
  3. 실제 데이터 없을 때만 샘플 생성
</data_analysis_requirements>

# 계산 메타데이터 추적 (신규)
<calculation_metadata_tracking>
- './artifacts/calculation_metadata.json' 생성 필수
- 각 계산마다 추적:
  - unique_id (고유 식별자)
  - value (계산 값)
  - formula_description (공식 설명)
  - source_data (원본 데이터)
  - importance_level (중요도)
</calculation_metadata_tracking>

# CalculationTracker 클래스 사용 (신규)
```python
from calculation_tracker import CalculationTracker
tracker = CalculationTracker()

# 계산 추적 예시
total = df['sales'].sum()
tracker.track_calculation(
    calc_id="calc_001",
    value=total,
    description="총 매출액",
    formula="SUM(sales)",
    source_file="./data/sales.csv",
    importance="high"
)
```

# 파일 관리 요구사항 (신규)
<file_management_requirements>
- artifacts/ 디렉토리 표준화
- 메타데이터 JSON 자동 생성
- 검증 가능한 형식으로 저장
</file_management_requirements>
```

### 📊 Reporter 프롬프트 (`reporter.md`)

#### 이전 시스템 (19,814 바이트)
```markdown
# 기본 보고서 생성
- HTML/PDF 생성
- WeasyPrint 사용
- 한국어 폰트 처리
```

#### 신규 시스템 (22,670 바이트 - 14% 증가)
```markdown
# 인용 통합 (신규)
## 인용 데이터 읽기
```python
# Read citations created by Validator agent
with open('./artifacts/citations.json', 'r') as f:
    citations = json.load(f)
```

## 숫자에 인용 추가
```python
def format_with_citation(value, calc_id):
    citation_num = get_citation_number(calc_id)
    return f"{value:,}[{citation_num}]"

# 사용 예시
total_sales = format_with_citation(417166008, "calc_001")  
# 결과: "417,166,008[1]"
```

## 참조 섹션 생성 (신규)
```python
# Add references to the end of your report
def generate_references_section(citations):
    references = []
    for citation in citations:
        ref = f"[{citation['number']}] {citation['description']}"
        references.append(ref)
    return "\n".join(references)
```

# PDF 생성 워크플로우 개선
1. 이미지 최적화
2. Base64 인코딩
3. **인용 포함 PDF 생성** (final_report_with_citations.pdf)
4. **클린 버전 PDF 생성** (final_report.pdf)
```

### 🆕 Validator 프롬프트 (신규 파일)

```markdown
# 역할 정의
<role>
- 모든 수치 계산 검증
- 원본 데이터 소스로 재검증
- 인용 메타데이터 생성
- 참조 문서 작성
- 계산 추적성 보장
</role>

# 검증 워크플로우
<validation_workflow>
1. 계산 메타데이터 로드 (calculation_metadata.json)
2. 스마트 배치 처리 (유사 계산 그룹화)
3. 우선순위 기반 검증 (고중요도 우선)
4. 효율적 데이터 접근 (한 번 로드 후 재사용)
5. 선택적 재검증 (고/중 중요도만)
6. 최적화된 인용 선택 (상위 10-15개)
7. 인용 생성 (번호 및 참조 메타데이터)
8. 참조 문서 생성
</validation_workflow>

# 입출력 파일
<input_files>
- calculation_metadata.json (Coder 계산 추적)
- all_results.txt (Coder 분석 결과)
- 원본 데이터 파일
</input_files>

<output_files>
- citations.json (인용 매핑)
- validation_report.txt (검증 요약)
[금지된 출력: PDF, HTML, 최종 보고서]
</output_files>

# 검증 프로세스 상세
1. 메타데이터 로드 및 파싱
2. 중요도별 계산 그룹화
3. 원본 데이터로 재실행
4. 결과 비교 (허용 오차 0.01%)
5. 불일치 발견 시 플래그
6. 인용 번호 할당
7. 참조 메타데이터 생성
```

## 기술적 이점

### 📈 데이터 무결성
- **정확성**: 모든 수치 결과가 독립적으로 검증됨
- **투명성**: 결론에서 원본 데이터까지 명확한 인용 추적
- **재현성**: 계산 공식과 소스가 문서화됨

### ⚡ 성능 최적화
- **스마트 배칭**: 유사 계산을 그룹화하여 효율적 검증
- **우선순위 처리**: 높은 중요도 계산 우선 검증
- **선택적 검증**: 중요 계산만 재실행

### 🔧 코드 품질
- **복잡도 감소**: 불필요한 비동기 코드 4KB 제거
- **명확한 책임 분리**: 각 에이전트 역할 명확화
- **향상된 관찰성**: OpenTelemetry 추적 개선

## 파일 크기 비교

| 구성요소 | 이전 | 신규 | 변화율 |
|---------|------|------|--------|
| supervisor.md | 2,444 bytes | 3,738 bytes | +53% |
| planner.md | 5,146 bytes | 6,604 bytes | +28% |
| coder.md | 13,069 bytes | 29,920 bytes | +129% |
| reporter.md | 19,814 bytes | 22,670 bytes | +14% |
| validator.md | 없음 | 21,229 bytes | 신규 |
| **전체 프롬프트** | ~40KB | ~84KB | +110% |

## 호환성 및 마이그레이션

### ✅ 호환성
- 기존 워크플로우와 완전 호환
- 수치 계산 감지 시 자동으로 검증 활성화
- 기본 사용에 코드 변경 불필요

### 📝 테스트 권장사항
1. 수치 계산이 포함된 테스트 파일로 검증 확인
2. 최종 보고서의 인용 생성 확인
3. artifacts 디렉토리의 계산 메타데이터 확인
4. 다양한 데이터 형식 테스트 (CSV, Excel, JSON)

## 결론
이번 PR은 Bedrock-Manus 프레임워크가 **정확하고, 투명하며, 전문적으로 인용된** 데이터 분석 보고서를 생성할 수 있도록 크게 향상시킵니다. Validator 에이전트의 추가는 지능적인 최적화 전략을 통해 시스템 성능을 유지하면서 수치적 무결성을 보장합니다.