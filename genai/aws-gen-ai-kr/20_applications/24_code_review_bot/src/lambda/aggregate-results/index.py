import json
import os
from typing import Dict, List, Any, Union
from dataclasses import dataclass
from collections import defaultdict
from decimal import Decimal
import boto3
from datetime import datetime

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

def convert_decimal(obj):
    """DynamoDB Decimal 타입을 float로 변환"""
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimal(v) for v in obj]
    return obj


@dataclass
class ReviewSummary:
    total_files: int
    total_primary_files: int
    total_reference_files: int
    total_issues: int
    severity_counts: Dict[str, int]
    category_counts: Dict[str, int]
    critical_issues: List[Dict[str, Any]]
    major_issues: List[Dict[str, Any]]
    suggestions_by_file: Dict[str, List[Dict[str, Any]]]
    reference_context: Dict[str, List[str]]
    # 변경사항 요약
    functional_changes: List[str]
    architectural_changes: List[str]
    technical_improvements: List[str]
    # 이전 리뷰와의 비교
    previous_reviews_count: int = 0
    resolved_issues_count: int = 0
    new_issues_count: int = 0
    persistent_issues_count: int = 0
    # 모든 이슈 목록 추가
    all_issues: List[Dict[str, Any]] = None  # 기본값은 None
    
    def __post_init__(self):
        if self.all_issues is None:
            self.all_issues = []

class ResultAggregator:
    def __init__(self, event_data: Dict[str, Any]):
        self.ssm = boto3.client('ssm')
        self.event_data = event_data
        self.dynamodb = boto3.resource('dynamodb')
        self.results_table = self.dynamodb.Table('PRReviewerResults')
        self.reports_table = self.dynamodb.Table('PRReviewerReports')
        
        # 실행 ID 먼저 추출
        try:
            self.execution_id = self._extract_execution_id()
            print(f"Extracted execution_id: {self.execution_id}")
        except Exception as e:
            print(f"Error extracting execution ID: {e}")
            self.execution_id = f"error-{int(datetime.now().timestamp())}"
    
        # 그 다음 청크 결과 로드
        self.chunk_results = self._load_chunk_results_from_dynamodb()
        
        # 나머지 초기화
        self.pr_details = self._extract_pr_details()
        self.secrets = boto3.client('secretsmanager')
        self.config = self._load_config()
        self.previous_reviews = []
        
        # PR 정보가 있는 경우 이전 리뷰 로드
        if self.pr_details and 'repository' in self.pr_details and 'pr_id' in self.pr_details:
            self.previous_reviews = self._get_previous_reviews(
                self.pr_details['repository'], 
                self.pr_details['pr_id']
            )


    def _load_chunk_results_from_dynamodb(self) -> List[Dict[str, Any]]:
        """DynamoDB에서 청크 결과 로드"""
        results = []
        try:
            # 실행 ID가 설정되었는지 확인
            if not hasattr(self, 'execution_id') or not self.execution_id:
                print("Execution ID not set, cannot load chunk results")
                return []

            # 현재 실행에 대한 모든 청크 결과 조회
            response = self.results_table.query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key('execution_id').eq(self.execution_id)
            )

            if 'Items' in response:
                for item in response['Items']:
                    if 'results' in item:  # 결과가 있는 항목만 처리
                        results.extend(item['results'])

            print(f"Loaded {len(results)} chunk results from DynamoDB")
            return results
        except Exception as e:
            print(f"Error loading chunk results from DynamoDB: {e}")
            return []

    def _extract_execution_id(self) -> str:
        """이벤트에서 실행 ID 추출"""
        try:
            # Step Functions 맵 상태에서 오는 결과 목록에서 실행 ID 추출
            if isinstance(self.event_data, dict):
                classified_results = self.event_data.get('classifiedResults', {})
                succeeded_results = classified_results.get('succeeded', [])
                retry_results = self.event_data.get('retryResults', [])

                all_results = succeeded_results + retry_results

                if all_results:
                    for result in all_results:
                        if isinstance(result, dict) and result.get('body'):
                            body = json.loads(result['body']) if isinstance(result['body'], str) else result['body']
                            if execution_id := body.get('execution_id'):
                                return execution_id

                # 실행 ID를 찾지 못한 경우 임시 ID 생성
                temp_id = f"unknown-{int(datetime.now().timestamp())}"
                print(f"Could not find execution_id in event data, using generated ID: {temp_id}")
                return temp_id

            else:
                temp_id = f"unknown-format-{int(datetime.now().timestamp())}"
                print(f"Event data is not a dictionary, using generated ID: {temp_id}")
                return temp_id

        except Exception as e:
            temp_id = f"error-{int(datetime.now().timestamp())}"
            print(f"Error extracting execution ID: {e}, using generated ID: {temp_id}")
            return temp_id

    def _load_config(self) -> Dict[str, Any]:
        """Parameter Store에서 설정 로드"""
        config = {}
        try:
            # 기본 설정 로드
            response = self.ssm.get_parameters_by_path(
                Path='/pr-reviewer/config/',
                Recursive=True,
                WithDecryption=True
            )
            
            for param in response['Parameters']:
                # 파라미터 이름에서 마지막 부분만 추출
                name = param['Name'].split('/')[-1]
                config[name] = param['Value']
                   
        except Exception as e:
            print(f"Error loading config: {e}")
            raise

        return config

    def _extract_pr_details(self) -> Dict[str, Any]:
        """PR 상세 정보 추출"""
        try:
            # DynamoDB에서 하나의 항목만 가져와 PR 상세 정보 추출
            response = self.results_table.query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key('execution_id').eq(self.execution_id),
                Limit=1
            )

            if response.get('Items'):
                pr_details = convert_decimal(response['Items'][0].get('pr_details', {}))
                print(f"Extracted PR details: {json.dumps(pr_details, cls=DecimalEncoder)}")
                return pr_details

            # DynamoDB에서 정보를 찾지 못한 경우 입력 이벤트에서 직접 추출 시도
            if isinstance(self.event_data, dict):
                pr_details = {}
                if 'body' in self.event_data and 'pr_details' in self.event_data['body']:
                    pr_details = self.event_data['body']['pr_details']

                if pr_details:
                    print(f"Extracted PR details from event data: {json.dumps(pr_details, cls=DecimalEncoder)}")
                    return pr_details

            print("Failed to extract PR details from DynamoDB or event data")
            return {}

        except Exception as e:
            print(f"Error extracting PR details: {e}")
            return {}

    def _extract_chunk_results(self) -> List[Dict[str, Any]]:
        """청크 결과 추출"""
        results = []
        try:
            if isinstance(self.event_data, list):
                # 병렬 처리 결과
                for chunk in self.event_data:
                    if isinstance(chunk, dict) and chunk.get('body'):
                        body = json.loads(chunk['body'])
                        if chunk_results := body.get('results'):
                            results.extend(chunk_results)
            elif isinstance(self.event_data, dict):
                # 단일 처리 결과
                if self.event_data.get('body'):
                    body = json.loads(self.event_data['body'])
                    if chunk_results := body.get('results'):
                        results.extend(chunk_results)
        except Exception as e:
            print(f"Error extracting chunk results: {e}")

        return results

    def _normalize_line_number(self, line_number: Union[str, int]) -> str:
        """라인 번호 정규화"""
        if isinstance(line_number, str) and line_number.lower() == 'all':
            return 'Throughout file'
        try:
            return str(int(line_number))
        except (ValueError, TypeError):
            return 'N/A'


    def _prepare_summary_prompt(self, changes: Dict[str, List[str]]) -> str:
        """Key Changes Summary 요약을 위한 프롬프트 준비"""
        prompt = """다음 변경사항들을 각 카테고리별로 5문장 이내로 요약해주세요.
        원본 변경사항:

        🔄 Functional Changes:
        """
        for change in changes.get('functional_changes', []):
            prompt += f"- {change}\n"

        prompt += "\n🏗 Architectural Changes:\n"
        for change in changes.get('architectural_changes', []):
            prompt += f"- {change}\n"

        prompt += "\n🔧 Technical Improvements:\n"
        for change in changes.get('technical_improvements', []):
            prompt += f"- {change}\n"

        prompt += """
        위 변경사항들을 다음 형식으로 요약해주세요:

            {
                "summary": {
                    "functional_changes": "기능적 변경사항 요약",
                    "architectural_changes": "아키텍처 변경사항 요약",
                    "technical_improvements": "기술적 개선사항 요약"
                }
            }

            각 요약은 한글로 작성하고, 전문 용어나 고유명사는 원문 그대로 사용해주세요."""
        print(prompt)
        return prompt

    def _summarize_changes_with_bedrock(self, changes: Dict[str, List[str]]) -> Dict[str, str]:
        """Bedrock을 사용하여 변경사항 요약"""
        try:
            bedrock = boto3.client('bedrock-runtime')
            prompt = self._prepare_summary_prompt(changes)

            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "temperature": 0.7,
                "top_p": 0.9,
                "system": "5문장 이내로 간결하게 요약하는 전문 리뷰어입니다.",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })

            response = bedrock.invoke_model(
                modelId=self.config['model'],
                contentType='application/json',
                accept='application/json',
                body=body.encode()
            )

            response_body = json.loads(response['body'].read())
            summary = json.loads(response_body['content'][0]['text'])
            return summary.get('summary', {})

        except Exception as e:
            print(f"Error summarizing with Bedrock: {e}")
            return {
                'functional_changes': '',
                'architectural_changes': '',
                'technical_improvements': ''
            }

    def analyze_results(self) -> ReviewSummary:
        """리뷰 결과 분석"""
        severity_counts = defaultdict(int)
        category_counts = defaultdict(int)
        critical_issues = []
        major_issues = []
        suggestions_by_file = defaultdict(list)
        reference_context = defaultdict(list)
        total_issues = 0

        # primary/reference 파일 구분
        primary_files = []
        reference_files = []
        all_issues = []

        for result in self.chunk_results:
            file_path = result['file_path']

            if result.get('is_primary', True):
                primary_files.append(file_path)
                severity_counts[result['severity']] += 1

                # 참조 파일 정보 저장
                if referenced_by := result.get('referenced_by'):
                    reference_context[file_path].extend(referenced_by)

                for suggestion in result.get('suggestions', []):
                    total_issues += 1
                    category = suggestion.get('category', 'other')
                    severity = suggestion.get('severity', 'NORMAL')

                    category_counts[category] += 1

                    # 라인 번호 정규화
                    suggestion['line_number'] = self._normalize_line_number(
                        suggestion.get('line_number')
                    )

                    issue_details = {
                        'file': file_path,
                        'description': suggestion.get('description'),
                        'line_number': suggestion['line_number'],
                        'suggestion': suggestion.get('suggestion'),
                        'severity': severity,
                        'category': category
                    }

                    all_issues.append(issue_details)  # 이슈를 all_issues 리스트에 추가

                    if severity == 'CRITICAL':
                        critical_issues.append(issue_details)
                    elif severity == 'MAJOR':
                        major_issues.append(issue_details)

                    suggestions_by_file[file_path].append(suggestion)
            else:
                reference_files.append(file_path)

        # 변경사항 요약 수집
        functional_changes = set()
        architectural_changes = set()
        technical_improvements = set()

        for result in self.chunk_results:
            if summary := result.get('summary', {}):
                functional_changes.update(summary.get('functional_changes', []))
                architectural_changes.update(summary.get('architectural_changes', []))
                technical_improvements.update(summary.get('technical_improvements', []))

        # 이전 리뷰와 비교
        comparison_result = self._compare_with_previous_reviews(all_issues)

        return ReviewSummary(
            total_files=len(primary_files) + len(reference_files),
            total_primary_files=len(primary_files),
            total_reference_files=len(reference_files),
            total_issues=total_issues,
            severity_counts=dict(severity_counts),
            category_counts=dict(category_counts),
            critical_issues=critical_issues,
            major_issues=major_issues,
            suggestions_by_file=dict(suggestions_by_file),
            reference_context=dict(reference_context),
            functional_changes=sorted(list(functional_changes)),
            architectural_changes=sorted(list(architectural_changes)),
            technical_improvements=sorted(list(technical_improvements)),
            previous_reviews_count=comparison_result['previous_reviews_count'],
            resolved_issues_count=len(comparison_result['resolved_issues']),
            new_issues_count=len(comparison_result['new_issues']),
            persistent_issues_count=len(comparison_result['persistent_issues']),
            all_issues=all_issues  # all_issues 필드에 저장
        )

    def generate_markdown_report(self, summary: ReviewSummary) -> str:
        pr_title = self.pr_details.get('title', 'Unknown PR')
        pr_author = self.pr_details.get('author', 'Unknown Author')
    
        report = [
            f"# 🧾 Code Review Report: {pr_title}",
            f"\nGenerated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    
            "\n## Overview",
            f"- Pull Request by: {pr_author}",
            f"- Primary Files Reviewed: {summary.total_primary_files}",
            f"- Reference Files: {summary.total_reference_files}",
            f"- Total Issues Found: {summary.total_issues}",
        ]
        
        # 이전 리뷰가 있는 경우 비교 정보 추가
        if summary.previous_reviews_count > 0:
            report.extend([
                f"- Previous Reviews: {summary.previous_reviews_count}",
                f"- Resolved Issues: {summary.resolved_issues_count}",
                f"- New Issues: {summary.new_issues_count}",
                f"- Persistent Issues: {summary.persistent_issues_count}"
            ])
    
        if summary.functional_changes or summary.architectural_changes or summary.technical_improvements:
            # 모든 변경사항 통합
            all_changes = {
                'functional_changes': summary.functional_changes,
                'architectural_changes': summary.architectural_changes,
                'technical_improvements': summary.technical_improvements
            }
        
            # Bedrock을 사용하여 요약
            summarized_changes = self._summarize_changes_with_bedrock(all_changes)
    
            report.extend([
                "\n## Key Changes Summary",
                "\n### 🔄 Functional Changes",
                summarized_changes.get('functional_changes', ''),
                "\n### 🏗 Architectural Changes",
                summarized_changes.get('architectural_changes', ''),
                "\n### 🔧 Technical Improvements",
                summarized_changes.get('technical_improvements', '')
            ])
        
        # 이전 리뷰 대비 변경 사항 (있는 경우)
        if summary.previous_reviews_count > 0:
            report.append("\n## Review History Analysis")
            
            # 해결된 이슈
            if summary.resolved_issues_count > 0:
                report.append("\n### ✅ Resolved Issues")
                comparison_result = self._compare_with_previous_reviews([])
                
                for issue in comparison_result['resolved_issues']:
                    report.extend([
                        f"\n- **{issue['file']}** (Line {issue['line_number']})",
                        f"  - {issue['description']}"
                    ])
                
                if len(comparison_result['resolved_issues']) < summary.resolved_issues_count:
                    report.append(f"\n... and {summary.resolved_issues_count - len(comparison_result['resolved_issues'])} more resolved issues.")
            
            # 지속적인 이슈
            if summary.persistent_issues_count > 0:
                report.append("\n### ⚠️ Persistent Issues")
                comparison_result = self._compare_with_previous_reviews([])
                
                for issue in comparison_result['persistent_issues']:
                    report.extend([
                        f"\n- **{issue['file']}** (Line {issue['line_number']})",
                        f"  - {issue['description']}"
                    ])
                
                if len(comparison_result['persistent_issues']) < summary.persistent_issues_count:
                    report.append(f"\n... and {summary.persistent_issues_count - len(comparison_result['persistent_issues'])} more persistent issues.")
    
        report.extend([
            "\n## Severity Summary",
            "| Severity | Count |",
            "|----------|-------|"
        ])
    
        # 심각도 요약 테이블
        for severity, count in sorted(summary.severity_counts.items()):
            report.append(f"| {severity} | {count} |")
    
        # 카테고리 요약 테이블    
        report.extend([
            "\n## Category Summary",
            "| Category | Count |",
            "|----------|-------|"
        ])
    
        for category, count in sorted(summary.category_counts.items()):
            report.append(f"| {category.title()} | {count} |")
    
        # 중요 이슈 섹션
        if summary.critical_issues:
            report.append("\n## Critical Issues")
            for issue in summary.critical_issues:
                report.extend([
                    f"\n### {issue['file']} (Line {issue['line_number']})",
                    f"**Issue:** {issue['description']}",
                    f"**Suggestion:** {issue['suggestion']}"
                ])
        
        if summary.major_issues:
            report.append("\n## Major Issues")
            for issue in summary.major_issues:
                report.extend([
                    f"\n### {issue['file']} (Line {issue['line_number']})",
                    f"**Issue:** {issue['description']}",
                    f"**Suggestion:** {issue['suggestion']}"
                ])
    
        # 파일별 상세 리뷰
        report.append("\n## Detailed Review by File")
        
        # 모든 이슈를 하나의 테이블로 통합
        report.extend([
            "\n| File | Line | Category | Severity | Description | Suggestion |",
            "|------|------|-----------|-----------|--------------|-------------|"
        ])
    
        # 모든 파일의 제안사항을 하나의 리스트로 통합
        all_suggestions = []
        for file_path, suggestions in summary.suggestions_by_file.items():
            for suggestion in suggestions:
                all_suggestions.append((file_path, suggestion))
    
        # 파일명과 라인 번호로 정렬
        sorted_suggestions = sorted(
            all_suggestions,
            key=lambda x: (
                x[0],  # 파일명으로 먼저 정렬
                # 'Throughout file'를 마지막으로
                x[1]['line_number'] == 'Throughout file',
                # 숫자는 숫자순으로
                int(x[1]['line_number']) if x[1]['line_number'].isdigit() else float('inf'),
                # 나머지는 문자열 순으로
                x[1]['line_number']
            )
        )
    
        # 테이블 생성
        for file_path, suggestion in sorted_suggestions:
            # 마크다운 테이블에서 파이프(|) 문자 이스케이프
            description = suggestion.get('description', 'N/A').replace('|', '\\|')
            suggestion_text = suggestion.get('suggestion', 'N/A').replace('|', '\\|')
    
            report.append(
                f"| {file_path} | {suggestion['line_number']} | "
                f"{suggestion.get('category', 'Other').title()} | "
                f"{suggestion.get('severity', 'NORMAL')} | "
                f"{description} | "
                f"{suggestion_text} |"
            )
    
        # 파일 의존성 정보를 별도 섹션으로 분리
        report.append("\n### File Dependencies")
        for file_path, ref_files in sorted(summary.reference_context.items()):
            if ref_files:  # 참조 파일이 있는 경우만 표시
                report.extend([
                    f"\n#### {file_path}",
                    "Related Files:"
                ])
                dedup_ref_files = list(set(ref_files))
                for ref_file in sorted(dedup_ref_files):
                    report.append(f"- {ref_file}")
    
        # 추가 정보 및 메타데이터
        report.extend([
            "\n## Additional Information",
            "- Review Date: " + datetime.now().strftime('%Y-%m-%d'),
            "- Base Branch: " + self.pr_details.get('base_branch', 'Unknown'),
            "- Head Branch: " + self.pr_details.get('head_branch', 'Unknown'),
            f"- Repository: {self.pr_details.get('repository', 'Unknown')}",
            f"- PR Number: {self.pr_details.get('pr_id', 'Unknown')}"
        ])
    
        # 리포트 하단에 자동 생성 표시
        report.extend([
            "\n---",
            "🤖 _This report was automatically generated by PR Review Bot & Amazon Bedrock_ 🧾"
        ])
    
        return '\n'.join(report)

    def prepare_pr_comment(self, summary: ReviewSummary) -> str:
        """PR 코멘트용 요약 생성"""
        comment = [
            "# Code Review Summary",
            f"\nReviewed {summary.total_primary_files} primary files "
            f"(with {summary.total_reference_files} reference files) "
            f"and found {summary.total_issues} issues.",
            
            "\n## Severity Breakdown",
            "| Severity | Count |",
            "|----------|-------|",
        ]
        
        for severity, count in summary.severity_counts.items():
            comment.append(f"| {severity} | {count} |")
        
        if summary.critical_issues:
            comment.append("\n### Critical Issues Found")
            for issue in summary.critical_issues:
                comment.extend([
                    f"\n- **{issue['file']}** (Line {issue['line_number']})",
                    f"  - {issue['description']}",
                    f"  - Suggestion: {issue['suggestion']}"
                ])
        
        if summary.major_issues:
            comment.append("\n### Major Issues Found")
            for issue in summary.major_issues[:5]:  # 상위 5개만 표시
                comment.extend([
                    f"\n- **{issue['file']}** (Line {issue['line_number']})",
                    f"  - {issue['description']}"
                ])
            
            if len(summary.major_issues) > 5:
                comment.append(f"\n... and {len(summary.major_issues) - 5} more major issues.")
        
        # 이전 리뷰 비교 정보 추가
        if summary.previous_reviews_count > 0:
            comment.extend([
                "\n## Review History",
                f"- Previous Reviews: {summary.previous_reviews_count}",
                f"- Resolved Issues: {summary.resolved_issues_count}",
                f"- New Issues: {summary.new_issues_count}",
                f"- Persistent Issues: {summary.persistent_issues_count}"
            ])
        
        
        return '\n'.join(comment)

    def prepare_slack_message(self, summary: ReviewSummary) -> Dict[str, Any]:
        """Slack 메시지 준비"""
        pr_title = self.pr_details.get('title', 'Unknown PR')
        pr_author = self.pr_details.get('author', 'Unknown Author')
        pr_url = self.pr_details.get('pr_url', '#')

        # PR 제목이 길 경우 축약
        MAX_TITLE_LENGTH = 100
        shortened_title = (pr_title[:MAX_TITLE_LENGTH] + '...') if len(pr_title) > MAX_TITLE_LENGTH else pr_title
        
        severity_emoji = {
            'CRITICAL': '🚨',
            'MAJOR': '⚠️',
            'MINOR': '📝',
            'NORMAL': '✅'
        }
        
        # 전체 심각도 결정
        overall_severity = 'NORMAL'
        if summary.critical_issues:
            overall_severity = 'CRITICAL'
        elif summary.major_issues:
            overall_severity = 'MAJOR'
        elif summary.severity_counts.get('MINOR', 0) > 0:
            overall_severity = 'MINOR'
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{severity_emoji[overall_severity]} Review: {shortened_title}"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Author:*\n{pr_author}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Files:*\n{summary.total_primary_files} primary + {summary.total_reference_files} reference"
                    }
                ]
            }
        ]
        
        # 심각도 요약
        severity_text = []
        for severity, count in summary.severity_counts.items():
            if count > 0:
                severity_text.append(f"{severity_emoji[severity]} {severity}: {count}")
        
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "\n".join(severity_text)
            }
        })
        
        # 중요 이슈 하이라이트
        if summary.critical_issues or summary.major_issues:
            highlight_text = ["*Critical/Major Issues:*"]
            
            for issue in (summary.critical_issues + summary.major_issues)[:3]:
                highlight_text.append(
                    f"• {issue['file']} (Line {issue['line_number']}): {issue['description'][:100]}..."
                )
            
            if len(summary.critical_issues + summary.major_issues) > 3:
                remaining = len(summary.critical_issues + summary.major_issues) - 3
                highlight_text.append(f"_...and {remaining} more critical/major issues_")
            
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "\n".join(highlight_text)
                }
            })
        
        # 파일 통계 섹션
        if summary.reference_context:
            file_stats = ["*File Dependencies:*"]
            for primary_file, ref_files in list(summary.reference_context.items())[:3]:
                file_stats.append(f"• `{primary_file}` - {len(ref_files)} related files")
            
            if len(summary.reference_context) > 3:
                remaining = len(summary.reference_context) - 3
                file_stats.append(f"_...and {remaining} more files with dependencies_")
            
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "\n".join(file_stats)
                }
            })
        
        # PR 링크 버튼
        if pr_url and pr_url != '#':
            blocks.append({
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "Review PR 👀"
                        },
                        "url": pr_url,
                        "style": "primary"
                    }
                ]
            })
        
                # 이전 리뷰와 비교 정보 추가
        if summary.previous_reviews_count > 0:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Review History:*\n✅ Resolved: {summary.resolved_issues_count} | 🆕 New: {summary.new_issues_count} | ⚠️ Persistent: {summary.persistent_issues_count}"
                }
            })
        
        
        return {
            "blocks": blocks,
            "text": f"Code Review completed for PR: {shortened_title} - Found {summary.total_issues} issues in {summary.total_primary_files} primary files"  # 폴백 텍스트
        }

    def _get_previous_reviews(self, repository: str, pr_id: str) -> List[Dict[str, Any]]:
        """동일한 PR에 대한 이전 리뷰 결과 조회"""
        try:
            response = self.results_table.query(
                IndexName='repository-pr-index',
                KeyConditionExpression=boto3.dynamodb.conditions.Key('repository').eq(repository) &
                                      boto3.dynamodb.conditions.Key('pr_id').eq(pr_id),
                ScanIndexForward=False  # 최신 항목부터 조회
            )

            # 현재 실행 ID가 아닌 이전 실행의 결과만 필터링
            previous_reviews = []
            execution_ids = set()

            for item in response.get('Items', []):
                exec_id = item.get('execution_id')
                if exec_id != self.execution_id and exec_id not in execution_ids:
                    execution_ids.add(exec_id)
                    previous_reviews.append(item)

                    # 최근 5개 실행만 가져옴
                    if len(previous_reviews) >= 5:
                        break

            print(f"Found {len(previous_reviews)} previous reviews for PR {repository}/{pr_id}")
            return previous_reviews

        except Exception as e:
            print(f"Error retrieving previous reviews: {e}")
            return []


    def _compare_with_previous_reviews(self, current_issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """현재 이슈와 이전 리뷰의 이슈를 비교"""
        if not self.previous_reviews:
            return {
                'previous_reviews_count': 0,
                'resolved_issues': [],
                'new_issues': current_issues,
                'persistent_issues': []
            }

        # 가장 최근 리뷰의 결과 가져오기
        latest_review = self.previous_reviews[0]
        previous_results = []

        # 이전 리뷰에서 모든 이슈 수집
        for item in self.results_table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key('execution_id').eq(latest_review.get('execution_id'))
        ).get('Items', []):
            if 'results' in item:
                for result in item['results']:
                    for suggestion in result.get('suggestions', []):
                        previous_results.append({
                            'file': result['file_path'],
                            'line_number': suggestion.get('line_number', 'N/A'),
                            'description': suggestion.get('description', ''),
                            'severity': suggestion.get('severity', 'NORMAL')
                        })

        # 현재 이슈와 이전 이슈 비교
        current_issue_keys = {
            f"{issue['file']}:{issue['line_number']}:{issue['description'][:50]}"
            for issue in current_issues
        }

        previous_issue_keys = {
            f"{issue['file']}:{issue['line_number']}:{issue['description'][:50]}"
            for issue in previous_results
        }

        # 해결된 이슈, 새로운 이슈, 지속적인 이슈 식별
        resolved_keys = previous_issue_keys - current_issue_keys
        new_keys = current_issue_keys - previous_issue_keys
        persistent_keys = current_issue_keys & previous_issue_keys

        # 원본 이슈 객체 찾기
        resolved_issues = [
            issue for issue in previous_results
            if f"{issue['file']}:{issue['line_number']}:{issue['description'][:50]}" in resolved_keys
        ]

        new_issues = [
            issue for issue in current_issues
            if f"{issue['file']}:{issue['line_number']}:{issue['description'][:50]}" in new_keys
        ]

        persistent_issues = [
            issue for issue in current_issues
            if f"{issue['file']}:{issue['line_number']}:{issue['description'][:50]}" in persistent_keys
        ]

        return {
            'previous_reviews_count': len(self.previous_reviews),
            'resolved_issues': resolved_issues[:10],  # 상위 10개만 표시
            'new_issues': new_issues[:10],  # 상위 10개만 표시
            'persistent_issues': persistent_issues[:10]  # 상위 10개만 표시
        }

    def store_report_in_dynamodb(self, markdown_report: str) -> bool:
        """마크다운 보고서를 별도의 DynamoDB 테이블에 저장"""
        try:
            if not self.pr_details or 'repository' not in self.pr_details or 'pr_id' not in self.pr_details:
                print("PR details missing, cannot store report")
                return False

            # 리포지토리 및 PR ID 정보 추출
            repository = self.pr_details.get('repository')
            pr_id = self.pr_details.get('pr_id')

            timestamp = datetime.now().isoformat()

            # 보고서 테이블에 저장
            self.reports_table.put_item(
                Item={
                    'repository': repository,
                    'pr_id': pr_id,
                    'execution_id': self.execution_id,
                    'timestamp': timestamp,
                    'report': markdown_report,
                    'title': self.pr_details.get('title', 'Unknown PR'),
                    'author': self.pr_details.get('author', 'Unknown Author'),
                    'base_branch': self.pr_details.get('base_branch', ''),
                    'head_branch': self.pr_details.get('head_branch', ''),
                    'pr_url': self.pr_details.get('pr_url', '')
                }
            )

            print(f"Successfully stored report for PR {repository}/{pr_id}")
            return True

        except Exception as e:
            print(f"Error storing report in DynamoDB: {e}")
            return False

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda 핸들러"""
    try:
        # 실행 ID 설정: 이전의 청크 메타데이터의 execution_id를 사용
        # 결과 집계기 초기화 - event를 직접 전달
        aggregator = ResultAggregator(event)
        print(f"Initialized ResultAggregator with execution_id: {getattr(aggregator, 'execution_id', 'NOT_SET')}")
        
        summary = aggregator.analyze_results()
        
        # DynamoDB에서 로드한 데이터로 보고서 생성
        markdown_report = aggregator.generate_markdown_report(summary)
        pr_comment = aggregator.prepare_pr_comment(summary)
        slack_message = aggregator.prepare_slack_message(summary)
        
        execution_id = getattr(aggregator, 'execution_id', f"fallback-{int(datetime.now().timestamp())}")

        report_stored = aggregator.store_report_in_dynamodb(markdown_report)
        print(f"Report stored in reports table: {report_stored}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'summary': {
                    'total_files': summary.total_files,
                    'total_primary_files': summary.total_primary_files,
                    'total_reference_files': summary.total_reference_files,
                    'total_issues': summary.total_issues,
                    'severity_counts': summary.severity_counts,
                    'category_counts': summary.category_counts,
                    'previous_reviews_count': summary.previous_reviews_count,
                    'resolved_issues_count': summary.resolved_issues_count,
                    'new_issues_count': summary.new_issues_count,
                    'persistent_issues_count': summary.persistent_issues_count
                },
                'markdown_report': markdown_report,
                'pr_comment': pr_comment,
                'slack_message': slack_message,
                'pr_details': aggregator.pr_details,
                'reference_context': summary.reference_context,
                'execution_id': execution_id  # 안전하게 가져온 실행 ID 사용
            }, ensure_ascii=False)
        }
        
    except Exception as e:
        print(f"Error aggregating results: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }