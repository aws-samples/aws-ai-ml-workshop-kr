import decimal
import json
import os
import boto3
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import botocore
import re
from datetime import datetime

# DynamoDB를 위한 JSON 직렬화 헬퍼 함수
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return float(obj) if obj % 1 else int(obj)
        return super(DecimalEncoder, self).default(obj)

def convert_floats_to_decimals(obj: Any) -> Any:
    """float 값을 DynamoDB에 저장 가능한 Decimal로 변환"""
    if isinstance(obj, float):
        return decimal.Decimal(str(obj))
    elif isinstance(obj, dict):
        return {k: convert_floats_to_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats_to_decimals(item) for item in obj]
    return obj

@dataclass
class ReviewResult:
    file_path: str
    language: str
    summary: Dict[str, List[str]]   # 변경사항 서머리
    severity: str
    suggestions: List[Dict[str, Any]]
    is_primary: bool = True
    referenced_by: List[str] = None

    def __post_init__(self):
        if self.referenced_by is None:
            self.referenced_by = []
        
        # 각 suggestion에 file_path 추가
        for suggestion in self.suggestions:
            suggestion['file'] = self.file_path

class ChunkProcessor:
    def __init__(self, event_data: Dict[str, Any]):
        print(event_data)
        # chunk와 config 데이터 추출
        self.chunk_data = event_data
        self.config = self.chunk_data.get('pr_details', {}).get('config', {})
        self.bedrock = boto3.client('bedrock-runtime', region_name=self.config.get('aws_region'))
        self.dynamodb = boto3.resource('dynamodb')
        self.results_table = self.dynamodb.Table('PRReviewerResults')

    def _detect_language(self, file_path: str) -> str:
        """파일 확장자 기반 프로그래밍 언어 감지"""
        ext_to_lang = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.java': 'Java',
            '.go': 'Go',
            '.cpp': 'C++',
            '.hpp': 'C++',
            '.c': 'C',
            '.h': 'C',
            '.cs': 'C#',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.rs': 'Rust',
            '.sql': 'SQL',
            '.sh': 'Shell',
            '.yaml': 'YAML',
            '.yml': 'YAML',
            '.json': 'JSON',
            '.xml': 'XML',
            '.md': 'Markdown',
            '.css': 'CSS',
            '.scss': 'SCSS',
            '.html': 'HTML'
        }
        ext = os.path.splitext(file_path)[1].lower()
        return ext_to_lang.get(ext, 'Unknown')

    def _extract_code_patterns(self, content: str) -> Dict[str, List[str]]:
        """코드에서 중요 패턴 추출"""
        patterns = {
            'security_risks': [
                r'eval\s*\(',
                r'exec\s*\(',
                r'subprocess\.',
                r'os\.system',
                r'password\s*=',
                r'api_key\s*=',
                r'token\s*=',
                r'\.exec\(',
                r'sqlite\s*\.',
            ],
            'performance_issues': [
                r'while\s*True',
                r'\.sleep\(',
                r'\.all\(',
                r'\.filter\(',
                r'\.order_by\(',
            ],
            'error_prone': [
                r'except\s*:',
                r'catch\s*\(\s*\)',
                r'null',
                r'undefined',
                r'TODO',
                r'FIXME',
            ]
        }
        
        findings = {category: [] for category in patterns}
        lines = content.splitlines()
        
        for i, line in enumerate(lines, 1):
            for category, pattern_list in patterns.items():
                for pattern in pattern_list:
                    if re.search(pattern, line):
                        findings[category].append({
                            'line_number': i,
                            'line_content': line.strip(),
                            'pattern': pattern
                        })
        
        return findings

    def _prepare_review_prompt(self, file_path: str, content: str, language: str, patterns: Dict[str, List[str]], is_primary: bool, related_files: List[str] = None) -> str:
        """리뷰를 위한 프롬프트 준비"""
        prompt = f"""당신은 전문 코드 리뷰어 입니다. Pull Request에서 변경된 코드를 리뷰해주세요.

File Status: {'Primary File' if is_primary else 'Reference File'}
File Path: {file_path}

Code Changes:
```{language}
{content}
```

Detected Patterns:
"""
        for category, findings in patterns.items():
            if findings:
                prompt += f"\n{category.replace('_', ' ').title()}:\n"
                for finding in findings:
                    prompt += f"- Line {finding['line_number']}: {finding['line_content']}\n"

        if related_files:
            prompt += "\nRelated Files:\n"
            for related_file in related_files:
                prompt += f"- {related_file}\n"

        prompt += """
해당 파일의 주요 변경사항을 다음 카테고리별로 분석하고, JSON 형식으로 응답해주세요.

응답 형식:
{
    "summary": {
        "functional_changes": [
            "새로운 기능 A 추가",
            "기존 기능 B 수정"
        ],
        "architectural_changes": [
            "디자인 패턴 X 도입",
            "모듈 구조 Y 변경"
        ],
        "technical_improvements": [
            "성능 최적화 적용",
            "코드 품질 개선"
        ]
    },
    "severity": "CRITICAL/MAJOR/MINOR/NORMAL",
    "review_points": [
        {
            "category": "security/performance/style/logic",
            "severity": "CRITICAL",
            "line_number": "42",
            "description": "SQL 인젝션 취약점 발견",
            "suggestion": "파라미터화된 쿼리 사용 권장"
        }
    ]
}

모든 설명은 한글로 작성하되, 파일명, 메서드명, 클래스명은 원본 그대로 사용해주세요.
변경사항은 실제 코드에서 발견된 내용만 포함해주세요.
"""
        return prompt

    def _analyze_with_bedrock(self, prompt: str) -> Dict[str, Any]:
        """Bedrock을 사용한 코드 분석"""
        try:
            print(f"Sending request to Bedrock with prompt: {prompt}")

            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.config.get('max_tokens', 4096),
                "temperature": self.config.get('temperature', 0.7),
                "top_p": self.config.get('top_p', 0.9),
                "system": "당신은 senior code reviewer입니다. 피드백을 구체적이고 실행 가능한 내용으로 작성하는 데 집중하세요. 요청된 JSON 형식으로 응답을 반환하세요.",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })

            try:
                response = self.bedrock.invoke_model(
                    modelId=self.config.get('model', 'apac.anthropic.claude-3-5-sonnet-20241022-v2:0'),
                    contentType='application/json',
                    accept='application/json',
                    body=body.encode()
                )
            except self.bedrock.exceptions.ThrottlingException as e:
                print(f"Bedrock throttling error: {e}")
                raise Exception("RATE_LIMIT_ERROR: Bedrock API rate limit exceeded") from e
            except botocore.exceptions.ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'ThrottlingException':
                    raise Exception("RATE_LIMIT_ERROR: Bedrock API rate limit exceeded") from e
                raise

            response_body = json.loads(response['body'].read())
            print(f"Bedrock response: {json.dumps(response_body, indent=2)}")

            if isinstance(response_body.get('content'), list):
                content = response_body['content'][0].get('text', '')
            else:
                content = response_body.get('content', '{}')

            review_json = json.loads(content)
            return review_json

        except json.JSONDecodeError as e:
            print(f"Error parsing Bedrock response as JSON: {e}")
            raise Exception("VALIDATION_ERROR: Failed to parse Bedrock response") from e
        except Exception as e:
            print(f"Error analyzing with Bedrock: {e}")
            raise

    def _determine_severity(self, review_points: List[Dict[str, Any]]) -> str:
        """리뷰 포인트를 기반으로 전체 심각도 결정"""
        severity_levels = {
            'CRITICAL': 4,
            'MAJOR': 3,
            'MINOR': 2,
            'NORMAL': 1
        }
        
        max_severity = 'NORMAL'
        max_severity_value = 1
        
        for point in review_points:
            severity = point.get('severity', 'NORMAL')
            severity_value = severity_levels.get(severity, 1)
            if severity_value > max_severity_value:
                max_severity_value = severity_value
                max_severity = severity
        
        return max_severity

    def _get_related_files(self, file_data: Dict[str, Any]) -> List[str]:
        """같은 청크 내의 관련 파일 목록 가져오기"""
        current_file_path = file_data['path']
        return [
            f['path'] for f in self.chunk_data.get('files', [])
            if f['path'] != current_file_path
        ]

    def process_file(self, file_data: Dict[str, Any]) -> ReviewResult:
        """개별 파일 처리"""
        file_path = file_data['path']
        content = file_data['content']
        is_primary = file_data.get('is_primary', True)
        
        language = self._detect_language(file_path)
        patterns = self._extract_code_patterns(content)
        related_files = self._get_related_files(file_data)
        
        prompt = self._prepare_review_prompt(
            file_path, content, language, patterns, 
            is_primary, related_files
        )
        
        review_data = self._analyze_with_bedrock(prompt)
        
        return ReviewResult(
            file_path=file_path,
            language=language,
            summary=review_data.get('summary', {
                'functional_changes': [],
                'architectural_changes': [],
                'technical_improvements': []
            }),
            severity=review_data.get('severity', 'NORMAL'),
            suggestions=review_data.get('review_points', []),
            is_primary=is_primary,
            referenced_by=related_files
        )
    
    def store_results_in_dynamodb(self, execution_id: str, chunk_id: str, chunk_results: List[Dict[str, Any]], severity: str, pr_details: Dict[str, Any]) -> bool:
        """DynamoDB에 분석 결과 저장"""
        try:
            # TTL 값 설정 (1일 후 만료)
            ttl = int((datetime.now().timestamp() + (1 * 24 * 60 * 60)))

            # PR 정보 추출
            repository = pr_details.get('repository', 'unknown')
            pr_id = pr_details.get('pr_id', 'unknown')

            # float 값을 DynamoDB에 저장 가능한 Decimal로 변환
            processed_chunk_results = convert_floats_to_decimals(chunk_results)
            processed_pr_details = convert_floats_to_decimals(pr_details)

            # DynamoDB에 결과 저장
            self.results_table.put_item(
                Item={
                    'execution_id': execution_id,
                    'chunk_id': chunk_id,
                    'repository': repository,
                    'pr_id': pr_id,
                    'review_time': datetime.now().isoformat(),
                    'results': processed_chunk_results,
                    'severity': severity,
                    'pr_details': processed_pr_details,
                    'timestamp': datetime.now().isoformat(),
                    'ttl': ttl
                }
            )
            return True
        except Exception as e:
            print(f"Error storing results in DynamoDB: {e}")
            return False

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda 핸들러"""
    try:
        processor = ChunkProcessor(event)
        chunk_results = []
        
        # PR 상세 정보 추출
        pr_details = processor.chunk_data.get('pr_details', {})
        chunk_id = processor.chunk_data.get('chunk_id')
        
        # 실행 ID 추출 - Step Functions에서 사용하는 실행 ID
        execution_id = event.get('execution_id', f"manual-{int(datetime.now().timestamp())}")
        
        # 청크 내의 각 파일 처리
        for file_data in processor.chunk_data.get('files', []):
            try:
                result = processor.process_file(file_data)
                chunk_results.append({
                    'file_path': result.file_path,
                    'language': result.language,
                    'summary': result.summary,
                    'severity': result.severity,
                    'suggestions': result.suggestions,
                    'is_primary': result.is_primary,
                    'referenced_by': result.referenced_by
                })
            except Exception as e:
                print(f"Error processing file {file_data['path']}: {e}")
                chunk_results.append({
                    'file_path': file_data['path'],
                    'language': 'Unknown',
                    'summary': {
                        'functional_changes': [],
                        'architectural_changes': [],
                        'technical_improvements': []
                    },
                    'severity': 'NORMAL',
                    'suggestions': [],
                    'is_primary': file_data.get('is_primary', True),
                    'referenced_by': []
                })

        # 전체 청크의 심각도 결정 (primary 파일만 고려)
        primary_suggestions = [
            s for r in chunk_results 
            if r['is_primary']
            for s in r.get('suggestions', [])
        ]
        chunk_severity = processor._determine_severity(primary_suggestions)
        
        # DynamoDB에 결과 저장
        storage_success = processor.store_results_in_dynamodb(
            execution_id, 
            chunk_id, 
            chunk_results, 
            chunk_severity, 
            pr_details
        )

        # Step Functions에 최소한의 정보만 반환
        return {
            'statusCode': 200,
            'body': json.dumps({
                'chunk_id': chunk_id,
                'execution_id': execution_id,
                'chunk_severity': chunk_severity,
                'files_count': len(chunk_results),
                'primary_files_count': sum(1 for r in chunk_results if r['is_primary']),
                'reference_files_count': sum(1 for r in chunk_results if not r['is_primary']),
                'stored_in_dynamodb': storage_success,
                'repository': pr_details.get('repository', 'unknown'),  # PR 식별 정보 추가
                'pr_id': pr_details.get('pr_id', 'unknown')            # PR 식별 정보 추가
            }, ensure_ascii=False)
        }

    except Exception as e:
        print(f"Error processing chunk: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'chunk_id': event.get('chunk_data', {}).get('chunk_id', None)
            })
        }