import json
import os
import re
import boto3
from typing import Dict, List, Any, Optional, Set, Tuple
import requests
from dataclasses import dataclass, field
from collections import defaultdict
import uuid
import networkx as nx

@dataclass
class FileChange:
    path: str
    content: str
    weight: float = 0.0
    related_files: List[str] = field(default_factory=list)
    review_status: str = 'pending'  # pending, in_review, reviewed
    primary_chunk_id: Optional[str] = None
    reference_chunks: Set[str] = field(default_factory=set)

@dataclass
class ChunkMetadata:
    chunk_id: str
    total_weight: float = 0.0
    primary_files: Set[str] = field(default_factory=set)
    reference_files: Set[str] = field(default_factory=set)

class FileCache:
    def __init__(self):
        self.reviewed_files: Dict[str, Dict[str, Any]] = {}
        self.file_to_chunk_map: Dict[str, str] = {}
        
    def add_reviewed_file(self, file_path: str, chunk_id: str, review_data: Dict[str, Any]):
        self.reviewed_files[file_path] = review_data
        self.file_to_chunk_map[file_path] = chunk_id
        
    def get_review_data(self, file_path: str) -> Optional[Dict[str, Any]]:
        return self.reviewed_files.get(file_path)
        
    def is_file_reviewed(self, file_path: str) -> bool:
        return file_path in self.reviewed_files
        
    def get_chunk_id(self, file_path: str) -> Optional[str]:
        return self.file_to_chunk_map.get(file_path)

class PRChunkSplitter:
    def __init__(self, event_data: Dict[str, Any]):
        self.event_data = event_data
        self.secrets = boto3.client('secretsmanager')
        self.file_cache = FileCache()
        self.chunk_metadata: Dict[str, ChunkMetadata] = {}
        self.dependency_graph = nx.DiGraph()
        self._load_credentials()
        
    def _load_credentials(self):
        """저장소 인증 정보 로드"""
        try:
            secret = self.secrets.get_secret_value(
                SecretId=f'/pr-reviewer/tokens/{self.event_data["repository_type"]}'
            )
            self.credentials = json.loads(secret['SecretString'])
        except Exception as e:
            print(f"Error loading credentials: {e}")
            raise

    def get_pr_diff(self) -> Optional[str]:
        """저장소 타입별 PR diff 가져오기"""
        repo_type = self.event_data["repository_type"]
        repository = self.event_data["repository"]
        pr_id = self.event_data["pr_id"]

        try:
            if repo_type == "github":
                return self._get_github_diff(repository, pr_id)
            elif repo_type == "gitlab":
                return self._get_gitlab_diff(repository, pr_id)
            elif repo_type == "bitbucket":
                return self._get_bitbucket_diff(repository, pr_id)
            else:
                raise ValueError(f"Unsupported repository type: {repo_type}")
        except Exception as e:
            print(f"Error fetching PR diff: {e}")
            raise

    def _get_github_diff(self, repository: str, pr_id: str) -> str:
        """GitHub PR diff 가져오기"""
        headers = {
            "Authorization": f"Bearer {self.credentials['access_token']}",
            "Accept": "application/vnd.github.v3.diff"
        }
        url = f"https://api.github.com/repos/{repository}/pulls/{pr_id}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text

    def _get_gitlab_diff(self, repository: str, pr_id: str) -> str:
        """GitLab PR diff 가져오기"""
        headers = {"PRIVATE-TOKEN": self.credentials['access_token']}
        url = f"https://gitlab.com/api/v4/projects/{requests.utils.quote(repository, safe='')}/merge_requests/{pr_id}/changes"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return self._format_gitlab_diff(response.json()['changes'])

    def _get_bitbucket_diff(self, repository: str, pr_id: str) -> str:
        """Bitbucket PR diff 가져오기"""
        headers = {
            "Authorization": f"Bearer {self.credentials['access_token']}",
            "Accept": "application/vnd.github.v3.diff"
        }
        workspace, repo_slug = repository.split('/')
        url = f"https://api.bitbucket.org/2.0/repositories/{workspace}/{repo_slug}/pullrequests/{pr_id}/diff"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text

    def _format_gitlab_diff(self, changes: List[Dict]) -> str:
        """GitLab 변경사항을 diff 형식으로 변환"""
        diff_lines = []
        for change in changes:
            diff_lines.append(f"diff --git a/{change['old_path']} b/{change['new_path']}")
            diff_lines.append(f"--- a/{change['old_path']}")
            diff_lines.append(f"+++ b/{change['new_path']}")
            diff_lines.append(change['diff'])
        return '\n'.join(diff_lines)

    def parse_diff(self, diff: str) -> List[FileChange]:
        """diff를 파일 단위로 파싱"""
        files = []
        current_file = None
        current_content = []

        for line in diff.splitlines():
            if line.startswith('diff --git'):
                if current_file:
                    files.append(FileChange(
                        path=current_file,
                        content='\n'.join(current_content)
                    ))
                file_match = re.search(r'diff --git a/(.*?) b/', line)
                if file_match:
                    current_file = file_match.group(1)
                    current_content = []
            elif current_file:
                current_content.append(line)

        if current_file:
            files.append(FileChange(
                path=current_file,
                content='\n'.join(current_content)
            ))

        return files

    def calculate_file_weights(self, files: List[FileChange]) -> List[FileChange]:
        """파일별 가중치 계산"""
        for file in files:
            # 메서드 변경 수 계산
            method_changes = len(re.findall(r'^\s*[+-]\s*def\s+\w+\s*\(', file.content, re.MULTILINE))
            
            # 클래스 변경 수 계산
            class_changes = len(re.findall(r'^\s*[+-]\s*class\s+\w+\s*[:\(]', file.content, re.MULTILINE))
            
            # import 변경 수 계산
            import_changes = len(re.findall(r'^\s*[+-]\s*(import|from)\s+\w+', file.content, re.MULTILINE))
            
            # 변경된 라인 수 계산
            changed_lines = len([l for l in file.content.splitlines() if l.startswith('+') or l.startswith('-')])
            
            # 가중치 계산
            file.weight = (
                method_changes * 3 +  # 메서드 변경은 높은 가중치
                class_changes * 5 +   # 클래스 변경은 더 높은 가중치
                import_changes * 1 +  # import 변경은 낮은 가중치
                changed_lines * 0.1   # 일반 라인 변경은 기본 가중치
            )

        return files

    def analyze_file_relationships(self, files: List[FileChange]) -> None:
        """파일 간의 관계를 분석하여 의존성 그래프 구축"""
        # 그래프 초기화
        self.dependency_graph.clear()
        
        # 모든 파일을 노드로 추가
        for file in files:
            self.dependency_graph.add_node(file.path, file_data=file)
        
        # 파일 간의 관계 분석 및 엣지 추가
        for i, file1 in enumerate(files):
            for j, file2 in enumerate(files):
                if i != j:
                    relationship_strength = self._calculate_relationship_strength(file1, file2)
                    if relationship_strength > 0:
                        self.dependency_graph.add_edge(
                            file1.path, 
                            file2.path, 
                            weight=relationship_strength
                        )

    def _calculate_relationship_strength(self, file1: FileChange, file2: FileChange) -> float:
        """두 파일 간의 관계 강도 계산"""
        strength = 0.0
        
        # 같은 디렉토리에 있는 파일
        if os.path.dirname(file1.path) == os.path.dirname(file2.path):
            strength += 1.0
        
        # 비슷한 이름을 가진 파일
        file1_name = os.path.splitext(os.path.basename(file1.path))[0]
        file2_name = os.path.splitext(os.path.basename(file2.path))[0]
        if file1_name in file2_name or file2_name in file1_name:
            strength += 2.0
        
        # import 관계
        if self._has_import_relationship(file1, file2):
            strength += 3.0
        
        return strength

    def _has_import_relationship(self, file1: FileChange, file2: FileChange) -> bool:
        """두 파일 간의 import 관계 확인"""
        file1_module = os.path.splitext(file1.path)[0].replace('/', '.')
        file2_module = os.path.splitext(file2.path)[0].replace('/', '.')
        
        # file1이 file2를 import하는지 확인
        if re.search(f'^\s*[+-]\s*(from|import)\s+{file2_module}', file1.content, re.MULTILINE):
            return True
            
        # file2가 file1을 import하는지 확인
        if re.search(f'^\s*[+-]\s*(from|import)\s+{file1_module}', file2.content, re.MULTILINE):
            return True
            
        return False

    def create_initial_chunks(self, files: List[FileChange], max_weight_per_chunk: float = 15.0) -> List[Dict]:
        """Phase 1: 초기 청크 생성"""
        components = list(nx.connected_components(self.dependency_graph.to_undirected()))
        chunks = []
        
        for component in components:
            component_files = [file for file in files if file.path in component]
            subgraph = self.dependency_graph.subgraph(component)
            
            # 강한 연결 요소 찾기
            strong_components = list(nx.strongly_connected_components(subgraph))
            
            for strong_component in strong_components:
                chunk_files = []
                current_weight = 0.0
                
                # 강한 연결 요소 내의 파일들을 가중치 기준으로 정렬
                sorted_files = sorted(
                    [f for f in component_files if f.path in strong_component],
                    key=lambda x: x.weight,
                    reverse=True
                )
                
                # 청크에 파일 추가
                for file in sorted_files:
                    if current_weight + file.weight <= max_weight_per_chunk:
                        chunk_files.append(file)
                        current_weight += file.weight
                    else:
                        # 새로운 청크 시작
                        if chunk_files:
                            chunk_id = str(uuid.uuid4())
                            chunks.append(self._create_chunk(chunk_files, chunk_id))
                            chunk_files = [file]
                            current_weight = file.weight
                
                # 남은 파일들로 청크 생성
                if chunk_files:
                    chunk_id = str(uuid.uuid4())
                    chunks.append(self._create_chunk(chunk_files, chunk_id))
        
        return chunks

    def _create_chunk(self, files: List[FileChange], chunk_id: str) -> Dict:
        """청크 생성 및 메타데이터 업데이트"""
        # 청크 메타데이터 생성
        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            total_weight=sum(f.weight for f in files),
            primary_files={f.path for f in files}
        )
        
        # 관련 파일 찾기 및 참조 파일 추가
        reference_files = set()
        for file in files:
            neighbors = set(self.dependency_graph.neighbors(file.path))
            reference_files.update(neighbors - metadata.primary_files)
        
        metadata.reference_files = reference_files
        self.chunk_metadata[chunk_id] = metadata
        
        # 파일 상태 업데이트
        for file in files:
            file.primary_chunk_id = chunk_id
            file.review_status = 'in_review'
        
        return {
            'chunk_id': chunk_id,
            'files': [{
                'path': f.path,
                'content': f.content,
                'is_primary': True
            } for f in files] + [{
                'path': path,
                'content': self.dependency_graph.nodes[path]['file_data'].content,
                'is_primary': False
            } for path in reference_files],
            'total_weight': metadata.total_weight
        }

    def optimize_chunks(self, initial_chunks: List[Dict]) -> List[Dict]:
        """Phase 2: 청크 최적화"""
        # 중복 제거 및 참조 정보 최적화
        optimized_chunks = []
        processed_files = set()
        
        for chunk in initial_chunks:
            chunk_id = chunk['chunk_id']
            metadata = self.chunk_metadata[chunk_id]
            
            # 이미 처리된 파일 필터링
            new_primary_files = []
            new_reference_files = []
            
            # 주 파일 처리
            for file_data in chunk['files']:
                if file_data['is_primary']:
                    if file_data['path'] not in processed_files:
                        new_primary_files.append(file_data)
                        processed_files.add(file_data['path'])
                else:
                    # 참조 파일이 다른 청크의 주 파일이 아닌 경우에만 포함
                    if not any(
                        file_data['path'] in self.chunk_metadata[c['chunk_id']].primary_files
                        for c in optimized_chunks
                    ):
                        new_reference_files.append(file_data)
            
            # 최적화된 청크에 충분한 파일이 있는 경우에만 추가
            if new_primary_files:
                optimized_chunk = {
                    'chunk_id': chunk_id,
                    'files': new_primary_files + new_reference_files,
                    'total_weight': sum(
                        self.dependency_graph.nodes[f['path']]['file_data'].weight
                        for f in new_primary_files
                    )
                }
                optimized_chunks.append(optimized_chunk)
        
        return optimized_chunks

    def create_chunks(self, files: List[FileChange], max_weight_per_chunk: float = 15.0) -> List[Dict]:
        """Two-Phase 청크 생성 프로세스"""
        # 파일이 1개인 경우 바로 단일 청크 생성
        if len(files) == 1:
            single_file = files[0]
            chunk = {
                'chunk_id': str(uuid.uuid4()),
                'files': [{
                    'path': single_file.path,
                    'content': single_file.content,
                    'is_primary': True
                }],
                'total_weight': single_file.weight,
                'pr_details': self.event_data
            }
            return [chunk]

        # Phase 1: 파일 관계 분석 및 초기 청크 생성
        self.analyze_file_relationships(files)
        initial_chunks = self.create_initial_chunks(files, max_weight_per_chunk)
        
        # Phase 2: 청크 최적화
        optimized_chunks = self.optimize_chunks(initial_chunks)
        
        # PR 상세 정보 포함
        enriched_chunks = []
        for chunk in optimized_chunks:
            enriched_chunk = {
                **chunk,  # 기존 chunk 정보
                'pr_details': self.event_data  # PR 상세 정보 추가
            }
            enriched_chunks.append(enriched_chunk)
        
        return enriched_chunks

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda 핸들러"""
    try:
        # 이전 단계에서 전달받은 데이터로 초기화
        splitter = PRChunkSplitter(event)
        
        # PR diff 가져오기
        diff = splitter.get_pr_diff()
        if not diff:
            raise ValueError("Failed to get PR diff")
            
        # 파일별로 분리
        files = splitter.parse_diff(diff)
        
        # 가중치 계산
        files = splitter.calculate_file_weights(files)
        
        # Two-Phase 청크 생성
        chunks = splitter.create_chunks(files)
        
        return {
            'statusCode': 200,
            'body': {
                'chunks': chunks,
                'total_files': len(files),
                'total_chunks': len(chunks)
            }
        }
        
    except Exception as e:
        print(f"Error splitting PR into chunks: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }