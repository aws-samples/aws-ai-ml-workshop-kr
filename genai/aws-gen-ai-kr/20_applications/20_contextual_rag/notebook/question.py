import json
import logging
from pathlib import Path
from typing import List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Question:
    id: str
    question: str
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Question':
        """Create Question instance from dictionary"""
        return cls(
            id=data['id'],
            question=data['question']
        )

class QuestionLoader:
    def __init__(self, file_path: str = 'data/questions.jsonl'):
        self.file_path = Path(file_path)
        
    def load_questions(self) -> List[Question]:
        """Load questions from JSONL file"""
        if not self._validate_file():
            raise FileNotFoundError(f"Questions file not found: {self.file_path}")
            
        questions = []
        try:
            with self.file_path.open('r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        question = self._parse_line(line, line_num)
                        if question:
                            questions.append(question)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing line {line_num}: {e}")
                        continue
                    
            logger.info(f"Successfully loaded {len(questions)} questions from {self.file_path}")
            return questions
            
        except Exception as e:
            logger.error(f"Error loading questions: {e}")
            raise
            
    def _validate_file(self) -> bool:
        """Validate if file exists and is not empty"""
        return self.file_path.exists() and self.file_path.stat().st_size > 0
        
    def _parse_line(self, line: str, line_num: int) -> Question:
        """Parse single line from JSONL file"""
        try:
            data = json.loads(line.strip())
            self._validate_question_data(data)
            return Question.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Error in line {line_num}: {e}")
            return None
            
    @staticmethod
    def _validate_question_data(data: dict) -> None:
        """Validate question data structure"""
        required_fields = ['id', 'question']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}") 