from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class Task(BaseModel):
    issue_id: str
    issue_num: str
    project_key: str
    project_name: str
    issue_type: str
    issue_summary: str
    creation_date: str
    current_assignee: Optional[str]
    epic_issue_id: Optional[str]
    labels: List[str]
    description: Optional[str]
    estimated_story_points: Optional[float] = None
    complexity_score: Optional[float] = None
    business_value: Optional[float] = None
    urgency_score: Optional[float] = None
    risk_level: Optional[str] = None
    risk_score: Optional[float] = None
    dependencies: Optional[List[str]] = None
    priority_score: Optional[float] = None
    assigned_analyst: Optional[str] = None
    embedding: Optional[List[float]] = None

class SprintPlan(BaseModel):
    sprint_tasks: List[Task]
    backlog_tasks: List[Task]
    analyst_allocation: Dict[str, List[Task]]
    total_story_points: float
    capacity_utilization: Dict[str, float]

class EstimationResponse(BaseModel):
    story_points: float = Field(description="Оценка в сторипоинтах (1, 2, 3, 4, 5, 6, 8)")
    complexity: Optional[str] = Field(default=None, description="Сложность задачи: Low, Medium, High")

class PriorityResponse(BaseModel):
    priority_score: float = Field(description="Приоритет от 0 до 100")
    business_value: float = Field(description="Бизнес-ценность от 1 до 10")
    urgency: float = Field(description="Срочность от 1 до 10")
    dependencies: List[str] = Field(description="Зависимости от других задач")
    risk_level: str = Field(description="Уровень риска: Low, Medium, High")
    risk_score: Optional[float] = Field(default=None, description="Риск от 1 до 10")
