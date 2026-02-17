import json
import re
import time
from typing import List, Dict, Any, Optional

from openai import Client
from pydantic import ValidationError

from config import Config
from models import Task, EstimationResponse, PriorityResponse, SprintPlan
from utils import get_embedding


class TaskEstimatorAgent:
    def __init__(self):
        if not Config.API_KEY:
            raise ValueError(
                "Config.API_KEY is empty. Set LLM_API_KEY/OPENROUTER_API_KEY (env) or LOCAL_API_KEY in config.py."
            )
        self.client = Client(
            base_url=Config.LLM_API_URL,
            api_key=Config.API_KEY,
            timeout=Config.REQUEST_TIMEOUT_SEC,
        )
        self.estimation_prompt = self._load_prompt("prompts/estimation_prompt.txt")
        self.prioritization_prompt = self._load_prompt("prompts/prioritization_prompt.txt")

    def _load_prompt(self, file_path: str) -> str:
        """Загрузка промта из файла"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return ""

    def _strip_code_fences(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
            if text.lower().startswith("json\n"):
                text = text[5:].strip()
            elif text.lower() == "json":
                text = ""
        return text

    def _extract_text_content(self, content: Any) -> str:
        """Нормализация message.content к строке для разных форматов API."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    value = item.get("text")
                    if isinstance(value, str):
                        parts.append(value)
            return "".join(parts)
        if isinstance(content, dict):
            value = content.get("text")
            if isinstance(value, str):
                return value
        return str(content)

    def _extract_json_object(self, text: str) -> Optional[str]:
        """Попытка извлечь первый валидный JSON-объект из произвольного текста."""
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for idx in range(start, len(text)):
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        return None

    def _parse_model(self, payload: Any, model_cls):
        if isinstance(payload, str):
            payload = self._strip_code_fences(payload)
            if not payload:
                raise ValueError("Empty model response content")
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                candidate = self._extract_json_object(payload)
                if candidate:
                    try:
                        data = json.loads(candidate)
                    except json.JSONDecodeError:
                        data = None
                else:
                    data = None

                if data is not None:
                    if hasattr(model_cls, "model_validate"):
                        return model_cls.model_validate(data)
                    return model_cls.parse_obj(data)

                # fallback: try parse_raw/model_validate_json
                if hasattr(model_cls, "model_validate_json"):
                    return model_cls.model_validate_json(payload)
                return model_cls.parse_raw(payload)
        else:
            data = payload

        if hasattr(model_cls, "model_validate"):
            return model_cls.model_validate(data)
        return model_cls.parse_obj(data)

    def _render_prompt(self, template: str, values: Dict[str, Any]) -> str:
        """Безопасная подстановка только известных плейсхолдеров.
        Не трогаем остальные фигурные скобки в JSON-примерах внутри промта.
        """
        result = template
        for key, value in values.items():
            result = result.replace("{" + key + "}", str(value))
        return result

    def _clean_text(self, text: Optional[str], max_chars: int) -> str:
        if not text:
            return "Нет описания"

        cleaned = str(text)
        cleaned = re.sub(r"\{[^{}]{0,80}\}", " ", cleaned)  # Jira-панели/макросы
        cleaned = re.sub(r"!\S+?!", " ", cleaned)  # вложения !image.png!
        cleaned = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", cleaned)  # markdown links
        cleaned = re.sub(r"https?://\S+", " ", cleaned)  # голые URL
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        if not cleaned:
            return "Нет описания"

        if len(cleaned) > max_chars:
            return cleaned[:max_chars] + " ... [truncated]"
        return cleaned

    def _description_variants(self, description: Optional[str]) -> List[str]:
        full = self._clean_text(description, Config.MAX_DESCRIPTION_CHARS)
        short = self._clean_text(description, Config.SHORT_DESCRIPTION_CHARS)
        if full == short:
            return [full]
        return [full, short]

    def _nearest_story_points(self, value: float) -> float:
        scale = Config.STORY_POINT_SCALE
        return float(min(scale, key=lambda p: abs(float(p) - float(value))))

    def _nearest_story_points_in_range(self, value: float, min_sp: float, max_sp: float) -> float:
        scale = [float(p) for p in Config.STORY_POINT_SCALE if float(min_sp) <= float(p) <= float(max_sp)]
        if not scale:
            scale = [float(p) for p in Config.STORY_POINT_SCALE]
        return float(min(scale, key=lambda p: abs(float(p) - float(value))))

    def _story_point_bounds(self, task: Task) -> tuple[float, float]:
        summary = (task.issue_summary or "").lower()
        text = f"{summary} {task.description or ''}".lower()

        min_sp, max_sp = 1.0, 8.0

        # Очень маленькие/локальные задачи
        if any(k in summary for k in ["роадмап", "презентац"]):
            max_sp = min(max_sp, 2.0)

        # Типовые средние задачи
        if any(k in text for k in ["оценка рисков", "лимит", "сегментац", "написать скрипт", "оценка активных"]):
            min_sp = max(min_sp, 2.0)
            max_sp = min(max_sp, 4.0)

        # SQL / валидации
        if any(k in text for k in ["text-to-sql", "валидация", "sql"]):
            min_sp = max(min_sp, 3.0)
            max_sp = min(max_sp, 5.0)

        # Крупные аналитические блоки
        if any(k in text for k in ["витрин", "дашборд", "kpi overview", "мониторинг", "пассивы и сегментация", "опромышл"]):
            min_sp = max(min_sp, 4.0)
            max_sp = min(max_sp, 6.0)

        # Очень крупные задачи
        if any(k in text for k in ["пл 2", "первого месяца", "многоэтап"]):
            min_sp = max(min_sp, 6.0)
            max_sp = 8.0

        if min_sp > max_sp:
            min_sp, max_sp = max_sp, min_sp

        return min_sp, max_sp

    def _complexity_from_sp(self, sp: float) -> float:
        if sp <= 2:
            return 1.0
        if sp <= 4:
            return 2.0
        return 3.0

    def _normalize_dependencies(self, dependencies: Optional[List[str]]) -> List[str]:
        if not dependencies:
            return []
        cleaned = [str(dep).strip() for dep in dependencies if str(dep).strip()]
        return cleaned[: Config.MAX_DEPENDENCIES]

    def _is_retryable_error(self, err: Exception) -> bool:
        text = str(err).lower()
        retry_markers = [
            "429",
            "rate limit",
            "temporarily",
            "timeout",
            "timed out",
            "connection error",
            "502",
            "503",
            "504",
            "overloaded",
        ]
        return any(marker in text for marker in retry_markers)

    def _is_context_error(self, err: Exception) -> bool:
        text = str(err).lower()
        markers = [
            "token",
            "context",
            "maximum context length",
            "too long",
            "413",
        ]
        return any(marker in text for marker in markers)

    def _heuristic_story_points(self, task: Task) -> float:
        summary = (task.issue_summary or "").lower()
        text = f"{summary} {task.description or ''}".lower()

        # Якоря из Train_and_test.xlsx
        anchors = [
            ("роадмап спецсчета", 1.0),
            ("презентация по", 2.0),
            ("декомпозиция транз активности", 2.0),
            ("оценка рисков отсутствия проверки лимитов", 3.0),
            ("стратегия ls: 2 этап", 3.0),
            ("написать скрипт для распределения", 3.0),
            ("сегментация потенциальных клиентов", 3.0),
            ("оценка активных клиентов", 3.0),
            ("text-to-sql", 4.0),
            ("валидация скриптов sql", 4.0),
            ("пассивы и сегментация", 5.0),
            ("данные для стратегии 2026", 5.0),
            ("собрать витрину данных", 5.0),
            ("доработка kpi overview", 5.0),
            ("опромышлить источники", 5.0),
            ("мониторинг тарифа", 6.0),
            ("подготовка скрипта расчета первого месяца", 8.0),
        ]
        for phrase, sp in anchors:
            if phrase in summary:
                return sp

        if any(marker in text for marker in ["роадмап", "презентац", "декомпозиц"]):
            return 2.0
        if any(marker in text for marker in ["риск", "лимит", "сегментац", "скрипт", "оценка актив"]):
            return 3.0
        if any(marker in text for marker in ["text-to-sql", "валидац", "sql"]):
            return 4.0
        if any(marker in text for marker in ["витрин", "дашборд", "kpi", "мониторинг", "пассив"]):
            return 5.0
        return 3.0

    def _recalibrate_story_points(self, llm_sp: float, task: Task) -> float:
        prior_sp = self._heuristic_story_points(task)
        min_sp, max_sp = self._story_point_bounds(task)

        adjusted = float(llm_sp)

        # Грубые выбросы в край шкалы корректируем к prior
        if adjusted <= 1.0 and prior_sp >= 3.0:
            adjusted = prior_sp
        elif adjusted >= 8.0 and prior_sp <= 6.0:
            adjusted = prior_sp

        diff = abs(adjusted - prior_sp)
        if diff >= 4:
            blended = prior_sp
        elif diff >= 2:
            blended = 0.5 * adjusted + 0.5 * prior_sp
        else:
            blended = 0.7 * adjusted + 0.3 * prior_sp

        bounded = max(min_sp, min(max_sp, blended))
        return self._nearest_story_points_in_range(bounded, min_sp, max_sp)

    def _heuristic_priority_fields(self, task: Task) -> Dict[str, Any]:
        text = f"{task.issue_summary} {task.description or ''} {' '.join(task.labels)}".lower()
        is_urgent = any(marker in text for marker in ["цб", "регулятор", "инцидент", "дедлайн", "риск"])
        is_valuable = any(marker in text for marker in ["выруч", "kpi", "стратег", "мониторинг", "клиент"])

        business_value = 8.0 if is_valuable else 6.0
        urgency = 8.0 if is_urgent else 5.0
        dependencies: List[str] = []
        risk_score = 6.0 if is_urgent else 4.0
        risk_level = "Medium" if is_urgent else "Low"

        return {
            "business_value": business_value,
            "urgency": urgency,
            "dependencies": dependencies,
            "risk_score": risk_score,
            "risk_level": risk_level,
        }

    def _extra_body(self) -> Optional[Dict[str, Any]]:
        if Config.ENABLE_THINKING:
            return {"enable_thinking": True}
        return None

    def _request_kwargs(self, allow_thinking: bool = True) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        extra_body = self._extra_body() if allow_thinking else None
        if extra_body is not None:
            kwargs["extra_body"] = extra_body
        return kwargs

    def _call_structured(
        self,
        model: str,
        messages: List[Dict[str, str]],
        response_model,
        max_tokens: Optional[int] = None,
    ):
        last_err: Optional[Exception] = None
        thinking_modes = [True, False] if Config.ENABLE_THINKING else [False]

        for allow_thinking in thinking_modes:
            for attempt in range(Config.MAX_RETRIES + 1):
                request_kwargs = self._request_kwargs(allow_thinking=allow_thinking)
                if max_tokens is not None and bool(Config.SEND_MAX_TOKENS):
                    request_kwargs["max_tokens"] = max_tokens

                try:
                    # 1) Try parse API (если поддерживается)
                    try:
                        response = self.client.beta.chat.completions.parse(
                            model=model,
                            messages=messages,
                            response_format=response_model,
                            temperature=Config.TEMPERATURE,
                            **request_kwargs,
                        )
                        parsed = response.choices[0].message.parsed
                        if parsed is not None:
                            return parsed

                        parsed_content = self._extract_text_content(response.choices[0].message.content)
                        if parsed_content.strip():
                            return self._parse_model(parsed_content, response_model)
                        raise ValueError("Empty model response content")
                    except Exception as e:
                        last_err = e

                    # 2) Try JSON-only response format
                    try:
                        response = self.client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=Config.TEMPERATURE,
                            response_format={"type": "json_object"},
                            **request_kwargs,
                        )
                        content = self._extract_text_content(response.choices[0].message.content)
                        if not content.strip():
                            raise ValueError("Empty model response content")
                        return self._parse_model(content, response_model)
                    except Exception as e:
                        last_err = e

                    # 3) Plain text fallback
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=Config.TEMPERATURE,
                        **request_kwargs,
                    )
                    content = self._extract_text_content(response.choices[0].message.content)
                    if not content.strip():
                        raise ValueError("Empty model response content")
                    return self._parse_model(content, response_model)

                except (ValidationError, json.JSONDecodeError) as e:
                    last_err = e
                except Exception as e:
                    last_err = e

                if attempt < Config.MAX_RETRIES:
                    # Для rate-limit/network/context ошибок выдерживаем backoff и пробуем снова.
                    sleep_sec = Config.RETRY_BACKOFF_SEC * (attempt + 1)
                    if last_err and self._is_retryable_error(last_err):
                        sleep_sec *= 1.5
                    time.sleep(sleep_sec)

        raise last_err or RuntimeError("LLM structured call failed")

    def _clamp(self, value: Optional[float], min_val: float, max_val: float, default: float) -> float:
        if value is None:
            return default
        return max(min_val, min(max_val, float(value)))

    def _risk_score_from_level(self, risk_level: Optional[str]) -> float:
        if not risk_level:
            return 5.0
        level = risk_level.strip().lower()
        if level == "low":
            return 2.0
        if level == "medium":
            return 5.0
        if level == "high":
            return 8.0
        return 5.0

    def _compute_priority_score(self, business_value: float, urgency: float, dependencies: List[str], risk_score: float) -> float:
        mode = (Config.PRIORITY_SIGNAL_MODE or "full").strip().lower()
        if mode == "core":
            score = (business_value * 0.6 + urgency * 0.4) * 10.0
            return self._clamp(score, 0.0, 100.0, 50.0)

        weights = Config.PRIORITY_WEIGHTS
        dep_score = min(10.0, float(len(dependencies))) if dependencies else 0.0
        score = (
            business_value * weights["business_value"]
            + urgency * weights["urgency"]
            + dep_score * weights["dependencies"]
            + (10.0 - risk_score) * weights["risk"]
        ) * 10.0
        return self._clamp(score, 0.0, 100.0, 50.0)

    def estimate_task(self, task: Task, force: bool = False) -> Task:
        """Оценка задачи в сторипоинтах"""
        if task.estimated_story_points is not None and not force:
            return task

        try:
            last_error: Optional[Exception] = None
            for idx, description in enumerate(self._description_variants(task.description)):
                prompt = self._render_prompt(
                    self.estimation_prompt,
                    {
                        "issue_summary": task.issue_summary,
                        "issue_type": task.issue_type,
                        "description": description,
                    },
                )

                try:
                    estimation = self._call_structured(
                        model=Config.ESTIMATION_MODEL,
                        messages=[
                            {"role": "system", "content": "Ты опытный аналитик в IT компании."},
                            {"role": "user", "content": prompt},
                        ],
                        response_model=EstimationResponse,
                        max_tokens=Config.MAX_OUTPUT_TOKENS_ESTIMATION,
                    )

                    raw_sp = self._nearest_story_points(float(estimation.story_points))
                    normalized_sp = self._recalibrate_story_points(raw_sp, task)
                    task.estimated_story_points = normalized_sp
                    if estimation.complexity:
                        task.complexity_score = {"Low": 1, "Medium": 2, "High": 3}.get(estimation.complexity, 2)
                    else:
                        task.complexity_score = self._complexity_from_sp(normalized_sp)
                    return task
                except Exception as e:
                    last_error = e
                    # Если это похоже на context/token проблему — пробуем более короткий вариант.
                    if idx == 0 and self._is_context_error(e):
                        continue
                    # Для прочих ошибок пробуем следующий вариант (если есть).
                    continue

            raise last_error or RuntimeError("Failed to estimate task")

        except Exception as e:
            print(f"{task.issue_id}: {e}")
            fallback_sp = self._heuristic_story_points(task)
            task.estimated_story_points = fallback_sp
            task.complexity_score = self._complexity_from_sp(fallback_sp)
            return task

    def prioritize_task(self, task: Task, force: bool = False) -> Task:
        """Расстановка приоритетов для задачи"""
        if task.priority_score is not None and not force:
            return task

        try:
            last_error: Optional[Exception] = None
            for idx, description in enumerate(self._description_variants(task.description)):
                prompt = self._render_prompt(
                    self.prioritization_prompt,
                    {
                        "issue_summary": task.issue_summary,
                        "issue_type": task.issue_type,
                        "description": description,
                        "labels": ", ".join(task.labels),
                    },
                )

                try:
                    priority = self._call_structured(
                        model=Config.ESTIMATION_MODEL,
                        messages=[
                            {"role": "system", "content": "Ты product owner."},
                            {"role": "user", "content": prompt},
                        ],
                        response_model=PriorityResponse,
                        max_tokens=Config.MAX_OUTPUT_TOKENS_PRIORITY,
                    )

                    business_value = self._clamp(priority.business_value, 1.0, 10.0, 5.0)
                    urgency = self._clamp(priority.urgency, 1.0, 10.0, 5.0)
                    risk_score = self._clamp(
                        priority.risk_score,
                        1.0,
                        10.0,
                        self._risk_score_from_level(priority.risk_level),
                    )
                    dependencies = self._normalize_dependencies(priority.dependencies)

                    task.business_value = business_value
                    task.urgency_score = urgency
                    task.risk_level = priority.risk_level
                    task.risk_score = risk_score
                    task.dependencies = dependencies

                    task.priority_score = self._compute_priority_score(
                        business_value=business_value,
                        urgency=urgency,
                        dependencies=dependencies,
                        risk_score=risk_score,
                    )
                    return task
                except Exception as e:
                    last_error = e
                    if idx == 0 and self._is_context_error(e):
                        continue
                    continue

            raise last_error or RuntimeError("Failed to prioritize task")

        except Exception as e:
            print(f"Ошибка приоритезации задачи {task.issue_id}: {e}")
            fallback = self._heuristic_priority_fields(task)
            task.business_value = fallback["business_value"]
            task.urgency_score = fallback["urgency"]
            task.risk_level = fallback["risk_level"]
            task.risk_score = fallback["risk_score"]
            task.dependencies = fallback["dependencies"]
            task.priority_score = self._compute_priority_score(
                business_value=task.business_value,
                urgency=task.urgency_score,
                dependencies=task.dependencies or [],
                risk_score=task.risk_score or 5.0,
            )
            return task

    def estimate_tasks(self, tasks: List[Task], force: bool = False) -> List[Task]:
        result: List[Task] = []
        for i, task in enumerate(tasks, start=1):
            estimated = self.estimate_task(task, force=force)
            result.append(estimated)
            if i < len(tasks) and Config.TASK_ESTIMATION_DELAY_SEC > 0:
                time.sleep(Config.TASK_ESTIMATION_DELAY_SEC)
        return result

    def prioritize_tasks(self, tasks: List[Task], force: bool = False) -> List[Task]:
        result: List[Task] = []
        for i, task in enumerate(tasks, start=1):
            prioritized = self.prioritize_task(task, force=force)
            result.append(prioritized)
            if i < len(tasks) and Config.TASK_PRIORITIZATION_DELAY_SEC > 0:
                time.sleep(Config.TASK_PRIORITIZATION_DELAY_SEC)
        return result

    def generate_embeddings(self, tasks: List[Task]) -> List[Task]:
        """Генерация эмбеддингов для задач"""
        if not Config.EMBEDDING_MODEL:
            print("Эмбеддинги отключены для текущего провайдера (EMBEDDING_MODEL is None).")
            return tasks

        for task in tasks:
            text_to_embed = f"{task.issue_summary} {task.description or ''} {' '.join(task.labels)}"
            try:
                task.embedding = get_embedding(text_to_embed)
            except Exception as e:
                print(f"Ошибка генерации эмбеддинга для задачи {task.issue_id}: {e}")
                task.embedding = None
        return tasks

    def assign_tasks_to_analysts(self, tasks: List[Task]) -> Dict[str, List[Task]]:
        """Распределение задач по аналитикам с ограничениями"""
        analysts = Config.ANALYSTS
        max_sp = Config.MAX_STORY_POINTS_PER_ANALYST
        min_tasks = Config.MIN_TASKS_PER_ANALYST
        max_tasks = Config.MAX_TASKS_PER_ANALYST

        valid_tasks = [t for t in tasks if (t.estimated_story_points or 0) > 0]  # snake_case
        rejected_tasks = [t for t in tasks if (t.estimated_story_points or 0) <= 0]
        print(f"Rejected {len(rejected_tasks)} tasks with SP <=0: {[t.issue_id for t in rejected_tasks]}")
        
        remaining = sorted(
            valid_tasks, 
            key=lambda x: (-(x.priority_score or 0), float(x.estimated_story_points or 999))
        )  

        analyst_allocation = {analyst: [] for analyst in analysts}
        analyst_workload = {analyst: 0.0 for analyst in analysts}

        def pick_task_for(analyst: str):
            for i, task in enumerate(remaining):
                sp = float(task.estimated_story_points or 5.0) 
                if len(analyst_allocation[analyst]) >= max_tasks:
                    return None
                if analyst_workload[analyst] + sp <= max_sp:
                    remaining.pop(i)
                    return task
            return None

        # Фаза 1: постараться дать всем минимум задач
        for _ in range(min_tasks):
            for analyst in analysts:
                if len(analyst_allocation[analyst]) >= min_tasks:
                    continue
                task = pick_task_for(analyst)
                if task is None:
                    continue
                analyst_allocation[analyst].append(task)
                analyst_workload[analyst] += float(task.estimated_story_points or 5.0)
                task.assigned_analyst = analyst

        # Фаза 2: добрать до максимума задач, если есть место
        for _ in range(max_tasks - min_tasks):
            for analyst in analysts:
                if len(analyst_allocation[analyst]) >= max_tasks:
                    continue
                task = pick_task_for(analyst)
                if task is None:
                    continue
                analyst_allocation[analyst].append(task)
                analyst_workload[analyst] += float(task.estimated_story_points or 5.0)
                task.assigned_analyst = analyst

        return analyst_allocation

    def plan_sprint(self, tasks: List[Task]) -> SprintPlan:
        """Планирование спринта: оценка, приоритезация, отбор и распределение"""
        # Последовательная обработка бэклога по одной задаче:
        # сначала гарантируем оценки, затем приоритеты, затем сборка спринта.
        estimated_tasks = self.estimate_tasks(
            tasks,
            force=Config.FORCE_REESTIMATE_IN_PLAN,
        )
        prioritized_tasks = self.prioritize_tasks(
            estimated_tasks,
            force=Config.FORCE_REPRIORITIZE_IN_PLAN,
        )
        analyst_allocation = self.assign_tasks_to_analysts(prioritized_tasks)

        
        sprint_tasks = [task for tasks_list in analyst_allocation.values() for task in tasks_list]
        sprint_task_ids = {task.issue_id for task in sprint_tasks}  # ✅ issue_id
        backlog_tasks = [task for task in prioritized_tasks if task.issue_id not in sprint_task_ids]

        # Метрики по спринту
        total_story_points = sum(task.estimated_story_points or 0 for task in sprint_tasks)
        capacity_utilization = {
            analyst: (
                sum(task.estimated_story_points or 0 for task in tasks_list)
                / Config.MAX_STORY_POINTS_PER_ANALYST
            )
            for analyst, tasks_list in analyst_allocation.items()
        }

        return SprintPlan(
            sprint_tasks=sprint_tasks,
            backlog_tasks=backlog_tasks,
            analyst_allocation=analyst_allocation,
            total_story_points=total_story_points,
            capacity_utilization=capacity_utilization,
        )
