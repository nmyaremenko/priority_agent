import os

class Config:
    # API настройки
    # VPN_API_KEY = "<YOUR_VPN_API_KEY>"
    # OPENROUTER_API_KEY = "<OPENROUTER_API_KEY>"
    OPENROUTER_API_KEY = ""
    LOCAL_API_KEY = ""
    API_KEY = (
        os.getenv("LLM_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or OPENROUTER_API_KEY
        or LOCAL_API_KEY
    ).strip()

    # Автоматический выбор провайдера по формату ключа:
    # - sk-or-* или пустой ключ -> OpenRouter (режим по умолчанию для локального теста)
    # - иначе                  -> корпоративный VPN API
    USE_OPENROUTER = API_KEY.startswith("sk-or-") or not API_KEY
    LLM_API_URL = (
        "https://openrouter.ai/api/v1"
        if USE_OPENROUTER
        else "https://llm-api.cibaa.raiffeisen.ru"
    )

    # Модели
    ESTIMATION_MODEL = (
        "deepseek/deepseek-r1-0528:free"
        if USE_OPENROUTER
        else "qwen3-next-80b-a3b-thinking"
    )
    EMBEDDING_MODEL = None if USE_OPENROUTER else "qwen3-embedding-06b"
    
    # Поведение запросов
    TEMPERATURE = 0.2
    ENABLE_THINKING = not USE_OPENROUTER
    MAX_RETRIES = 3
    RETRY_BACKOFF_SEC = 1.0
    REQUEST_TIMEOUT_SEC = 90
    MAX_OUTPUT_TOKENS_ESTIMATION = 220
    MAX_OUTPUT_TOKENS_PRIORITY = 260
    SEND_MAX_TOKENS = USE_OPENROUTER
    MAX_DESCRIPTION_CHARS = 3200
    SHORT_DESCRIPTION_CHARS = 1200
    TASK_ESTIMATION_DELAY_SEC = 0.2
    TASK_PRIORITIZATION_DELAY_SEC = 0.2
    STORY_POINT_SCALE = [1, 2, 3, 4, 5, 6, 8]
    MAX_DEPENDENCIES = 5
    FORCE_REESTIMATE_IN_PLAN = True
    FORCE_REPRIORITIZE_IN_PLAN = True

    # Параметры планирования спринта
    ANALYSTS = ["analyst_1", "analyst_2", "analyst_3", "analyst_4"]
    SPRINT_DURATION_DAYS = 14
    MAX_STORY_POINTS_PER_ANALYST = 10
    MIN_TASKS_PER_ANALYST = 3
    MAX_TASKS_PER_ANALYST = 6

    # Веса для приоритезации (сумма должна быть 1.0)
    PRIORITY_WEIGHTS = {
        "business_value": 0.4,
        "urgency": 0.3,
        "dependencies": 0.2,
        "risk": 0.1
    }

    # Режим подсчета приоритета:
    # - "full": 4 сигнала (business_value, urgency, dependencies, risk)
    # - "core": только 2 сигнала (business_value, urgency)
    PRIORITY_SIGNAL_MODE = "full"
