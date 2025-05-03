import os
import streamlit as st
import openai
from docx import Document
from io import BytesIO
from functools import lru_cache

# ----------------------
# Константы и локализация
# ----------------------
LANGUAGES = {"en": "English", "ru": "Русский"}
TRANSLATIONS = {
    "title": {"en": "AI DevArchitect — Intelligent Architecture Assistant",
              "ru": "AI DevArchitect — ИИ-помощник по проектированию архитектуры"},
    "description": {"en": "First, AI suggests project ideas. Select one to get architecture and multilingual alternatives, including dependencies.",
                    "ru": "Сначала ИИ предложит идеи проектов. Затем выберите одну, чтобы получить архитектуру, зависимости и альтернативы на выбранном языке."},
    "generate_ideas": {"en": "Generate ideas", "ru": "Сгенерировать идеи"},
    "select_project": {"en": "Select project", "ru": "Выберите проект"},
    "generate_architecture": {"en": "Generate architecture", "ru": "Сгенерировать архитектуру"},
    "base_architecture": {"en": "Base Architecture & Dependencies", "ru": "Базовая архитектура и зависимости"},
    "show_alternatives": {"en": "Show alternatives", "ru": "Показать альтернативы"},
    "alternatives": {"en": "Alternative Architectures", "ru": "Альтернативные архитектуры"},
    "export_arch": {"en": "Download Architecture Report", "ru": "Скачать отчёт по архитектуре"},
    "export_full": {"en": "Download Full Report", "ru": "Скачать полный отчёт"},
    "select_model": {"en": "Select model", "ru": "Выберите модель"}
}

@lru_cache(maxsize=None)
def t(key: str) -> str:
    lang = st.session_state.get("lang", "ru")
    return TRANSLATIONS.get(key, {}).get(lang, key)

# ----------------------
# Настройка OpenAI
# ----------------------
openai.api_key = os.getenv("OPENAI_API_KEY")

# ----------------------
# Интерфейс
# ----------------------
st.set_page_config(page_title="AI DevArchitect", layout="wide")
# Сайдбар: язык и модель
st.sidebar.selectbox(
    "Language / Язык",
    options=list(LANGUAGES.keys()),
    format_func=lambda x: LANGUAGES[x],
    key="lang"
)
available_models = ["gpt-4", "gpt-3.5-turbo"]
st.sidebar.selectbox(
    t("select_model"),
    options=available_models,
    key="model"
)

st.title(t("title"))
st.write(t("description"))

# ----------------------
# OpenAI Helper
# ----------------------
@lru_cache(maxsize=None)
def call_openai_system(prompt: str, max_tokens: int = 500) -> str:
    response = openai.ChatCompletion.create(
        model=st.session_state.get("model", "gpt-4"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()

# ----------------------
# Генерация
# ----------------------
def generate_ideas() -> list[str]:
    lang = st.session_state.get("lang", "ru")
    if lang == "ru":
        prompt = (
            "Сгенерируй 5 идей небольших учебных проектов для студентов-программистов. "
            "Пример: To-Do App, Чат, Опросник. Формат:\n1. ..."
        )
    else:
        prompt = (
            "Generate 5 small educational project ideas for programming students, e.g.: To-Do App, Chat, Survey. Format:\n1. ..."
        )
    text = call_openai_system(prompt, max_tokens=300)
    return [line.split('. ', 1)[1] for line in text.splitlines() if '. ' in line]

def generate_architecture(project: str) -> str:
    lang = st.session_state.get("lang", "ru")
    if lang == "ru":
        prompt = f"""
Ты — помощник по архитектуре программных систем. Идея: \"{project}\".
Сгенерируй:
1. Сущности и их атрибуты
2. Слои приложения (Controller, Service, Repository и т.д.)
3. Основные классы и связи между ними
4. Необходимые зависимости (библиотеки, фреймворки)
5. Краткое обоснование архитектуры
"""
    else:
        prompt = f"""
You are a software architecture assistant. Idea: \"{project}\".
Generate:
1. Entities and their attributes
2. Application layers (Controller, Service, Repository, etc.)
3. Main classes and their relationships
4. Required dependencies (libraries, frameworks)
5. Brief rationale for the architecture
"""
    return call_openai_system(prompt, max_tokens=800)

def generate_alternatives(project: str, architecture: str) -> str:
    lang = st.session_state.get("lang", "ru")
    styles_ru = ["монолит", "микросервисы", "event-driven", "CQRS", "CQRS+ES", "Clean Architecture", "Hexagonal Architecture", "Onion Architecture"]
    styles_en = ["Monolith", "Microservices", "Event-driven", "CQRS", "CQRS+ES", "Clean Architecture", "Hexagonal Architecture", "Onion Architecture"]
    if lang == "ru":
        options = '\n- '.join(styles_ru)
        prompt = f"""
Проект: \"{project}\".
Архитектура и зависимости:
{architecture}

Предложи альтернативные архитектурные подходы:
- {options}
Для каждого: плюсы, минусы и случаи применения.
"""
    else:
        options = '\n- '.join(styles_en)
        prompt = f"""
Project: \"{project}\".
Architecture and dependencies:
{architecture}

Suggest alternative architectural approaches:
- {options}
For each: pros, cons, and use cases.
"""
    return call_openai_system(prompt, max_tokens=800)

# ----------------------
# UI: Взаимодействие
# ----------------------
if st.button(t("generate_ideas")):
    st.session_state.ideas = generate_ideas()

if "ideas" in st.session_state:
    project = st.selectbox(t("select_project"), st.session_state["ideas"])

    if st.button(t("generate_architecture")):
        st.session_state.arch = generate_architecture(project)
        st.session_state.project = project
        st.subheader(t("base_architecture"))
        st.write(st.session_state.arch)

        # Download architecture only
        buf = BytesIO()
        doc = Document()
        doc.add_heading(project, level=0)
        doc.add_heading(t('base_architecture'), level=1)
        doc.add_paragraph(st.session_state.arch)
        doc.save(buf)
        buf.seek(0)
        st.download_button(
            label=t("export_arch"),
            data=buf,
            file_name=f"{project}_architecture.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

if "arch" in st.session_state:
    if st.button(t("show_alternatives")):
        st.session_state.alts = generate_alternatives(st.session_state.project, st.session_state.arch)
        st.subheader(t("alternatives"))
        st.write(st.session_state.alts)

        # Download full report
        buf_full = BytesIO()
        doc_full = Document()
        doc_full.add_heading(st.session_state.project, level=0)
        doc_full.add_heading(t('base_architecture'), level=1)
        doc_full.add_paragraph(st.session_state.arch)
        doc_full.add_heading(t('alternatives'), level=1)
        doc_full.add_paragraph(st.session_state.alts)
        doc_full.save(buf_full)
        buf_full.seek(0)
        st.download_button(
            label=t("export_full"),
            data=buf_full,
            file_name=f"{st.session_state.project}_full_report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

