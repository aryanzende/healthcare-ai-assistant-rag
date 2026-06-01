from __future__ import annotations

import re
from langchain_core.tools import tool

DEPARTMENTS = [
    "cardiology",
    "dermatology",
    "general medicine",
    "orthopedics",
    "pediatrics",
]
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]


@tool
def check_available_slots(input_text: str):
    """Check mock appointment slots. Input format: department|day."""
    parts = [p.strip() for p in input_text.split("|", maxsplit=1)]
    department = parts[0].lower() if parts and parts[0] else "general medicine"
    day = parts[1] if len(parts) > 1 and parts[1] else "Next available day"

    if department == "orthopedic":
        department = "orthopedics"
    if department not in DEPARTMENTS:
        department = "general medicine"

    valid_days = {d.title() for d in DAYS}
    if day not in valid_days:
        day = "Next available day"

    mock_slots = {
        "cardiology": ["10:00 AM", "3:00 PM"],
        "dermatology": ["11:30 AM", "4:00 PM"],
        "general medicine": ["9:30 AM", "12:30 PM", "5:00 PM"],
        "orthopedics": ["1:00 PM", "6:00 PM"],
        "pediatrics": ["10:30 AM", "2:30 PM"],
    }
    return {
        "department": department,
        "day": day,
        "available_slots": mock_slots.get(department, ["No slots available"]),
    }


def is_appointment_question(question: str) -> bool:
    q = question.lower()
    keywords = ["appointment", "book", "slot", "schedule", "available", "doctor", "consultation"]
    return any(k in q for k in keywords)


def extract_department(question: str) -> str:
    q = question.lower()
    if "orthopedic" in q:
        return "orthopedics"
    for department in DEPARTMENTS:
        if department in q:
            return department
    return "general medicine"


def extract_day(question: str) -> str:
    q = question.lower()
    for day in DAYS:
        if re.search(rf"\b{day}\b", q):
            return day.title()
    return "Next available day"


def handle_appointment_question(question: str):
    department = extract_department(question)
    day = extract_day(question)
    result = check_available_slots.invoke(f"{department}|{day}")

    return {
        "answer": (
            f"This is demo availability. Available slots for {result['department']} on {result['day']} are: "
            f"{', '.join(result['available_slots'])}."
        ),
        "sources": [
            {
                "document": "langchain_appointment_tool",
                "chunk_index": 0,
                "chunk": "Appointment availability is generated using a LangChain tool for demo purposes.",
            }
        ],
        "confidence": "medium",
        "route": "appointment_tool",
    }
