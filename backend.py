from __future__ import annotations
import os
import time
import operator
from pathlib import Path
from typing import TypedDict, List, Annotated
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


class Task(BaseModel):
    id: int
    title: str
    brief: str = Field(description="A brief description of the task")


class Plan(BaseModel):
    blog_title: str
    tasks: List[Task]


class State(TypedDict):
    topic: str
    plan: Plan
    sections: Annotated[List[str], operator.add]
    final: str


llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.7, max_retries=5)


def orchestrator(state: State) -> dict:
    messages = [
        SystemMessage(content="Create a blog plan with 3-5 sections on the following topic."),
        HumanMessage(content=f"Topic: {state['topic']}"),
    ]
    plan = None
    for attempt in range(5):
        try:
            plan = llm.with_structured_output(Plan).invoke(messages)
            break
        except Exception as e:
            err = str(e)
            if any(code in err for code in ["503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"]):
                time.sleep((attempt + 1) * 4)
            else:
                raise e
    if plan is None:
        raise RuntimeError("Failed to generate plan.")
    return {"plan": plan}


def fanout(state: State):
    return [
        Send("worker", {"task": task, "topic": state["topic"], "plan": state["plan"]})
        for task in state["plan"].tasks
    ]


def worker(payload: dict) -> dict:
    task = payload["task"]
    topic = payload["topic"]
    plan = payload["plan"]

    messages = [
        SystemMessage(content="Write one clean Markdown section."),
        HumanMessage(
            content=(
                f"Blog: {plan.blog_title}\n"
                f"Topic: {topic}\n\n"
                f"Section: {task.title}\n"
                f"Brief: {task.brief}\n\n"
                "Return only the section content in Markdown."
            )
        ),
    ]

    res = None
    for attempt in range(5):
        try:
            res = llm.invoke(messages)
            break
        except Exception as e:
            err = str(e)
            if any(code in err for code in ["503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"]):
                time.sleep((attempt + 1) * 4)
            else:
                raise e

    if res is None:
        raise RuntimeError(f"Failed to generate section '{task.title}'.")

    if hasattr(res, "text") and res.text:
        section_md = res.text.strip()
    elif isinstance(res.content, list):
        section_md = "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in res.content
        ).strip()
    else:
        section_md = str(res.content).strip()

    return {"sections": [section_md]}


def reducer(state: State) -> dict:
    title = state["plan"].blog_title
    body = "\n\n".join(state["sections"]).strip()
    final_md = f"# {title}\n\n{body}\n"

    clean_title = "".join(c if c.isalnum() or c in " _-" else "" for c in title.lower())
    filename = clean_title.strip().replace(" ", "_") + ".md"
    output_path = Path(filename)
    output_path.write_text(final_md, encoding="utf-8")

    return {"final": final_md}


graph = StateGraph(State)
graph.add_node("orchestrator", orchestrator)
graph.add_node("worker", worker)
graph.add_node("reducer", reducer)

graph.add_edge(START, "orchestrator")
graph.add_conditional_edges("orchestrator", fanout, ["worker"])
graph.add_edge("worker", "reducer")
graph.add_edge("reducer", END)

blog_app = graph.compile()
