from __future__ import annotations
import os
import time
import operator
from pathlib import Path
from typing import TypedDict, List, Annotated
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


# 1. Schema Definitions
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


# 2. LLM Initialization (Gemini 3.5 Flash)
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.7, max_retries=5)


# 3. Graph Nodes with Exponential Backoff Retries
def orchestrator(state: State) -> dict:
    print(f"[*] Orchestrator: Generating blog plan for topic: '{state['topic']}'...")
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
            err_msg = str(e)
            if any(code in err_msg for code in ["503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"]):
                wait_sec = (attempt + 1) * 4
                print(f"[*] Orchestrator: Server busy/rate-limited, retrying in {wait_sec}s (attempt {attempt+1}/5)...")
                time.sleep(wait_sec)
            else:
                raise e

    if plan is None:
        raise RuntimeError("Failed to generate plan after retries.")

    print(f"[+] Plan created: '{plan.blog_title}' with {len(plan.tasks)} sections.")
    return {"plan": plan}


def fanout(state: State):
    print("[*] Fanout: Dispatching workers for each section...")
    return [
        Send("worker", {"task": task, "topic": state["topic"], "plan": state["plan"]})
        for task in state["plan"].tasks
    ]


def worker(payload: dict) -> dict:
    task = payload["task"]
    topic = payload["topic"]
    plan = payload["plan"]

    print(f"[*] Worker: Writing section '{task.title}'...")
    messages = [
        SystemMessage(content="Write one clean, detailed Markdown section."),
        HumanMessage(
            content=(
                f"Blog Title: {plan.blog_title}\n"
                f"Topic: {topic}\n\n"
                f"Section: {task.title}\n"
                f"Brief: {task.brief}\n\n"
                "Return only the section content in clean Markdown format."
            )
        ),
    ]

    res = None
    for attempt in range(5):
        try:
            res = llm.invoke(messages)
            break
        except Exception as e:
            err_msg = str(e)
            if any(code in err_msg for code in ["503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"]):
                wait_sec = (attempt + 1) * 4
                print(f"[*] Worker '{task.title}': API busy (503/429), retrying in {wait_sec}s (attempt {attempt+1}/5)...")
                time.sleep(wait_sec)
            else:
                raise e

    if res is None:
        raise RuntimeError(f"Failed to generate section '{task.title}' after retries.")

    if hasattr(res, "text") and res.text:
        section_md = res.text.strip()
    elif isinstance(res.content, list):
        section_md = "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in res.content
        ).strip()
    else:
        section_md = str(res.content).strip()

    print(f"[+] Worker: Completed section '{task.title}'")
    return {"sections": [section_md]}


def reducer(state: State) -> dict:
    print("[*] Reducer: Combining all sections into final blog...")
    title = state["plan"].blog_title
    body = "\n\n".join(state["sections"]).strip()
    final_md = f"# {title}\n\n{body}\n"

    # Save to Markdown file
    clean_title = "".join(c if c.isalnum() or c in " _-" else "" for c in title.lower())
    filename = clean_title.strip().replace(" ", "_") + ".md"
    output_path = Path(filename)
    output_path.write_text(final_md, encoding="utf-8")

    print(f"[✓] Successfully saved full blog to: {output_path.resolve()}")
    return {"final": final_md}


# 4. Build and Compile Graph
graph = StateGraph(State)
graph.add_node("orchestrator", orchestrator)
graph.add_node("worker", worker)
graph.add_node("reducer", reducer)

graph.add_edge(START, "orchestrator")
graph.add_conditional_edges("orchestrator", fanout, ["worker"])
graph.add_edge("worker", "reducer")
graph.add_edge("reducer", END)

app = graph.compile()


# 5. Main Execution Entrypoint
if __name__ == "__main__":
    topic = "Write a blog on Self Attention in Transformers"
    print(f"\n=== Starting Multi-Agent Blog Generator ===")
    out = app.invoke({"topic": topic, "sections": []})
    print("\n=== Blog Generated Successfully! ===")
    print(f"Blog Title: {out['plan'].blog_title}")
    print(f"Sections Count: {len(out['sections'])}")
    print(f"Total Markdown Characters: {len(out['final'])}")
    print("\nPreview:\n" + out["final"][:400] + "\n\n... (full content saved to file) ...\n")
