
import streamlit as st
import re
import ast
import operator
from ddgs import DDGS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

st.set_page_config(page_title="AI Agent (Tool-Calling)", page_icon="🤖", layout="centered")

st.markdown("""
<style>
.main-title { text-align: center; font-size: 2.1rem; font-weight: 700; margin-bottom: 0; }
.subtitle { text-align: center; color: gray; margin-bottom: 25px; }
.step-box { background: #f0f2f6; padding: 10px 14px; border-radius: 8px; margin: 6px 0; font-size: 13px; font-family: monospace; }
.tool-tag { color: #ff4b4b; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🤖 AI Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Give it a task — it decides which tools to use to solve it</div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading agent model (first time takes a while)...")
def load_model():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    return tokenizer, model

tokenizer, model = load_model()


def tool_calculator(expression: str) -> str:
    """Safely evaluates a math expression — no eval() of arbitrary code."""
    ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.Mod: operator.mod,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](_eval(node.operand))
        else:
            raise ValueError("Unsupported expression")

    try:
        tree = ast.parse(expression, mode='eval')
        result = _eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Calculator error: could not evaluate '{expression}'"

def tool_web_search(query: str) -> str:
    """Searches the web using DuckDuckGo (no API key needed)."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No search results found."
        summary = "\n".join([f"- {r['title']}: {r['body'][:150]}" for r in results])
        return summary
    except Exception as e:
        return f"Web search error: {e}"

def tool_file_reader(filepath: str) -> str:
    """Reads a local text file's content."""
    try:
        filepath = filepath.strip().strip('"').strip("'")
        if not os.path.exists(filepath):
            return f"File not found: {filepath}"
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return content[:1000] + ("..." if len(content) > 1000 else "")
    except Exception as e:
        return f"File reader error: {e}"

TOOLS = {
    "calculator": tool_calculator,
    "web_search": tool_web_search,
    "file_reader": tool_file_reader,
}

TOOL_DESCRIPTIONS = """You have access to these tools:
- calculator[expression]: evaluates a math expression, e.g. calculator[15 * 23]
- web_search[query]: searches the web for current information, e.g. web_search[population of Pakistan 2026]
- file_reader[filepath]: reads a local text file, e.g. file_reader[notes.txt]"""


SYSTEM_PROMPT = f"""You are a helpful AI agent that solves tasks step by step using tools.

{TOOL_DESCRIPTIONS}

To use a tool, respond with EXACTLY this format on its own line:
Action: tool_name[input]

After you see the Observation (tool result), continue reasoning. When you have enough information to answer, respond with EXACTLY:
Final Answer: <your answer here>

Always think briefly before acting. Use at most one Action per turn. Do not make up tool results."""

MAX_STEPS = 5

def run_agent(task, log_placeholder):
    """Runs the ReAct loop and returns the final answer, logging steps live."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Task: {task}"},
    ]
    steps_log = []

    for step in range(MAX_STEPS):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs, max_new_tokens=200, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

        # Check for final answer
        final_match = re.search(r"Final Answer:\s*(.+)", response, re.DOTALL)
        if final_match:
            steps_log.append(("thought", response))
            log_placeholder.markdown("\n\n".join(
                f'<div class="step-box">{s[1]}</div>' for s in steps_log
            ), unsafe_allow_html=True)
            return final_match.group(1).strip(), steps_log

        # Check for tool action
        action_match = re.search(r"Action:\s*(\w+)\[(.*?)\]", response, re.DOTALL)
        if action_match:
            tool_name = action_match.group(1).strip()
            tool_input = action_match.group(2).strip()

            steps_log.append(("thought", response))

            if tool_name in TOOLS:
                observation = TOOLS[tool_name](tool_input)
            else:
                observation = f"Unknown tool: {tool_name}"

            steps_log.append(("obs", f'<span class="tool-tag">Observation:</span> {observation}'))

            log_placeholder.markdown("\n\n".join(
                f'<div class="step-box">{s[1]}</div>' for s in steps_log
            ), unsafe_allow_html=True)

            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Observation: {observation}"})
        else:
            # Model didn't follow format — treat as final answer attempt
            steps_log.append(("thought", response))
            log_placeholder.markdown("\n\n".join(
                f'<div class="step-box">{s[1]}</div>' for s in steps_log
            ), unsafe_allow_html=True)
            return response, steps_log

    return "Agent reached max steps without a final answer.", steps_log


task = st.text_input("Give the agent a task:", placeholder="e.g. What is 45 * 12, then search who won the last T20 World Cup?")

if st.button("Run Agent", type="primary", use_container_width=True) and task.strip():
    st.markdown("### Agent's Reasoning Steps")
    log_placeholder = st.empty()

    with st.spinner("Agent is working..."):
        final_answer, steps = run_agent(task, log_placeholder)

    st.markdown("### Final Answer")
    st.success(final_answer)

st.markdown("---")
with st.expander("ℹ️ About this agent"):
    st.markdown("""
    This agent uses the **ReAct pattern**: it alternates between reasoning and taking actions (tool calls) until it reaches a final answer.

    **Tools available:** calculator, web_search (DuckDuckGo), file_reader

    **Model:** Qwen2.5-0.5B-Instruct (local, free, no API key)

    Note: small local models follow instructions less reliably than large hosted LLMs (GPT-4, Claude) — the agent may occasionally skip tool calls or answer directly. This is expected with free/local models.
    """)
