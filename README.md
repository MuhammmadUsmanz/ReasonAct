# 🧠 ReasonAct — A ReAct-based Multi-Tool Reasoning Agent

An autonomous AI agent that solves multi-step tasks by reasoning about what it needs, then choosing the right tool — calculator, web search, or file reader — to get there. Built on the **ReAct (Reasoning + Acting)** pattern, using a fully local, free LLM (no API key required).

## What Problem Does This Solve?

A normal chatbot can only answer from what it already knows. **ReasonAct** is built for tasks that need more than one kind of help in a single request — for example:

> *"Calculate 25% of last year's revenue, then look up this year's inflation rate to compare"*

A plain LLM can't do the second half of that (it doesn't know current data). ReasonAct recognizes it needs a **tool**, calls it, reads the result, and continues reasoning — chaining as many tool calls as the task needs.

### Where this pattern is used in practice
- Research assistants that need to combine calculation with live information
- Customer support bots that need to look something up *and* compute a value
- Personal productivity tools ("read this file and summarize the numbers in it")
- The same underlying pattern used by AutoGPT, LangChain Agents, and most modern "agentic AI" products — implemented here from scratch for full transparency into the reasoning loop

## Features

- 🧠 **ReAct reasoning loop** — alternates between reasoning and acting, up to 5 steps per task
- 🧮 **Calculator tool** — safely evaluates math expressions using Python's AST (no unsafe `eval`)
- 🌐 **Web search tool** — live web search via DuckDuckGo, no API key needed
- 📄 **File reader tool** — reads and summarizes local text files
- 💬 **Streamlit UI** — shows the agent's reasoning steps live as it works, not just the final answer
- 🔒 **100% free** — runs entirely on a local LLM (Qwen2.5-0.5B-Instruct), no OpenAI/Anthropic key required

## Tech Stack

| Component | Technology |
|---|---|
| Agent reasoning | ReAct pattern (custom implementation) |
| LLM | Qwen2.5-0.5B-Instruct (local, via Hugging Face Transformers) |
| Web search | DuckDuckGo Search (`ddgs`) |
| UI | Streamlit |
| Math evaluation | Python `ast` module (safe expression parsing, not `eval`) |
