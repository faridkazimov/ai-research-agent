# 🧠 Smart Research Assistant (LangGraph AI Agent)

![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-1.0%2B-green.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-brightgreen.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-black.svg)

This project is an autonomous AI Research Assistant built using **LangChain 1.0+**, **LangGraph**, and **Streamlit**.

Unlike a standard RAG (Retrieval-Augmented Generation) system that only answers questions based on static documents, this agent can:
1.  **Reason:** Autonomously understand complex, multi-step user queries.
2.  **Plan:** Break down the problem into a sequence of required actions.
3.  **Act:** Use external tools (like live web search) to gather dynamic, real-time data.
4.  **Synthesize:** Combine all the gathered information to provide a comprehensive, final answer.

This agent is deployed with an interactive Streamlit UI and includes a simple rate-limiting feature (4 questions per session) to manage API costs.

## 🎥 Live Demo (Example Interaction)

*(**Recommendation:** Insert a GIF of your application working here. This makes your project 10x more professional. You can easily create one using tools like [LICEcap](https://www.cockos.com/licecap/) or [GIPHY Capture](https://giphy.com/apps/giphycapture).)*

**User:** "What company has a higher market cap right now, NVIDIA or Apple? And what's the difference in US dollars?"

**Agent:** *(Thinking...)*
1.  `[Action: tavily_search(query="NVIDIA market cap")]` -> Finds $4.8T
2.  `[Action: tavily_search(query="Apple market cap")]` -> Finds $3.9T
3.  `[Synthesizing]`
4.  *(Final Answer)* "Currently, NVIDIA has a higher market cap at approximately $4.8 trillion, which is about $900 billion more than Apple's $3.9 trillion."

## 🏛️ Core Architecture (How it Works)

This project uses **LangGraph** to define the agent's logic as a "state machine" or graph. This is the modern replacement for the older `AgentExecutor` class and allows for complex, cyclical, and stateful reasoning.

The graph consists of:
1.  **AgentState:** A simple dictionary (`TypedDict`) that defines the "memory" or "state" of our graph. It primarily tracks the list of messages.
2.  **Nodes:**
    * `agent_node`: The "brain" of the operation. It calls the LLM (`gpt-4o-mini`) to decide what to do next (call a tool or generate a final answer).
    * `tool_node`: The "action". This is a prebuilt `ToolNode` that executes any tool calls requested by the `agent_node` (e.g., performs the Tavily search).
3.  **Conditional Edges:**
    * The `should_continue` function acts as the router. After the `agent_node` runs, this edge checks if the LLM requested a tool.
    * **If YES (tool call exists):** The graph routes to the `tool_node`.
    * **If NO (no tool call):** The graph routes to `END`, and the agent provides its final answer.

This cyclical flow (`agent` -> `call_tool` -> `agent` -> `END`) allows the agent to call tools multiple times, reflect on the results, and solve complex problems.

```mermaid
graph TD
    A[Start: User Input] --> B(agent_node);
    B -- Tool Call? --> C{should_continue};
    C -- Yes --> D[tool_node];
    D --> B;
    C -- No --> E[END: Final Answer];
✨ Features
Autonomous Reasoning: Built with LangGraph for a robust "Reason -> Act -> Observe" loop. The agent decides for itself when to search vs. when to answer.

Dynamic Tool Use: Integrated with TavilySearch for real-time financial data, market caps, and simple math calculations.

Interactive UI: A clean, chat-based interface built with Streamlit.

Cost Control (Rate Limiting): Includes a Streamlit session state counter to limit users to 4 questions, preventing API key abuse.

Modern Tech Stack: Built on the latest langchain 1.0+ libraries (langchain-core, langchain-openai, langgraph) and a cost-effective, powerful LLM (gpt-4o-mini).

🛠️ Tech Stack
Language: Python 3.11+

Agent Framework: LangChain 1.0+

Reasoning Engine: LangGraph

LLM (Brain): OpenAI gpt-4o-mini (recommended for cost/performance) or gpt-4o

Tools: TavilySearch (Live Web Search)

Interface (UI): Streamlit

Environment Management: python-dotenv

🚀 Setup and Installation
1. Clone the Repository:

Bash

git clone [https://github.com/](https://github.com/)[YOUR-USERNAME]/[YOUR-REPO-NAME].git
cd [YOUR-REPO-NAME]
2. Create and Activate a Virtual Environment:

Bash

python -m venv ajan-env
# On Windows
.\ajan-env\Scripts\activate
# On macOS/Linux
source ajan-env/bin/activate
3. Install Dependencies:

Bash

pip install -r requirements.txt
4. Create Your .env File: In the root of the project, create a file named .env and add your secret API keys:

Ini, TOML

OPENAI_API_KEY="sk-..."
TAVILY_API_KEY="tvly-..."
5. Run the Streamlit App:

Bash

streamlit run agent_app.py
Your browser will automatically open to the app's local URL (usually http://localhost:8501).

🗺️ Future Roadmap & Advanced Implementation
This project serves as a strong foundation. Here are the planned next steps to evolve it into a "Master-level" agent, with details on how they would be implemented.

1. Add Conversational Memory
The Challenge: The agent is currently stateless. If you ask "What is NVIDIA's market cap?" and then "What about Apple's?", it won't remember you were comparing companies.

The Solution (LangGraph): The AgentState is already built for memory! The messages: Annotated[Sequence[BaseMessage], operator.add] line ensures that messages are added to the state, not replaced. The only change needed is in the Streamlit UI code:

Store the actual BaseMessage objects (like HumanMessage, AIMessage) in st.session_state["messages"], not just dictionaries.

When calling the agent, pass the entire history: inputs = {"messages": st.session_state.messages}. This will send the full conversation context to the LLM on every turn, allowing it to remember the past.

2. Add Custom Tools (e.g., RAG Tool)
The Challenge: The agent can only search the public web. It knows nothing about my private documents.

The Solution (@tool decorator): We can create a new tool for the agent by simply decorating a Python function.

Example:

Python

from langchain_core.tools import tool

# (Assuming you have a RAG function from another project)
def my_rag_retriever(query: str) -> str:
    """Use this to find information in private company documents or PDFs."""
    # ... your RAG retrieval logic here ...
    docs = vectorstore.similarity_search(query)
    return " ".join([doc.page_content for doc in docs])

@tool
def rag_search_tool(query: str) -> str:
    """Searches private company documents for specific information."""
    return my_rag_retriever(query)

# Then, just add it to the 'tools' list in agent_app.py:
tools = [search_tool, rag_search_tool]
llm_with_tools = llm.bind_tools(tools) # Re-bind tools
tool_node = ToolNode(tools) # Update ToolNode
The agent will now autonomously choose between searching the web (TavilySearch) or your private documents (rag_search_tool) based on the user's question.

3. Add Human-in-the-Loop (Approval Step)
The Challenge: The agent acts autonomously. What if it decides to call a very expensive tool or perform a dangerous action (like deleting a file, if we gave it that tool)?

The Solution (LangGraph Edges): We can add a "pause" button to the graph.

Modify the should_continue function to return a third string, "human_approval", instead of "call_tool" if the tool is sensitive.

Add a new node (workflow.add_node("human_approval", human_approval_node)) and update the conditional edge routing: {"call_tool": "call_tool", END: END, "human_approval": "human_approval"}.

This new human_approval_node would pause the graph (using Interrupt). In Streamlit, the app would show "[Yes] / [No]" buttons. If the user clicks "Yes", the app would resume the graph execution (sending None to continue), which would then proceed to the tool_node. If "No", it could route back to the agent_node to reconsider.

4. Implement True Response Streaming (Word-by-Word)
The Challenge: The UI shows a "Thinking..." spinner and dumps the whole answer at once. This feels slow and less interactive than ChatGPT.

The Solution (app.stream() + st.write_stream()):

Instead of using final_state = app.invoke(inputs) in the Streamlit code...

We will use response_stream = app.stream(inputs, stream_mode="values").

We then need a generator function that iterates through the response_stream, finds the final agent_node output (which contains the AIMessage), and yields the content chunks (yield chunk.content).

Finally, use Streamlit's built-in st.write_stream(my_generator_function). This will render the agent's final answer word-by-word as it's being generated by the LLM.