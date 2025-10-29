import os
from dotenv import load_dotenv
import streamlit as st
# --- 1. Import Necessary Modules ---

# Standard modules
from langchain_openai import ChatOpenAI 
from langchain_tavily import TavilySearch 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_core.messages import BaseMessage, HumanMessage

# Modules required for LangGraph
import operator
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode # A prebuilt node for running tools

# --- 2. Load API Keys ---
load_dotenv()

# --- 3. Define the Agent "Brain" (LLM) and Tools ---

# Define the LLM (Chat model)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define the tool (Tavily)
search_tool = TavilySearch(max_results=1, include_answer=True)
# The description (in Turkish) tells the agent when to use the tool
search_tool.description = (
   "It's used to find information about current events, stock prices, market caps, and to perform simple mathematical calculations (addition, subtraction, multiplication, and division)."
)
tools = [search_tool]

# In Langchain 1.x, we inform the LLM it can use these tools using the "bind" method.
llm_with_tools = llm.bind_tools(tools)

# --- 4. Define the Agent State ---
# LangGraph operates on a 'state' dictionary.
# This 'state' carries information between the 'nodes' in the graph.
# Our state will only hold a list of 'messages'.

class AgentState(TypedDict):
    # The 'messages' key is a list of BaseMessage objects.
    # 'operator.add' ensures new messages are appended to the list (not overwritten).
    messages: Annotated[Sequence[BaseMessage], operator.add]

# --- 5. Define the Graph Nodes ---
# The graph will consist of two nodes:
# 1. 'agent' node: Runs the LLM, decides which tool to use.
# 2. 'tool_node' node: Runs the chosen tool.

# Node 1: 'agent' (the decision-maker)
def agent_node(state):
    print("--- NODE: AGENT (Decision Making) ---")
    # Gets the current messages from the 'state' and sends them to the LLM
    response = llm_with_tools.invoke(state['messages'])
    # Returns the LLM's response (tool call request or final answer)
    # to be added to the 'messages' list.
    return {"messages": [response]}

# Node 2: 'tool_node' (the action-taker)
# 'ToolNode' is a prebuilt node provided by LangGraph.
# When we give it our 'tools' list, it automatically runs
# incoming tool call requests and returns the result.
tool_node = ToolNode(tools)

# --- 6. Define the Graph Edges (Flow Direction) ---
# This is a function that decides what happens after the 'agent' node.
def should_continue(state):
    print("--- EDGE: Decision Point ---")
    last_message = state['messages'][-1]
    
    # If the last message has NO 'tool_call' request,
    # it means the LLM has given the final answer. End the flow (END).
    if not last_message.tool_calls:
        print("-> Ending Flow (END)")
        return END
    
    # If there is a 'tool_call', go to the 'call_tool' direction.
    else:
        print("-> Tool Call Required (call_tool)")
        return "call_tool"

# --- 7. Create and Compile the Graph ---
print("--- Creating Graph... ---")

# StateGraph initializes a graph using our 'AgentState' definition.
workflow = StateGraph(AgentState)

# Add the nodes to the graph
workflow.add_node("agent", agent_node) # A node named "agent" that runs the 'agent_node' function
workflow.add_node("call_tool", tool_node) # A node named "call_tool" that runs the 'tool_node'

# Define the graph's starting point ('entry point')
workflow.set_entry_point("agent")

# Add the conditional edge:
# After the 'agent' node, run the 'should_continue' function.
# If 'should_continue' -> returns "call_tool": Go to the 'call_tool' node.
# If 'should_continue' -> returns END: Finish the graph.
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "call_tool": "call_tool",
        END: END
    }
)

# After the 'call_tool' node runs, unconditionally go back to the 'agent' node.
workflow.add_edge("call_tool", "agent")

# Compile the graph into a runnable application named 'app'.
# This is our new 'agent_executor'.
app = workflow.compile()

# --- 9. Create the Streamlit Interface (with Question Limit) ---

st.title("🧠 Smart Research Assistant")

# --- Define Constants ---
MAX_QUESTIONS = 4 # Maximum number of questions a user can ask

# --- Initialize Session State ---
# Initialize both chat history and the question counter
if "messages" not in st.session_state:
    st.session_state.messages = []
if "question_count" not in st.session_state:
    st.session_state.question_count = 0 # Start the counter at 0

# --- Display Past Messages ---
# Print all previous messages to the screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Check and Display Question Limit ---
questions_left = MAX_QUESTIONS - st.session_state.question_count
is_disabled = questions_left <= 0

# An info box showing the user how many questions are left
if not is_disabled:
    st.info(f"You have {questions_left} questions remaining in this session.")

# --- Define the Chat Input Box ---
# The 'disabled' parameter locks the box when the limit is reached
prompt = st.chat_input(
    "Ask a question...", 
    disabled=is_disabled
)

# --- Process New Prompt if Submitted ---
if prompt:
    # Increment the counter as soon as a question is asked
    st.session_state.question_count += 1
    
    # Add user's message to state and display on screen
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run the agent
    with st.chat_message("assistant"):
        # The format LangGraph expects
        inputs = {"messages": [HumanMessage(content=prompt)]} 
        
        with st.spinner("Thinking... Researching..."):
            # Run the agent and get the final answer
            final_state = app.invoke(inputs)
            response_content = final_state['messages'][-1].content
        
        st.markdown(response_content)
    
    # Add the agent's response to the state as well
    st.session_state.messages.append({"role": "assistant", "content": response_content})
    
    # If this was the last question (counter reached MAX_QUESTIONS),
    # 'rerun' to instantly update the page and lock the input box.
    if st.session_state.question_count >= MAX_QUESTIONS:
        st.rerun()

# --- Show Warning When Limit is Reached ---
if is_disabled:
    st.warning("You have reached your question limit. Thank you!")
   