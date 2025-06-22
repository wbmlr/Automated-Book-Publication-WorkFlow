from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from ai_services.llm_agents import get_llm_chain

# --- State Definition (Unchanged) ---
class GraphState(TypedDict):
    feedback: List[str]
    llm_provider: str
    scraped_text: str
    generated_text: str
    spun_content: str

# --- Prompt Engineering (Unchanged) ---
generator_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert editor. Your task is to synthesize and refine text. You will be given two versions: a 'Scraped Original' and a 'Current Generated Version'. You must use both texts along with the user's specific 'Feedback' to produce a single, improved version. Prioritize the user's feedback. If the 'Current Generated Version' is empty, treat this as the first generation and focus on rewriting the 'Scraped Original'."),
    ("human", """--- Scraped Original ---
{scraped_text}

--- Current Generated Version ---
{generated_text}

--- User Feedback ---
{feedback}

--- New, Improved Version Below (write only the text) ---""")
])

def create_generator_chain(llm: Runnable) -> Runnable:
    return generator_prompt | llm

# --- Graph Nodes ---
# FIXED: Converted the node to be fully asynchronous
async def generator_node(state: GraphState):
    """An ASYNCHRONOUS node that streams text token-by-token."""
    print(f"---NODE: ASYNC GENERATING (using {state['llm_provider']})---")
    llm = get_llm_chain(state['llm_provider'])
    generator_chain = create_generator_chain(llm)

    prompt_inputs = {
        "scraped_text": state['scraped_text'],
        "generated_text": state['generated_text'],
        "feedback": state.get('feedback', ["Initial spin."])[-1]
    }
    
    # Use the asynchronous streaming method: .astream()
    stream = generator_chain.astream(prompt_inputs)
    
    # Use 'async for' to iterate over the asynchronous stream
    async for chunk in stream:
        content = chunk.content if hasattr(chunk, 'content') else chunk
        yield {"spun_content": content}

# --- Graph Assembly (Unchanged) ---
workflow = StateGraph(GraphState)
workflow.add_node("generator", generator_node)
workflow.set_entry_point("generator")
workflow.add_edge("generator", END)
app = workflow.compile()