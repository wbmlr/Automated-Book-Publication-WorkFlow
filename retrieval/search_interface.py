from storage.chromadb_manager import client
from .rl_agent import RLSearchAgent

# Agent actions are keywords to enhance the query
agent = RLSearchAgent(actions=["summary", "characters", "style", "setting"])

def retrieve_version(collection_name: str, query: str):
    action_keyword = agent.choose_action()
    enhanced_query = f"{query} {action_keyword}".strip()
    print(f"RL Agent Used: '{action_keyword}' -> Query: '{enhanced_query}'")
    
    collection = client.get_or_create_collection(name=collection_name)
    result = collection.query(query_texts=[enhanced_query], n_results=1)
    return result, action_keyword # Return action for learning