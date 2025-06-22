# Automated Book Publication Workflow

This system fetches web content, refines it with AI and human oversight, and versions it for intelligent retrieval.

## Key Features

- **Scrape & Store:** Fetches content and screenshots via Playwright, caching in PostgreSQL.
- **AI Generation:** Uses a LangGraph workflow with LLMs (e.g., Gemini) for text generation.
- **Human-in-the-Loop:** UI allows for iterative feedback and approval of AI-generated text.
- **Vector Storage:** Stores approved content in ChromaDB for semantic search.
- **RL-Powered Search:** A Contextual Bandit agent learns from user ratings to enhance search queries, improving retrieval accuracy over time.

## How to Run

1.  **Backend Setup:**
    ```bash
    # Install Python dependencies
    pip install -r requirements.txt
    # Add API keys to a .env file
    touch .env 
    ```

2.  **Frontend Setup:**
    ```bash
    cd hitl-interface
    npm install
    cd ..
    ```

3.  **Launch Services:**
    -   Start Backend: `uvicorn api:api --reload`
    -   Start Frontend: `cd hitl-interface && npm start`

Access the application at `http://localhost:3000`.

