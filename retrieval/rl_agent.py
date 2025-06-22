import numpy as np
import os
import random
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.exceptions import NotFittedError

class ContextualBanditAgent:
    def __init__(self, actions: list, learning_rate=0.01, epsilon=0.1):
        self.actions = actions
        self.epsilon = epsilon
        self.policy_path = 'retrieval/tfidf_bandit_policy.joblib'
        self._load_policy()

    def _initialize_models(self):
        """Initializes fresh models and fits the vectorizer."""
        print("Initializing new models and vectorizer.")
        self.models = {action: SGDRegressor(learning_rate='constant', eta0=0.01) for action in self.actions}
        self.vectorizer = TfidfVectorizer()
        # Initial vocabulary is just the actions themselves
        self.vectorizer.fit(self.actions + [q for q, a, r in self.history])

    def _get_state_vector(self, query: str):
        return self.vectorizer.transform([query])

    def choose_action(self, query: str) -> str:
        if not query or random.random() < self.epsilon:
            return random.choice(self.actions)
        
        try:
            state_vector = self._get_state_vector(query)
            q_values = {action: model.predict(state_vector)[0] for action, model in self.models.items()}
            return max(q_values, key=q_values.get)
        except (NotFittedError, ValueError):
            # If models aren't trained or there's a feature mismatch, act randomly.
            return random.choice(self.actions)

    def update(self, query: str, action: str, reward: float):
        self.history.append((query, action, reward))
        
        # Check if the query adds new vocabulary
        current_vocab = self.vectorizer.vocabulary_
        new_words = any(word not in current_vocab for word in self.vectorizer.build_tokenizer()(query.lower()))

        if new_words:
            print("New vocabulary detected. Re-training all models from history...")
            self._initialize_models() # Re-fits vectorizer on new total vocab
            # Retrain on the entire history
            for q_hist, a_hist, r_hist in self.history:
                state_vector = self._get_state_vector(q_hist)
                self.models[a_hist].partial_fit(state_vector, [r_hist])
        else:
            # No new words, just do a normal update
            state_vector = self._get_state_vector(query)
            self.models[action].partial_fit(state_vector, [reward])
            print(f"Updated TF-IDF model for action '{action}' with reward {reward}.")

    def save_policy(self):
        policy_data = {
            'models': self.models,
            'vectorizer': self.vectorizer,
            'history': self.history
        }
        joblib.dump(policy_data, self.policy_path)
        print("TF-IDF Bandit policy saved.")

    def _load_policy(self):
        if os.path.exists(self.policy_path):
            try:
                policy_data = joblib.load(self.policy_path)
                self.models = policy_data['models']
                self.vectorizer = policy_data['vectorizer']
                self.history = policy_data['history']
                print(f"Loaded saved policy with {len(self.history)} history records.")
                return
            except Exception as e:
                print(f"Policy file corrupted or invalid: {e}. Initializing new policy.")
        
        # If loading fails or file doesn't exist
        self.history = []
        self._initialize_models()