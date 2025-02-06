import random
import numpy as np

# Step 1: Define the Recommender Environment with Multiple Genres per Movie

# Sample content (movies with multiple genres)

content = [
    {"id": 1, "name": "Movie A", "genres": ["Action", "Adventure"], "popularity": 5},
    {"id": 2, "name": "Movie B", "genres": ["Drama", "Romance"], "popularity": 4},
    {"id": 3, "name": "Movie C", "genres": ["Comedy", "Action"], "popularity": 3},
    {"id": 4, "name": "Movie D", "genres": ["Horror", "Thriller"], "popularity": 2},
    {"id": 5, "name": "Movie E", "genres": ["Action", "Sci-Fi"], "popularity": 4},
    {"id": 6, "name": "Movie F", "genres": ["Drama", "Mystery"], "popularity": 3},
    {"id": 7, "name": "Movie G", "genres": ["Action", "Comedy"], "popularity": 4},
    {"id": 8, "name": "Movie H", "genres": ["Fantasy", "Adventure"], "popularity": 3},
    {"id": 9, "name": "Movie I", "genres": ["Sci-Fi", "Action"], "popularity": 5},
    {"id": 10, "name": "Movie J", "genres": ["Romance", "Comedy"], "popularity": 3},
    {"id": 11, "name": "Movie K", "genres": ["Action", "Adventure"], "popularity": 5},
    {"id": 12, "name": "Movie L", "genres": ["Drama", "Romance"], "popularity": 4},
    {"id": 13, "name": "Movie M", "genres": ["Comedy", "Action"], "popularity": 3},
    {"id": 14, "name": "Movie N", "genres": ["Horror", "Thriller"], "popularity": 2},
    {"id": 15, "name": "Movie O", "genres": ["Action", "Sci-Fi"], "popularity": 4},
    {"id": 16, "name": "Movie P", "genres": ["Drama", "Mystery"], "popularity": 3},
    {"id": 17, "name": "Movie Q", "genres": ["Action", "Comedy"], "popularity": 4},
    {"id": 18, "name": "Movie R", "genres": ["Fantasy", "Adventure"], "popularity": 3},
    {"id": 19, "name": "Movie S", "genres": ["Sci-Fi", "Action"], "popularity": 5},
    {"id": 20, "name": "Movie T", "genres": ["Romance", "Comedy"], "popularity": 3},
]

# Sample user and their interests (user IDs, list of genres)
users = {
    1: {"interests": ["Action", "Comedy"]},
}

# Q-table structure: [user_id, content_id]
# States: User IDs
# Actions: Content IDs (movies/products)
# Q-values: Initializing all Q-values to 0
Q_table = np.zeros((len(users), len(content)))

# Step 2: Define the Q-learning agent
class RecommenderAgent:
    def __init__(self, users, content, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.users = users
        self.content = content
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def get_user_state(self, user_id):
        return user_id - 1  # Map user_id to state (index)

    def get_content_state(self, content_id):
        return content_id - 1  # Map content_id to state (index)

    def recommend(self, user_id):
        state = self.get_user_state(user_id)
        
        # Epsilon-greedy policy: Explore or exploit
        if random.uniform(0, 1) < self.epsilon:
            # Explore: Random recommendation
            action = random.choice(range(len(self.content)))
        else:
            # Exploit: Choose the content with the highest Q-value
            action = np.argmax(Q_table[state])
        
        return action

    def update_q_value(self, user_id, content_id, reward):
        state = self.get_user_state(user_id)
        action = self.get_content_state(content_id)
        
        # Q-learning formula: Q(s, a) = Q(s, a) + alpha * (reward + gamma * max_a' Q(s', a') - Q(s, a))
        max_future_q = np.max(Q_table[state])  # Best future reward
        current_q = Q_table[state][action]
        
        Q_table[state][action] = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)

# Step 3: Define the reward function
def get_reward(user_id, content_id, user_answer):
    """
    Reward is based on whether the user answers 'yes' (positive feedback)
    or 'no' (negative feedback) to the recommendation.
    """
    user_interests = users[user_id]["interests"]
    content_genres = content[content_id]["genres"]
    
    if user_answer == 'yes':
        # Positive reward for relevant recommendation (based on genre match)
        return content[content_id]["popularity"]
    else:
        # Negative reward for irrelevant recommendation
        return -content[content_id]["popularity"]

# Step 4: Interactive Function for Asking Questions
def interact_with_user(agent, user_id, num_questions=5):
    print(f"\nStarting interaction with User {user_id}...\n")
    
    for _ in range(num_questions):
        # Get a recommendation for the user
        content_id = agent.recommend(user_id)
        movie = content[content_id]
        genres = ", ".join(movie["genres"])
        
        # Ask the user if they like this movie
        user_answer = input(f"Do you like this movie? {movie['name']} (Genres: {genres}) [yes/no]: ").strip().lower()
        
        # Validate user input
        if user_answer not in ['yes', 'no']:
            print("Invalid response. Please answer 'yes' or 'no'.")
            continue
        
        # Get the reward based on the user's answer
        reward = get_reward(user_id, content_id, user_answer)
        
        # Update Q-values based on the feedback
        agent.update_q_value(user_id, content_id + 1, reward)  # content_id is 1-indexed
        
        print(f"Feedback recorded. Your response ('{user_answer}') has been used to improve recommendations.\n")
    
    print("Interaction completed.\n")

# Step 5: Recommend a Movie after Interaction
def recommend_after_interaction(agent, user_id):
    print("Recommending a movie based on your preferences...\n")
    content_id = agent.recommend(user_id)
    movie = content[content_id]
    print(f"We recommend you watch: {movie['name']} (Genres: {', '.join(movie['genres'])})\n")

# Step 6: Train the recommender system and interact with the user
def train_and_interact(agent, user_id, num_interactions=2):
    for _ in range(num_interactions):
        interact_with_user(agent, user_id)
    recommend_after_interaction(agent, user_id)

# Create and train the recommender agent
agent = RecommenderAgent(users, content)
train_and_interact(agent, user_id=1)
