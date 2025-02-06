# Recommender System with Q-Learning - Report

This report details the implementation of a movie recommender system using Q-learning. The system recommends movies to users based on their interests and feedback.  Movies can have multiple genres.

## 1. Problem Definition

The goal is to build a recommender system that suggests movies to users based on their preferences.  The system learns these preferences through interaction with the user, receiving positive or negative feedback on its recommendations.  The challenge lies in handling movies with multiple genres and effectively learning user preferences from limited interactions.

## 2. Solution Approach

Q-learning, a reinforcement learning algorithm, is used to train the recommender system. The key components are:

### 2.1. Content Representation

Movies are represented as dictionaries with `id`, `name`, `genres` (a list of genres), and `popularity`.  A sample `content` list is provided, containing movie data.

### 2.2. User Representation

Users are represented as dictionaries with `id` and `interests` (a list of preferred genres). A sample `users` dictionary is provided.

### 2.3. Q-Table

The Q-table is a NumPy array that stores the Q-values for each user-movie pair.  `Q_table[user_id - 1][movie_id - 1]` represents the Q-value for recommending a specific movie to a specific user.  The Q-values are initialized to 0.  Note that the user and movie IDs are 1-indexed in the original data but are 0-indexed in the Q-table.

### 2.4. Recommender Agent

The `RecommenderAgent` class implements the Q-learning logic:

*   **`__init__`**: Initializes the agent with users, content, learning rate (`alpha`), discount factor (`gamma`), and exploration rate (`epsilon`).
*   **`get_user_state`**: Maps user IDs to state indices (0-indexed).
*   **`get_content_state`**: Maps content IDs to state indices (0-indexed).
*   **`recommend`**: Implements an epsilon-greedy policy.  With probability `epsilon`, it recommends a random movie (exploration). Otherwise, it recommends the movie with the highest Q-value for the user (exploitation).
*   **`update_q_value`**: Updates the Q-value based on the user's feedback (reward).  It uses the Q-learning update rule:  `Q(s, a) = Q(s, a) + alpha * (reward + gamma * max_a' Q(s', a') - Q(s, a))`.

### 2.5. Reward Function

The `get_reward` function calculates the reward based on the user's feedback. A 'yes' answer gives a positive reward equal to the movie's popularity. A 'no' answer gives a negative reward equal to the movie's popularity. This encourages the agent to recommend popular movies that match user interests.

### 2.6. User Interaction

The `interact_with_user` function handles the interaction with the user:

1.  The agent recommends a movie.
2.  The user provides feedback ('yes' or 'no').
3.  The reward is calculated.
4.  The Q-value is updated.

### 2.7. Recommendation after Interaction

The `recommend_after_interaction` function recommends a movie based on the learned Q-values after the interaction phase.

### 2.8. Training and Interaction

The `train_and_interact` function orchestrates the training (interaction) and final recommendation.  It calls `interact_with_user` multiple times to allow the agent to learn from user feedback and then calls `recommend_after_interaction` to make a final recommendation.

## 3. Results

The code initializes the agent, trains it by interacting with the user, and then provides a final movie recommendation.  The user interacts with the system by providing 'yes' or 'no' feedback to the recommended movies.  The Q-table is updated after each interaction, allowing the agent to learn user preferences. The final recommendation is based on the learned Q-values.  Due to the random exploration and limited interactions, the recommendations might not be perfect but will improve as the agent interacts more with the user.

## 4. Discussion

This implementation demonstrates a basic Q-learning approach to building a movie recommender system.  It effectively handles movies with multiple genres and learns user preferences through interaction.

Further improvements could include:

*   **More sophisticated reward functions:**  Consider factors like genre overlap between user interests and movie genres, recency of recommendations, and diversity of recommendations.
*   **Exploration-exploitation strategies:**  Implement more advanced strategies for balancing exploration and exploitation.
*   **Handling cold-start problem:**  Develop methods to recommend movies to new users with no interaction history.
*   **Larger dataset and more users:**  Test the system with a larger dataset and more users to evaluate its scalability and performance.
*   **Parameter tuning:** Experiment with different values for `alpha`, `gamma`, and `epsilon` to optimize the learning process.
*   **User profiles:** Store more detailed user profiles, including demographics, ratings, and other preferences.

This report provides a comprehensive overview of the implemented Q-learning based movie recommender system.  The code provides a working foundation that can be extended and improved upon.