# Chapter 18: Reinforcement Learning

This notebook serves as a summary and submission for **Chapter 18** of the book *"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow."* It reproduces the code from the chapter and provides theoretical explanations for each concept.

---

## 📚 Chapter Summary: Key Concepts

This chapter introduces **Reinforcement Learning (RL)**, a distinct branch of machine learning where an **agent** learns to make optimal decisions by interacting with an **environment**.

The key concepts covered in this notebook include:

* **Core Concepts:** The fundamental RL loop is introduced: an agent takes **actions** in an environment, receives **observations** and **rewards**, and learns a **policy** (strategy) to maximize its cumulative reward. This involves the **credit assignment problem** and the use of a **discount factor (γ)**.
* **Policy Gradients (PG):** A technique where the policy is a neural network that outputs action probabilities. The model is trained by running episodes and increasing the probability of actions that led to good "returns" (future discounted rewards).
* **Markov Decision Processes (MDPs):** The formal framework for RL, which leads to algorithms like **Value Iteration** and **Q-Value Iteration** for finding the optimal policy when the environment's dynamics are known.
* **Q-Learning:** A **Temporal Difference (TD) learning** algorithm for cases where the environment's dynamics are *unknown*. The agent learns the optimal **Q-Value** (expected return) for every state-action pair by observing transitions and rewards as it explores.
* **Deep Q-Learning (DQN):** Solves the scaling problem of Q-Learning by using a **Deep Q-Network (DQN)** to *approximate* the Q-Values. This section introduces two critical components for stable training:
    1.  **Experience Replay (Replay Buffer):** Stores the agent's experiences to train on random batches, breaking temporal correlations.
    2.  **Fixed Q-Value Targets (Target Network):** Uses a separate target model to generate stable target Q-Values during training.
* **TF-Agents Library:** A practical guide to Google's official RL library, demonstrating its modular components (Environments, Agents, Replay Buffers, Drivers) to build a full training pipeline.
* **Advanced Algorithms:** A brief overview of modern algorithms like Actor-Critic, PPO, and SAC.

---

## 💻 Code & Concepts Reproduced

This notebook provides code examples and theoretical explanations for the following topics:

* **Introduction to OpenAI Gym (Gymnasium):**
    * Demonstrates how to use the `gymnasium` library (`gym.make`, `reset`, `step`, `render`) to create and interact with the **CartPole-v1** environment.
    * Tests a simple, hardcoded policy to establish a baseline performance.

* **Neural Network Policies (Policy Gradients):**
    * Builds a simple `keras.Sequential` model that acts as a stochastic policy.
    * Implements the core REINFORCE algorithm (Policy Gradients) from scratch.
    * Includes helper functions to `play_one_step` (collecting gradients), `play_multiple_episodes`, and `discount_and_normalize_rewards` (to calculate action advantages).
    * Shows the full custom training loop that applies the advantage-modulated gradients.

* **MDPs and Q-Learning:**
    * Manually implements the **Q-Value Iteration** algorithm for a small, tabular MDP where all transition probabilities and rewards are known.
    * Implements the **Q-Learning** algorithm for the same environment, this time learning the Q-Values through exploration without prior knowledge of the dynamics.

* **Deep Q-Learning (DQN):**
    * Builds a `keras.Sequential` model to serve as the Deep Q-Network for the CartPole environment.
    * Implements the key components for a DQN from scratch:
        * An `epsilon_greedy_policy` for exploration.
        * A `deque` replay buffer and a `sample_experiences` function.
        * A `training_step` function that implements the **Double DQN** update logic using both an online model and a target model.

* **TF-Agents Library:**
    * Provides the code structure (imports commented out due to installation issues) for building a high-performance DQN agent to solve the **Atari Breakout** game.
    * Shows how to set up the environment with `AtariPreprocessing` and `FrameStack4`.
    * Defines the `QNetwork`, `DqnAgent`, `TFUniformReplayBuffer`, and `DynamicStepDriver` required to run the full training pipeline.

---

## 🔧 Requirements

To run this notebook, you will need the following Python libraries:
* `tensorflow` (version 2.0 or later)
* `keras` (bundled with TensorFlow)
* `numpy`
* `matplotlib`
* `gymnasium` (the modern fork of OpenAI Gym)
* `tf-agents` (Note: This library had installation issues in the notebook)
* `jupyter` (to run the notebook environment)

You can install these dependencies using `pip`:
```bash
pip install tensorflow numpy matplotlib gymnasium tf-agents
