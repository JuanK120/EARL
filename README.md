# EARL: Explanations using Alternative Realities for Reinforcement Learning

EARL is an open-source Python library for generating counterfactual explanations in reinforcement learning (RL) settings. It enables researchers and developers to inspect and understand the decisions of black-box RL agents through high-level, user-friendly "what-if" scenarios. EARL supports multiple explanation strategies and is designed for realistic, self-adaptive systems beyond toy environments.

This library has been evaluated in the CitiBikes simulation environment, a self-adaptive bike-sharing system, and supports integration with common RL algorithms such as PPO and DQN.

---

## Installation and Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/JuanK120/EARL/ EARL
   cd EARL
   ```

2. **Install dependencies using pip:**
   If using pip:
   ```bash
   pip install -r requirements.txt

3. **(Optional) Train your own PPO or DQN agent:**
   The library provides default wrappers with training capability. You may also use pretrained agents.

4. **Environment setup:**
   - EARL was tested on models built using [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3).
   - Example integration is provided for the CitiBikes bike-sharing simulation. Additionally, as CitiBikes simulation is a multi discrete action space we also provide a simpler example for a discrete action space using gym's lunarlander game. both implementation can be found on the respective folders '/citibikes' or '/gymnasium_example' respectively.

5. **Docker Image**
    Additionally our library also goes with a built docker image for ease of reproducibility of our experiments. We provide a pre-built Docker image to simplify the setup and ensure reproducibility of our experiments. If you'd like to build the image yourself, you can run:

    `docker build -t earl .`

   Or Pull the docker image from dockerhub: 

   `docker pull juanrosero/earl-rl:latest`

    Once built, you can run the citibikes experiment by running:

    `docker run -it --rm earl`


---

## Running

To run EARL on a pretrained agent you can use one of our already created scripts, either citibikes, or lunarLander:

```bash
python run_<example>.py
```

During execution, EARL will:
- Load the environment and model
- Generate counterfactuals for selected informative states
- Log explanation metrics and output results

---

## Results

Explanation outputs and metrics are stored under the `/results/` directory of any of the examples and include:

- **Explanation sets:** Counterfactual examples with input and output actions
- **Performance logs:** Generation time, similarity, plausibility, diversity
- **Evaluation reports:** CSV/JSON files summarizing evaluation metrics per method

---

## How to Extend

EARL is designed for extensibility in several ways:

- **Adding new explanation methods:** Implement a new strategy by creating a class under `explanations/` that inherits from the `BaseExplainer` interface.
- **Custom environments:** Use the provided wrappers to adapt EARL to any Gym-compatible environment. Ensure your environment supports:
  - `reset(state)`
  - `get_actions()`
  - `check_done(state)`
  - `equal_states(state1, state2,)`
- **Model compatibility:** Add wrappers for new RL algorithms by extending `model_wrappers/`.

---

## Citation
