# 🧠 Week 1: Foundations & Setup (14 May 2025, 14:16)

## 🚀 Goal:
Get a solid intuition of **Spiking Neural Networks (SNNs)** and **Reinforcement Learning (RL)**—from bio-inspired neuron models to agent-based decision-making frameworks. This week is about understanding the "why" before diving into the "how".

---

## 🧩 Spiking Neural Networks (SNNs)

### 🔗 Resources:
- [GeeksforGeeks Intro](https://www.geeksforgeeks.org/spiking-neural-networks-in-deep-learning/)
- [Cnvrg.io Overview](https://cnvrg.io/spiking-neural-networks/)
- 🎥 [YT: SNN Explained Visually](https://www.youtube.com/watch?v=GTXTQ_sOxak)

### 🔍 Key Takeaways:
- **SNN ≠ traditional ANN**: Neurons fire *only* when membrane potential crosses threshold (spikes, not floats).
- **Temporal encoding**: SNNs use *spike timing* to encode information (great for energy efficiency + dynamic input like vision/audio).
- **Biological realism**: Implements LIF (Leaky Integrate-and-Fire), STDP (Spike-Timing Dependent Plasticity) — closer to real brains.
- **SNN vs ANN**: No ReLU, no sigmoid — it's all about spikes & decay.
- **Use-cases**: Event-driven sensing, robotics, neuromorphic hardware (Loihi, TrueNorth, etc.).

### 🧠 Insight:
SNNs are not just about mimicking biology for the sake of it — their **temporal sparsity and energy efficiency** make them ideal for low-power, real-time tasks. Think edge devices with actual brains.

---

## 🤖 Reinforcement Learning (RL)

### 🔗 Resources:
- [GFG: RL for Beginners](https://www.geeksforgeeks.org/spiking-neural-networks-in-deep-learning/) (yep, they accidentally tagged it SNNs—still a good primer)
- 🎥 [YT: RL Crash Course by Simplilearn](https://www.youtube.com/watch?v=Mut_u40Sqz4)

### 🔍 Key Takeaways:
- **Core concept**: Agent learns *by trial and error*, using rewards as feedback.
- **Main loop**: State → Action → Reward → Update policy (exploration vs exploitation)
- **Value-based**: Q-learning, SARSA
- **Policy-based**: Policy Gradient, PPO
- **Terms to lock in**:
  - `Policy`: what action to take
  - `Reward`: feedback
  - `Value`: expected future reward
  - `Environment`: the "world" the agent lives in
- **Bellman equation** is the backbone for most algorithms.

### 🧠 Insight:
What backprop is to supervised learning, **value iteration + reward feedback** is to RL. It’s not about hardcoding intelligence — it's about letting agents *figure things out* over time.

---

## 📝 Summary

This week was all about laying the foundation:
- 🔌 SNNs bring **biological timing** and **energy-efficient computation** to the table.
- 🧭 RL is how agents **learn by doing**, using **rewards instead of labels**.
- Both are *adaptive systems* — one mirrors brains, the other mimics behavior.

---


