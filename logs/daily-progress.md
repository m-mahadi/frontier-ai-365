# Frontier AI 365: Daily Progress Log

## Table of Contents
* [Warmup Day 0: Neural Network Foundation](#warmup-day-0-dec-20-2025)
* [Warmup Day 1: Implementing Neural Network from scratch using numpy only](#warmup-day-1-dec-21-2025)
* [Day 1: Kickoff (Coming Dec 27)](#day-1)

---

## Warmup Day 0: Dec 20, 2025
**Focus:** Breaking the "Black Box" of Neural Networks

### Today's Progress
* **Conceptual Deep Dive:** Completed **3Blue1Brown’s** "But what is a neural network?" to visualize how high-dimensional data is transformed across layers.
* **Computer Vision Foundations:** Watched **Welch Labs'** *Learning to See* (1-16), bridging the gap between biological vision and machine perception.
* **Implementation Math:** Finished **Welch Labs'** *Neural Networks Demystified* (1-7) to see the literal Python implementation of the fundamental neural network.

### Key Insights
* **Simplicity at Scale:** It is astonishing that the "intelligence" of an LLM is essentially an emergent property of billions of simple linear transformations.
* **The "Cheat" for NP-Problems:** Realized that AI doesn't mathematically solve NP-hard problems; it uses heuristics to find near-optimal solutions efficiently.
* **Stack Alignment:** Traditional ML felt like a dead-end for my stack, but understanding these foundations is necessary to master high-level LLM orchestration (RAG/Agents).

### 🏁 Status
* **Current Mode:** Warmup / NSU Exam Preparation.
* **Official Kickoff:** Dec 27th.


---

## Warmup Day 1: Dec 21, 2025
**Focus:** Implementing a Neural Network from scratch using NumPy only

### Today's Progress
* **The MNIST Challenge:** Successfully built and trained a 2-layer Feedforward Neural Network to classify handwritten digits using the MNIST dataset.
* **Architecture:** * **Input Layer:** 784 neurons (representing 28x28 pixel images).
    * **Hidden Layer:** 128 neurons utilizing the **ReLU** activation function.
    * **Output Layer:** 10 neurons utilizing the **Softmax** activation function for multi-class probability distribution.
* **Optimization & Training:**
    * Implemented **Forward Propagation** to calculate activations across layers.
    * Coded **Backpropagation** from scratch, manually applying the chain rule to compute gradients.
    * Used **Gradient Descent** to update parameters with a learning rate of 0.5.
* **Performance:** * **Final Training Accuracy:** 96.61%.
    * **Final Dev Accuracy:** 95.20%.
    * Successfully verified the model with visual test predictions on the development set.

### Key Insights
* **Initialization Matters:** Upgraded from simple random weights to **He Initialization** (`np.sqrt(2. / n_prev)`), which significantly stabilized the learning process and prevented vanishing/exploding gradients in the ReLU layer.
* **The Math-Code Bridge:** Writing the code for `dz2 = a2 - one_hot_y` provided a much deeper intuition for the relationship between the Cross-Entropy loss function and the Softmax output than theoretical reading alone.
* **Data Normalization:** Normalized input pixel values to a `[0, 1]` range by dividing by 255.0 to ensure efficient convergence during training.

### 🏁 Reflections & Next Steps
* **Scratch Pad Goal:** While I successfully implemented the network today with guidance from YouTube and LLM thought partners, I am not yet at the "blank screen" level where I can write the entire architecture from memory.
* **The "Hard Mode" Milestone:** During the semester break (starting Dec 27), I will return to this project. My goal is to implement the entire calculus and linear algebra chain from a blank file with zero assistance.
* **Mastery Focus:** Understanding the "why" behind every partial derivative is a non-negotiable step in my journey toward mastering LLM foundations and RAG architectures.
