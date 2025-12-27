# Frontier AI 365: Daily Progress Log

## Table of Contents
* [Warmup Day 0: Neural Network Foundation](#warmup-day-0-dec-20-2025)
* [Warmup Day 1: Implementing Neural Network from scratch using numpy only](#warmup-day-1-dec-21-2025)
* [Warmup Day 2: Understanding The Engine of Learning & The Architecture of Intelligence](#warmup-day-2-dec-22-2025)
* [Day 1: Manual Backpropagation Implementation & Understanding the Chain Rule](#day-1-dec-27-2025)

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
**Code:** [MNIST from Scratch Notebook](../01-foundations-nn/mnist_neural_net_scratch_numpy.ipynb)

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


## Warmup Day 2: Dec 22, 2025
**Focus:** The Engine of Learning & The Architecture of Intelligence

### Today's Progress
* **The Optimization Engine:** Completed **3Blue1Brown's** Deep Learning Chapters 2–4.
    * **Gradient Descent:** Visualized the "cost landscape" and understood the negative gradient as the direction of steepest descent to minimize error.
    * **Backpropagation (Intuition & Calculus):** Deconstructed how the network "assigns blame" for errors by propagating nudges backward through the layers using the Chain Rule.
* **The Transformer Breakthrough:** Completed Chapters 5–6 on LLM internals.
    * **Embeddings:** Explored how semantic meaning is mapped into high-dimensional vector space.
    * **Self-Attention Mechanism:** Understood the step-by-step logic of how words update their meanings based on context using Queries (Q), Keys (K), and Values (V).

### Key Insights
* **The Gradient as a Compass:** Realized that in a multi-thousand-dimensional space (like my MNIST model), the gradient is the only reliable way to navigate. It doesn't find the global best solution, but it effectively finds a "good enough" local minimum.
* **Recursive Attribution:** Backpropagation is essentially the mathematical implementation of "Error Attribution." It tracks how sensitive the final cost is to every single weight and bias in the early layers, allowing for precise, simultaneous updates across the entire architecture.
* **Dynamic Context:** In Transformers, the attention mechanism is what makes LLMs "smart." Unlike fixed word-to-vector mappings, Attention allows the word "bank" to literally change its mathematical value based on whether the query finds the key "river" or the key "money."


## Day 1: Dec 27, 2025
**Focus:** Manual Backpropagation Implementation & Understanding the Chain Rule
**Code:** [Micrograd from Scratch Notebook](../01-foundations-nn/micrograd_from_scratch.ipynb)

### Today's Progress
* **The Micrograd Challenge:** Built my own autograd engine from scratch by manually implementing backpropagation.
* **Implementation Highlights:**
    * Created a custom `Value` class to track computational graph operations (addition, multiplication).
    * Implemented forward pass for simple neural computations.
    * **Manually coded backward pass** - this is where the magic happened. I literally computed each gradient by hand, tracking how each intermediate value contributes to the final loss.
* **Visualization:** Used Graphviz to render the computational graph, making the chain rule visually obvious - seeing how gradients flow backward through nodes was mind-blowing.

### Key Insights
* **The Chain Rule = Gradient Highway:** Backprop is just the chain rule applied recursively. Each node multiplies its local gradient by the gradient flowing from above. It's elegant as hell once you see it.
* **Manual Implementation > Theory:** Actually writing `dL/da`, `dL/db`, `dL/dc` by hand gave me 10x more intuition than reading about it. The formula `∂L/∂a = (∂L/∂c) * (∂c/∂a)` finally clicked.
* **Why Autograd is Powerful:** After doing this manually, I deeply appreciate PyTorch's autograd. What I coded in 100+ lines, PyTorch does automatically with `.backward()`. But now I know *exactly* what's happening under the hood.

### The Breakthrough Moment
When I plotted the computational graph and manually traced gradients backward from the loss `L` through `d`, `e`, and finally to `a` and `b`, the entire neural network training loop suddenly made perfect sense. Backpropagation isn't magic - it's just calculus, organized.

### 🏁 Status
* **Current Mode:** Full Deep Dive / Semester Break Begins Today.
* **Next Step:** Implement a full neural network with this autograd engine (multi-layer, activation functions, etc.).
* **Confidence Level:** Feeling like I actually *understand* backprop now, not just memorize it.

