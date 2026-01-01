# Frontier AI 365: Daily Progress Log

## Table of Contents
* [Warmup Day 0: Neural Network Foundation](#warmup-day-0-dec-20-2025)
* [Warmup Day 1: Implementing Neural Network from scratch using numpy only](#warmup-day-1-dec-21-2025)
* [Warmup Day 2: Understanding The Engine of Learning & The Architecture of Intelligence](#warmup-day-2-dec-22-2025)
* [Day 0: Manual Backpropagation Implementation & Understanding the Chain Rule](#day-0-dec-27-2025)
* [Day 1: Building a Complete Neural Network with Manual Autograd](#day-1-dec-28-2025)
* [Day 2: Starting the journey into Language Modeling – Bigram Character-Level Model](#day-2-dec-29-2025)
* [Day 3: Normalization, Sampling, and the Mathematical Reality of Negative Log Likelihood](#day-3-dec-30-2025)
* [Day 4: Character Bigram Language Model (Neural Net Version)](#day-4-dec-31-2025)
* [Day 5: Jan 01 - Building an MLP Character Language Model(part 1)](#day-5-jan-1-2026)
  
---

## Warmup Day 0: Dec 20, 2025
**Focus:** Breaking the "Black Box" of Neural Networks

### Today's Progress
* **Conceptual Deep Dive:** Worked through **3Blue1Brown's** "But what is a neural network?" to actually visualize how high-dimensional data gets transformed across layers.
* **Computer Vision Foundations:** Binged **Welch Labs'** *Learning to See* (episodes 1-16)—bridging biological vision and machine perception.
* **Implementation Math:** Finished **Welch Labs'** *Neural Networks Demystified* (1-7) to see the literal Python implementation of a basic neural network.

### Key Insights
* **Simplicity at Scale:** Still can't get over the fact that LLM "intelligence" is just billions of simple linear transformations creating emergent behavior. That's it.
* **The "Cheat" for NP-Problems:** Finally clicked that AI doesn't actually *solve* NP-hard problems mathematically—it just uses heuristics to find near-optimal solutions fast enough to be useful.
* **Stack Alignment:** Traditional ML always felt like a dead-end for my stack, but I'm realizing these foundations are non-negotiable if I want to master high-level LLM orchestration (RAG/Agents).

### 🏁 Status
* **Current Mode:** Warmup / NSU Exam Preparation.
* **Official Kickoff:** Dec 27th.

---

## Warmup Day 1: Dec 21, 2025
**Focus:** Implementing a Neural Network from scratch using NumPy only. (Code along from youtube).
**Code:** [MNIST from Scratch Notebook](../01-foundations-nn/mnist_neural_net_scratch_numpy.ipynb)

### Today's Progress
* **The MNIST Challenge:** Built and trained a 2-layer Feedforward Neural Network to classify handwritten digits using the MNIST dataset.
* **Architecture:**
    * **Input Layer:** 784 neurons (28x28 pixel images flattened).
    * **Hidden Layer:** 128 neurons with **ReLU** activation.
    * **Output Layer:** 10 neurons with **Softmax** for multi-class probability distribution.
* **Optimization & Training:**
    * Implemented **Forward Propagation** to calculate activations across layers.
    * Coded **Backpropagation** from scratch—manually applying the chain rule to compute gradients.
    * Used **Gradient Descent** with a learning rate of 0.5 to update parameters.
* **Performance:**
    * **Final Training Accuracy:** 96.61%.
    * **Final Dev Accuracy:** 95.20%.
    * Verified the model with visual predictions on the dev set—actually seeing it work hit different.

### Key Insights
* **Initialization Matters:** Switched from random weights to **He Initialization** (`np.sqrt(2. / n_prev)`)—this alone stabilized learning and prevented the gradients from vanishing/exploding in the ReLU layer.
* **The Math-Code Bridge:** Writing `dz2 = a2 - one_hot_y` gave me way more intuition about the Cross-Entropy/Softmax relationship than any theoretical explanation. You have to *see* the math execute.
* **Data Normalization:** Normalized pixel values to `[0, 1]` by dividing by 255.0—convergence was significantly faster.

### 🏁 Reflections & Next Steps
* **Scratch Pad Goal:** I built this with guidance from YouTube and LLM assistance, but I'm not at the "blank screen" level yet where I can write the entire thing from memory.
* **The "Hard Mode" Milestone:** Starting Dec 27 (semester break), I'm coming back to this. Goal: implement the entire calculus and linear algebra chain from a blank file with zero help.
* **Mastery Focus:** Understanding the "why" behind every partial derivative isn't optional—it's the foundation for everything I want to build with LLMs and RAG architectures.

---

## Warmup Day 2: Dec 22, 2025
**Focus:** The Optimization Engine & Transformer Architecture

### Today's Progress
* **The Math of Learning:** Worked through **3Blue1Brown's** Deep Learning Chapters 2–4.
    * **Gradient Descent:** Finally visualized the "cost landscape"—the negative gradient literally points in the direction of steepest descent.
    * **Backpropagation (Intuition & Calculus):** Broke down how the network figures out which weights messed up by propagating error backward through the Chain Rule.
* **Inside the LLM Black Box:** Finished Chapters 5–6 on Transformer internals.
    * **Embeddings:** Saw how words get mapped into high-dimensional vector space where semantic meaning becomes geometry.
    * **Self-Attention:** Deconstructed the Q-K-V mechanism step-by-step—this is how words dynamically update their meanings based on context.

### Key Insights
* **The Gradient is Your Only Map:** In a space with tens of thousands of dimensions (like my MNIST model), you can't "look around." The gradient is the only compass. It doesn't guarantee the global optimum, but it gets you to a local minimum that works.
* **Backprop = Recursive Blame Assignment:** The elegance of backpropagation clicked today—it's systematic error attribution. It calculates exactly how sensitive the final loss is to every single parameter, even in the earliest layers, so you can update everything simultaneously.
* **Attention Makes Context Dynamic:** This is the breakthrough that makes Transformers actually intelligent. Unlike static embeddings, Attention lets "bank" literally change its vector representation based on whether it's next to "river" or "money." Same word, different vectors, different meanings.


## Day 0: Dec 27, 2025
**Focus:** Manual Backpropagation Implementation & Understanding the Chain Rule following Karpathy's micrograd lecture.
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
* **Confidence Level:** Feeling like I actually *understand* backprop now.

---

## Day 1: Dec 28, 2025
**Focus:** Building a Complete Neural Network with Manual Autograd following Karpathy's micrograd lecture.
**Code:** [Micrograd Complete Implementation](../01-foundations-nn/micrograd_from_scratch_complete.ipynb)

### Today's Progress
* **Full Autograd Engine:** Expanded yesterday's `Value` class into a complete automatic differentiation system.
* **Implemented Key Operations:**
    * Added support for subtraction (`__sub__`), negation (`__neg__`), power (`__pow__`), division (`__truediv__`)
    * Implemented reverse operations (`__radd__`, `__rmul__`, `__rsub__`) to handle cases like `2 * Value(3)`
    * Built the `tanh` activation function with proper gradient computation: `(1 - t²) * out.grad`
* **Topological Sort:** Wrote the recursive `build_topo()` function to traverse the computational graph in the correct order for backpropagation.
* **Automatic Backward Pass:** Created the `.backward()` method that:
    * Sets the output gradient to 1.0
    * Traverses the graph in reverse topological order
    * Calls each node's `._backward()` to propagate gradients
* **Built a Neural Network from Scratch:**
    * `Neuron` class: Single neuron with weights, bias, and tanh activation
    * `Layer` class: Collection of neurons
    * `MLP` class: Multi-layer perceptron with configurable architecture
* **Training Loop:** Implemented complete gradient descent:
    * Forward pass to compute predictions
    * Loss computation (mean squared error)
    * Backward pass via `.backward()`
    * Manual weight updates: `p.data += -0.05 * p.grad`

### Key Insights
* **Why Topological Sort Matters:** You can't just backprop in any order - you need to ensure a node's gradients from all paths are accumulated *before* propagating to its children. Topological sort guarantees this.
* **The Power of Operator Overloading:** Making `Value` objects work with `+`, `*`, `-`, `/` means you can write neural network code that looks like normal math but secretly builds a computational graph. That's exactly what PyTorch does under the hood.
* **Zero-Ing Gradients is Critical:** Before each backward pass, you MUST set all gradients to zero. Otherwise, gradients accumulate across iterations and you get completely wrong updates. This tripped me up initially. Funnily Karpathy himself did it wrong first time. 
* **Watching Loss Decrease is Satisfying:** Seeing the loss drop from `5.77 → 0.033` over 20 iterations was the moment everything clicked. The network actually *learned* from just 4 examples.

### Validation Against PyTorch
Implemented the exact same network in PyTorch and compared gradients - they matched perfectly (within floating-point precision). This confirmed my implementation is mathematically correct.

### The Breakthrough Moment
When I ran the training loop and watched `ypred` go from random garbage to nearly matching `[1.0, -1.0, -1.0, 1.0]`, it hit me: this is *literally* how all neural networks learn. Just this loop, scaled up billions of times. Everything from GPT to Stable Diffusion is fundamentally doing what I just coded.

### 🏁 Status
* **Current Milestone:** Built a complete working neural network with manual backprop from absolute scratch.
* **Confidence Level:** I don't just understand backprop anymore - I can *implement* it. Big difference.

## Day 2: Dec 29, 2025
**Focus:** Starting the journey into Language Modeling – Bigram Character-Level Model (following Andrej Karpathy's "The spelled-out intro to language modeling: building makemore")
**Code:** [Bigram Language Model Notebook](../01-foundations-nn/bigram_lm.ipynb)

### Today's Progress
* **Dataset:** Loaded the classic `names.txt` dataset – 32,000+ names, perfect for character-level modeling.
* **Exploratory Analysis:**
    * Shortest name: 2 characters, longest: 15.
    * Visualized the most common bigrams by manually counting transitions (adding start `<S>` and end `<E>` tokens as `.`).
* **Bigram Counting:**
    * Augmented every name with start/end tokens: `['.', ...chars, '.']`
    * Built a frequency dictionary of all character bigrams → character transitions.
    * Sorted and printed the top bigrams – immediately saw patterns like `(n, .)` being the most common ending (names ending in "n"), `(a, .)` second, etc.
* **Key Observations from Counts:**
    * Strong patterns emerged: names love ending in "a", "n", "y", "e".
    * Common starts: `.` → "a", "k", "m", "j", "s".
    * Some letters almost never follow others (e.g., very few "q" not followed by "u").

### Key Insights
* **Language Modeling is Just Predicting the Next Character:** At its core, a bigram model is literally asking "given this character, what's the most likely next one?" Everything else (attention, transformers, GPT) is just a fancier way to answer that same question.
* **Counting = Learning:** This naive counting approach is actually a maximum likelihood estimator. The "parameters" are just the frequencies. No gradients, no backprop – yet it already captures real structure in names.
* **The Power of Start/End Tokens:** Adding `.` at both ends is genius. It lets the model learn both how names start and how they end in one unified framework.
* **Sleepy Brain Still Works (Sometimes):** Was fighting heavy eyelids all day – ended up napping mid-study session. Progress was slower than I wanted, but even in this tired state, seeing those sorted bigrams pop out felt like unlocking a tiny piece of how language works.

### 🏁 Status & Reflections
* **Current Milestone:** Solid foundation – bigram counts are done, data is explored. Next: turn these counts into probabilities, implement sampling, and compute loss (negative log likelihood).
* **Honest Note:** Only got this far today because energy crashed hard. Feeling a bit frustrated with myself, but reminding myself this is a marathon. The concepts are clicking even when I'm half-asleep.
* **Next Step:** Finish the bigram model tomorrow inshaAllah – implement the probability matrix, generate names, evaluate with loss, then move to the neural net version.
* **Confidence Level:** The bridge from neural nets to language modeling is starting to form. Excited to see this evolve into actual name generation.

Still grinding. One day at a time. InshaAllah tomorrow will be stronger.


## Day 3: Dec 30, 2025
**Focus:** Normalization, Sampling, and the Mathematical Reality of Negative Log Likelihood (following Andrej Karpathy's "The spelled-out intro to language modeling: building makemore")

**Code:** [Character bigram language model](https://github.com/m-mahadi/frontier-ai-365/blob/eee2580f43a35c3f6d3b59a157398d274c0686fb/01-foundations-nn/character%20bigram%20language%20model.ipynb)

### Today's Progress
* **Vectorized Normalization:** 
    * Converted the count matrix `N` to a probability matrix `P`.
    * **The Broadcasting Battle:** Spent 30 minutes debugging why `P = N / N.sum(1)` wasn't throwing an error but was silently destroying my math.
    * **The Fix:** Used `keepdim=True` to force the divisor into shape `(27, 1)`, ensuring proper row-wise normalization. Without it, PyTorch was broadcasting a `(27,)` vector column-wise, turning my probability distribution into complete nonsense.
* **Generating Names (Sampling):**
    * Used `torch.multinomial` to sample the next character based on the current row's probability distribution.
    * Realized that even with broken math, the output looked identical due to fixed seeds and `multinomial`'s internal auto-normalization.
    * Generated names like `cexze`, `miana`, and `ss`. Clearly bigram behavior—phonetically plausible but zero long-term coherence.
* **Model Evaluation (Negative Log Likelihood):**
    * Calculated the **Likelihood** as the product of all actual bigram probabilities in the dataset.
    * Switched to **Log-Likelihood** to avoid numerical underflow (summing logs beats multiplying tiny floats).
    * Implemented **Average NLL** as the loss function.
    * **Final Baseline Loss:** ~2.454.
* **Model Smoothing:** Added a "fake count" of 1 to all bigrams (`P = (N+1).float()`) to prevent `log(0) = -∞` when hitting unseen character pairs like 'jq'.

### Key Insights
* **Tensor Contiguity is Everything:** A tensor isn't a grid - it's metadata (shape/strides) wrapped around a flat 1D memory block. This is why `view()` is free but `transpose()` often needs `.contiguous()` before certain ops.
* **Broadcasting is a Silent Killer:** It makes code concise but hides catastrophic logic errors. If you're reducing dimensions, always explicitly verify *which* dimension you're collapsing.
* **The NLL Intuition:** Minimizing NLL = maximizing the probability the model assigns to real training data. If the model is "surprised" by an actual name, loss explodes.
* **Bigram = Glorified Lookup Table:** It has zero memory beyond one character. To get better names, you need more context (trigrams) or a neural network that can learn representations.

### The "Aha" Moment
That 30-minute broadcasting bug wasn't a dumb mistake - it was a failure to respect the **Trailing Dimension Rule**. In PyTorch, broadcasting starts from the *right*. By skipping `keepdim`, I let the engine prepend a `1` to the left, which completely shifted the operation's semantics. This matters everywhere - normalization in embeddings, attention scores, loss calculations. Get this wrong and your model trains on garbage data while looking perfectly fine.

### 🏁 Status & Reflections
* **Current Milestone:** Finished the "counting" version of the Bigram model. Baseline loss: 2.454.
* **Confidence Level:** High. I understand the *why* behind every operation in the probability matrix.
* **Next Step:** Build the Neural Network version of the Bigram model. Instead of counting, use a single linear layer + Softmax to hit the same 2.454 loss via gradient descent.



## Day 4: Dec 31, 2025
**Focus:** From Lookup Tables to Gradient Descent: The Neural Network Reincarnation of the Bigram Model (following Andrej Karpathy's "The spelled-out intro to language modeling: building makemore")

**Code:** [Character Bigram Language Model (Neural Net Version)](https://github.com/m-mahadi/frontier-ai-365/blob/eee2580f43a35c3f6d3b59a157398d274c0686fb/01-foundations-nn/character_bigram_language_model_nn.ipynb)

### Today's Progress
* **The Single-Layer Architecture:**
    * Implemented a minimal "neural network" consisting of a single linear layer with 27 neurons, each receiving 27 inputs (the one-hot encoded characters).
    * Weight Initialization: Initialized $W \in \mathbb{R}^{27 \times 27}$ using `torch.randn` with `requires_grad=True`.
* **The Differentiable Forward Pass:**
    * One-Hot Encoding: Transformed integer indices `xs` into a sparse bitmask.
    * Logits to Probs: Calculated `logits = xenc @ W`, followed by manual Softmax implementation: `counts = logits.exp()` and `probs = counts / counts.sum(1, keepdims=True)`.
    * Broadcasting Reinforcement: Applied the `keepdims=True` lesson from yesterday to ensure normalization happened across the correct axis.
* **Optimization Loop:**
    * Implemented the Negative Log Likelihood (NLL) loss function using specialized tensor indexing: `-probs[torch.arange(num), ys].log().mean()`.
    * Backpropagation: Used `loss.backward()` to compute gradients and updated weights via basic SGD: `W.data += -50 * W.grad`.
    * Regularization (Weight Decay): Added a smoothing term `0.01 * (W**2).mean()` to the loss. This is the gradient-descent equivalent of the "add-one smoothing" I used in the counting model.
* **Results:** Successfully converged to the exact same 2.454 loss as the counting-based approach.

### Key Insights
* **Linear Layer = Differentiable Lookup Table:** When the input is a one-hot vector, the matrix multiplication `xenc @ W` is mathematically identical to selecting a specific row of $W$. The weights are essentially "learning" the log-counts of character transitions.
* **Softmax is a Bridge:** It turns unconstrained raw numbers (logits) into a valid probability distribution. The `exp()` ensures positivity, and the normalization ensures the sum equals 1.
* **Stochasticity vs. Determinism:** The model is "trained" via gradient descent, but the output generation remains stochastic because we sample from the resulting distribution using `torch.multinomial`.

### 🏁 Status & Reflections
* **Current Milestone:** Neural Network Bigram model complete. Loss matched baseline (2.454).
* **Confidence Level:** High. The connection between the count-based probability matrix and the optimized weight matrix is now crystal clear.
* **Year-End Reflection:** Ending 2025 by understanding that a neural network isn't magic—it's just optimization over a parameter space. The same problem (predicting the next character) solved two ways: once by counting, once by gradient descent. Both converge to the same answer. That's the beauty of it.

Tomorrow: Embeddings. The representation layer that makes everything else possible. InshaAllah.

## Day 5: Jan 1, 2026
**Focus:** Breaking the Bigram Barrier - Building an MLP Character Language Model

**Code:** [MLP Character Language Model](https://github.com/m-mahadi/frontier-ai-365/blob/eee2580f43a35c3f6d3b59a157398d274c0686fb/01-foundations-nn/MLP%20part%201.ipynb)

### Today's Progress
* **The Architecture Shift:**
    * Upgraded from 1-character context (Bigram) to a **Fixed-Size Window** approach. Set `block_size = 3` - the model now looks at the previous 3 characters to predict the 4th.
    * **Embedding Layer:** Instead of one-hot encoding (wasteful), implemented a lookup table `C ∈ ℝ^(27×2)`. This projects 27 discrete characters into a continuous 2D embedding space.
* **The Forward Pass:**
    * **Embedding Lookup:** `emb = C[X]` gives us shape `[Batch, Block_Size, Embed_Dim]`.
    * **The `.view()` Trick:** Instead of using `torch.cat()` (which copies memory), used `emb.view(-1, 6)` to flatten the context. Zero-copy operation because tensors are just metadata over contiguous memory.
    * **Hidden Layer:** `W1 ∈ ℝ^(6×100)` and `b1 ∈ ℝ^100` with `tanh` activation.
    * **Output Layer:** `W2 ∈ ℝ^(100×27)` and `b2 ∈ ℝ^27` to produce logits.
* **Loss Function Upgrade:**
    * Ditched manual softmax calculation for `F.cross_entropy`.
    * **Why this matters:** `F.cross_entropy` uses the log-sum-exp trick for numerical stability. It never instantiates the full probability tensor - goes straight from logits to loss. Way more efficient.

### Key Insights
* **Distributed Representations = The Entire Game:** In the bigram model, 'a' and 'b' are just indices - no relationship between them. With embeddings, the model learns that 'a' and 'e' (both vowels) should be "close" in vector space. This is the foundational principle behind every LLM.
* **`.view()` is Free:** Reconfirmed the memory contiguity lesson. Tensors are just metadata (shape + strides) over a 1D memory block. `.view()` changes the metadata without touching data. O(1) operation. Beautiful.
* **Parameter Sharing Scales:** A counting approach with `block_size=3` would need a 27³ table (19,683 entries). The MLP handles this by sharing parameters across positions through embeddings and hidden layers. This is how neural nets compress logic.
* **Why Tanh Instead of ReLU?** Following Bengio et al. 2003's original paper. ReLU came later. Want to see if the architecture choices from 2003 still work today 

### The Architecture in Action

The model flow:
```
Input: "hel" → [7, 4, 11]  (character indices)
Embedding: C[[7,4,11]] → [[e₁], [e₂], [e₃]]  (each eᵢ ∈ ℝ²)
Flatten: view(-1, 6) → [e₁, e₂, e₃]  (concatenated into ℝ⁶)
Hidden: tanh(xW₁ + b₁) → h ∈ ℝ¹⁰⁰
Output: hW₂ + b₂ → logits ∈ ℝ²⁷
Loss: F.cross_entropy(logits, target='l')
```

### The Scaling Realization

**Bigram limitations:**
- Only sees 1 character back
- No parameter sharing
- Can't capture patterns longer than 2 characters

**MLP improvements:**
- Sees 3 characters back
- Shares embedding parameters across positions
- Can learn context-dependent patterns

But still has a **fatal flaw**: fixed context window. Can't handle variable-length context. 















