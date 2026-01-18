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
* [Day 5: Building an MLP Character Language Model(part 1)](#day-5-jan-1-2026)
* [Day 6: Scaling with MLP - Mini-batching, Learning Rate Search, and the Train/Dev/Test Split](#day-6-jan-2-2026)
* [Day 7: Hyperparameter Warfare - MLP Self-Implementation and Crushing the Baseline](#day-7-jan-3-2026)
* [Day 8: Fixing Dead Neurons and the Hockey Stick Loss](#day-8-jan-4-2026)
* [Day 9: Reading the Paper That Started It All - Bengio et al. (2003)](#day-9-jan-5-2026)
* [Day 10: Batch Normalization Deep Dive - Understanding Why Networks Break](#day-10-jan-6-2026)
* [Day 11: Backpropagation Ninja Training - Manual Gradient Calculation Through Batch Normalization](#day-11-jan-7-2026)
* [Day 12: Backpropagation Through BatchNorm - The Calculus Reality Check](#day-12-jan-8-2026)
* [Day 13: Research Deep Dive - Multi-Hop Reasoning & Knowledge Graph RAG Architectures](#day-13-jan-9-2026)
* [Day 14: Building GPT from Scratch - Character-Level Language Modeling & Self-Attention Foundations(part 1)](#day-14-jan-10-2026)
* [Day 15: Completing Karpathy's Let's build GPT](#day-15-jan-11-2026)
* [Day 16: Understanding Tokenization - From Character-Level to Byte Pair Encoding](#day-16-jan-12-2026)
* [Day 17: Attention Mechanics & The KV Cache](#day-17-jan-13-2026)
* [Day 18: Byte Pair Encoding Implementation - Building a Tokenizer from Scratch](#day-18-jan-14-2026)
* [Day 19: The Regex Hell - Research Work Day](#day-19-jan-15-2026)
* [Day 20: Matrix Calculus Foundations - The Math Behind Backpropagation](#day-20-jan-16-2026)
* [Day 21: Positive Definite Matrices & GraphRAG Implementation](#day-21-jan-17-2026)
* [Day 22: Eigenvalues, Eigenvectors, and the Spectral Theorem](#day-22-jan-17-2026)

  

  
  
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


## Day 6: Jan 2, 2026
**Focus:** Scaling with MLP - Mini-batching, Learning Rate Search, and the Train/Dev/Test Split

**Code:** [MLP Finished Implementation](https://github.com/m-mahadi/frontier-ai-365/blob/cd495fc45764ee347654538355ed58e11ca542cd/01-foundations-nn/MLP_finished.ipynb)

### Today's Progress
* **Data Pipeline Setup:**
    * Implemented the **80/10/10 split** (Training, Dev, Test).
    * Training set: optimizes parameters (W, b)
    * Dev set: tunes hyperparameters (hidden size, embedding dim, learning rate)
    * Test set: kept locked away to evaluate final model - no peeking allowed
* **Mini-batching:**
    * Switched from full-batch gradient descent to **mini-batches (size 32)**.
    * Used `torch.randint` to sample indices. Way faster iteration even though the gradient estimate is noisier.
* **Learning Rate Search:**
    * Did a principled search: created `linspace(-3, 0)`, applied as exponent `10^x`, plotted loss.
    * Found `0.1` was solid starting point, with decay to `0.01` as loss plateaued.
* **Architecture Scaling:**
    * Expanded to **200 hidden neurons** and **10-dimensional embeddings**.
    * Total parameters: **11,897**.
    * Final Dev Loss: **~2.87**.
* **Embedding Visualization:**
    * Plotted the 2D character embeddings.
    * We should have observed the model clustering vowels (a, e, i, o, u) together and separating the "." token however our model didn't do this.
    * I will try to fix it tomorrow God willing.
    * 
### Key Insights
* **Why `F.cross_entropy` Matters:** It uses the **log-sum-exp trick** - subtracts max logit before exponentiating to prevent `inf` from large positive numbers. Not just cleaner code, it's required for numerical stability in deep networks.
* **The `.view()`:** When flattening `[32, 3, 10]` to `[32, 30]`, `.view()` is O(1) because it just changes the tensor's **stride** metadata. No data movement. This is how you think about efficiency in production.
* **Overfitting Signal:** Training loss slightly lower than dev loss. Classic sign the model is starting to memorize the 182k training examples instead of generalizing.

### The Problem I Hit

**Dev loss (~2.87) is *higher* than the Bigram baseline (~2.45).**

This shouldn't happen. Here's my analysis:

**What's going wrong:**
- Bigram uses an exhaustive lookup table - perfect counting with zero compression
- MLP with `block_size=3` has 27³ = 19,683 possible input combinations
- My model only has ~11k parameters
- I'm compressing the language representation too aggressively

**Possible causes:**
1. **Bugs in the code** - most likely, there are errors somewhere in the forward pass or loss calculation
2. **Hyperparameter issues** - learning rate, hidden size, or embedding dimension suboptimal
3. **Architectural bottleneck** - fixed 3-character window is too limiting

**The harsh truth:** If I can't beat a simple counting baseline with a neural network, something fundamental is wrong. Either the code has bugs or I'm not giving the model enough capacity.


### 🏁 Status & Reflections
* **Tomorrow's Mission:** Debug the code from scratch. Check every tensor shape, verify the loss calculation, tune hyperparameters systematically. Goal: get dev loss down to ~2.0 and actually beat the bigram baseline.
* **Confidence Level:** Medium. I understand the *architecture*, but clearly something in the *implementation* is broken. Time to debug.

## Day 7: Jan 3, 2026
**Focus:** Hyperparameter Warfare - MLP Self-Implementation and Crushing the Baseline
**Code:** 
- [MLP Self-Implementation - 2D/100 Units](https://github.com/m-mahadi/frontier-ai-365/blob/208a1c274ae136c1dbb10dd223cb3650b72c6f09/01-foundations-nn/MLP_self_implementation%20(2).ipynb)
- [MLP Self-Implementation - 2D/200 Units](https://github.com/m-mahadi/frontier-ai-365/blob/208a1c274ae136c1dbb10dd223cb3650b72c6f09/01-foundations-nn/MLP_self_200_hidden_layers.ipynb)
- [MLP Self-Implementation - 7D/200 Units](https://github.com/m-mahadi/frontier-ai-365/blob/208a1c274ae136c1dbb10dd223cb3650b72c6f09/01-foundations-nn/ML_self_implementation_200_hl_%2B_7_dimensions.ipynb)
- [MLP Self-Implementation - 10D/200 Units](https://github.com/m-mahadi/frontier-ai-365/blob/208a1c274ae136c1dbb10dd223cb3650b72c6f09/01-foundations-nn/ML_self_implementation_200_hl_%2B_10_dims.ipynb)

### Today's Progress
* **Total Ownership via Self-Implementation:** Instead of just following Karpathy's lecture, I rewrote the entire MLP architecture from scratch four different times. This forced me to internalize every tensor operation without relying on templates.
* **The Hyperparameter Grid Search:**
    * **Variation 1 (Baseline):** 2D embeddings + 100 hidden neurons (~3.4k params). Fast to train but hit a representational bottleneck early. Underfitted significantly.
    * **Variation 2 (More Capacity):** 2D embeddings + 200 hidden neurons (~6.8k params). Marginal improvement. Still bottlenecked by the 2D embedding space.
    * **Variation 3 (The Winner):** 7D embeddings + 200 hidden neurons (~10k params). This was the sweet spot. Reached **training loss ~2.09, dev loss ~2.17**. Finally crushed the bigram baseline of 2.45.
    * **Variation 4 (Too Much Capacity):** 10D embeddings + 200 hidden neurons (~11.8k params). Pushed training loss toward 2.0, but dev loss stayed higher. Clear overfitting - the model was memorizing training sequences instead of learning phonetic patterns.
* **Optimization Deep Dive:**
    * Started with learning rate `0.1`, stepped down to `0.01`, finally settled at `0.001` for the tail end of training.
    * Ran up to **200k iterations** on the 7D model to let it fully settle into the loss valley.
    * Verified that `F.cross_entropy` provides rock-solid numerical stability compared to manual softmax.

### Key Insights
* **The Overfitting Frontier:** The 10D experiment taught me a critical lesson - more parameters aren't always better. At 10 dimensions, the character latent space became so sparse that the model started memorizing training noise. The gap between training and dev loss widened significantly.
* **Embedding Dimension > Hidden Layer Size:** Going from 100 to 200 hidden neurons provided less gain than increasing embedding dimensionality from 2D to 7D. The bottleneck in my Day 7 disaster was the **input representation**, not the processing power of the hidden layer.
* **Learning Rate Decay is Non-Negotiable:** Keeping a constant `0.1` learning rate causes noisy oscillations near local minima. Stepping down to `0.001` lets the weights settle into deeper valleys. This is why modern optimizers have built-in schedulers.
* **Distributed Representations Work:** Even with overfitting, the 10D model showed vowels and consonants clustering with much higher resolution. This is where the "intelligence" lives - the model learned semantic similarity without being told what a vowel is.

### The Reality Check

**Yesterday's disaster (loss 2.87) → Today's victory (loss 2.09).**

What changed?
1. **Fixed the code bugs** (there were errors in the forward pass)
2. **Scaled embedding dimensions** from 2D to 7D
3. **Ran more iterations** (200k vs 50k)
4. **Proper learning rate decay** instead of constant rate

The 7D model proves the MLP architecture works. The 10D model proves you can have too much capacity for the data size.

### Architecture Bottlenecks

Even with perfect hyperparameters, I'm still limited by `block_size=3`. This is a **Trigram MLP** - it's blind to any character at position n-4 or earlier.

To truly beat 2.0 loss, I don't need more embedding dimensions.

### 🏁 Status & Reflections
* **Current Milestone:** Consistently beating statistical baselines. Can now diagnose whether a model is bottlenecked by embedding dimensionality or hidden layer capacity just by looking at loss curves.
* **What I Learned About Debugging:**
    - Yesterday's failure wasn't wasted time - it taught me how to systematically isolate problems
    - Loss curves tell you everything if you know how to read them
* **Next Step:** Moving to Part 3 of makemore - Kaiming Initialization and Batch Normalization. Time to stop guessing about weight initialization and understand the math that keeps gradients from exploding or vanishing.

## Day 8: Jan 4, 2026
**Focus:** The Calculus of Initialization - Fixing Dead Neurons and the Hockey Stick Loss (Following Karpathy's Building Makemore part 3)

**Code:** [Makemore Part 3: Activations & Gradients](https://github.com/m-mahadi/frontier-ai-365/blob/8238b273da07f1d8a374ca4f29347baca50c6cfb/01-foundations-nn/makemore_mlp_part2.ipynb)

### The Reality Check

Today was rough. Had zero motivation. The kind of day where opening the laptop feels like a chore. Brain foggy, energy low.

But I made a rule: no zero days. Even if I can only do 30 minutes, that's 30 minutes more than nothing.

So I opened the notebook.

### Today's Progress (The Short Version)

* **The "Hockey Stick" Loss Problem:**
    * Noticed the initial loss was ~27.0 instead of the expected ~3.29 (which is `-ln(1/27)` - uniform probability over 27 characters).
    * **The cause:** Initialized weights and biases too large. The model was starting out "confidently wrong" - assigning near-100% probability to random characters.
    * **The fix:** Scaled down `W2` and `b2` so the logits start near zero. This forces the model to begin with a roughly uniform distribution instead of random garbage.
* **Tanh Saturation Deep Dive:**
    * Investigated the "dead neuron" problem. When pre-activation values going into `tanh` are too large (either positive or negative), the output gets squashed to -1 or +1.
    * **The consequence:** The local derivative `(1 - tanh²)` becomes effectively zero. Gradients can't flow backward. The neuron is dead - it's not learning anything.
* **What I Actually Accomplished:** 
    * 30 minutes of focused work
    * Understood *why* initialization matters at a first-principles level
    * Diagnosed saturated neurons by looking at activation histograms

### Key Insights

**Initialization isn't just about "making training faster" - it's about making training *possible*.**

If your weights are too large:
- Neurons saturate instantly
- Gradients vanish
- The network can't "feel" which direction to move
- You're dead in the water before iteration 1

**The Softmax Confidence Problem:**

High initial loss happens because the model is confidently wrong. By forcing small initial weights, we make the model start uncertain (uniform distribution). This is the mathematically optimal starting point - the model should be *maximally confused* at initialization, then learn from there.

**Gradient Flow is Everything:**

### The Mental Game

Here's what I learned today that has nothing to do with neural networks:

**Motivation is bullshit. Discipline is what matters.**

I didn't *want* to study today. I felt like doing nothing. But I have a rule: no zero days. Even 30 minutes counts.

And you know what happened? After 15 minutes of forcing myself to focus, I actually got interested. The activation histogram visualization pulled me in. By minute 30, I was genuinely engaged.

The lesson: **Start anyway.** Motivation follows action, not the other way around.


## Day 9: Jan 5, 2026
**Focus:** Reading the Paper That Started It All - Bengio et al. (2003)
**Reading:** [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

### The Reality Check (Again)

Another low-energy day. Spent most of it dealing with business stuff. Not ideal, but that's reality when you're building multiple things at once.

Could have called it a zero day. Instead, spent an hour reading the paper that literally invented neural language models as we know them.

### Today's Progress

* **Theoretical Deep Dive:** Shifted from coding to understanding the foundational paper. This is the 2003 work that birthed modern embedding-based NLP.
* **The Curse of Dimensionality Problem:**
    * Traditional n-gram models fail because possible word sequences grow exponentially with vocabulary: V^n combinations.
    * With a 100k word vocabulary, even trigrams need to track 10^15 possible sequences.
    * You'll never see most of these sequences in training data, no matter how big your corpus.
* **Distributed Representations - The Solution:**
    * Core insight: map each word to a point in continuous space (ℝ^m).
    * Words with similar meanings cluster together geometrically.
    * **The magic:** "The cat is on the mat" and "A dog is on the rug" have similar geometric trajectories in vector space, even if you've never seen the exact second sentence before.
* **The 2003 Architecture:**
    * **Embedding Layer:** Shared matrix C that maps word indices to feature vectors
    * **Hidden Layer:** tanh non-linearity processing concatenated context
    * **Output Layer:** softmax over entire vocabulary V

### Key Insights

**Similarity = Generalization:**

N-gram model: If you haven't seen "The orange is..." in training, you have zero prediction ability. No data = no probability estimate.

Bengio's model: If you've seen "The apple is...", the model leverages the fact that "orange" and "apple" are close in embedding space to make intelligent guesses. It generalizes from similar contexts.

**This is why embeddings changed everything.**

**The Trade-off They Made:**

The paper openly acknowledges neural models are **way slower** than n-gram counting. But they provide superior generalization by learning the *joint probability function* of word sequences instead of just memorizing counts.

2003 compute couldn't handle this at scale. 2026 compute makes it trivial. That's why these ideas took 15+ years to dominate.

Everything I've learned so far (character embeddings, hidden layers, softmax output) is literally implementing this 2003 architecture. The only differences:
- We're working at character level, they worked at word level
- We're using tiny datasets, they used millions of words
- Modern tools make the implementation 100x simpler

But the math is **identical**.

### What I Learned About Learning

**Reading primary literature > watching tutorials.**

I could have watched another YouTube video explaining embeddings. Instead, I went to the source. Now I understand:
- *Why* embeddings were invented (curse of dimensionality)
- *What problem* they solved (generalization from sparse data)
- *What trade-offs* were made (compute vs statistical efficiency)

### 🏁 Status & Reflections

* **What I Learned About Productivity:** Even on busy days with other commitments, one focused hour on primary literature is valuable. Not every day needs to be a coding marathon.
* **Next Step:** Back to Part 3 of makemore tomorrow. Apply Batch Normalization to fix the saturated neurons, then visualize the activation histograms to confirm the fix worked.

Progress isn't always linear. Some days you code, some days you read, some days you handle business. All of it moves you forward.
Tomorrow InshaAllah we start cooking again.


## Day 10: Jan 6, 2026
**Focus:** Batch Normalization Deep Dive - Understanding Why Networks Break

**Watching:** makemore Part 3 - Karpathy's lecture on activation statistics and BatchNorm

### The Reality Check

No coding today. Just pure conceptual learning - following through Karpathy's lecture, pausing every few minutes to actually *think* about what's happening.

### Today's Progress

* **Watched the Full Lecture:** Made it through Part 3 - the one about saturated activations, proper initialization, and BatchNorm. Took notes, rewound sections multiple times, made sure every concept actually landed.

* **The Saturation Problem Finally Makes Sense:**
    * Tanh squashes everything to [-1, 1]. Sounds harmless until you see what happens.
    * When inputs get too large (|x| >> 1), tanh saturates → outputs stuck at ±1 → "white" areas in histograms.
    * The killer: tanh derivative = 1 - tanh²(x). When tanh outputs ±1, derivative → 0. **No gradient = dead neuron.**
    * This isn't a minor issue. This is how entire layers just *stop learning* during training.

* **Weight Initialization - Not Random At All:**
    * Naive approach: `torch.randn(fan_in, fan_out)` → gradients explode or vanish within a few layers.
    * **Kaiming Initialization** for tanh: Scale weights by `(5/3) / sqrt(fan_in)`.
    * The math behind it: You want activations to stay roughly Gaussian (mean 0, std ~1) as they pass through layers. The gain factor (5/3 for tanh) comes from analyzing how variance propagates through the non-linearity.
    * Different non-linearities need different gains. ReLU is different. This is why initialization papers exist.

* **Batch Normalization - The Game Changer:**
    * **The core idea:** Don't just initialize well - *force* activations to stay Gaussian throughout training.
    * How: At each layer, normalize the pre-activations to N(0,1) using the batch statistics, then scale/shift with learnable γ and β.
    * Order matters: Linear → BatchNorm → Tanh (normalize *before* the non-linearity).
    * Training vs Inference split:
        * Training: Use current batch statistics
        * Inference: Use running averages (exponential moving average with momentum ~0.1)

* **Why BatchNorm Changed Everything:**
    * Allows 10x higher learning rates (0.1 → 1.0)
    * Much less sensitive to weight initialization
    * Slight regularization effect (batch coupling adds noise)
    * Made very deep networks (ResNets) actually trainable

* **The Diagnostic Tools I Need to Learn:**
    * **Activation histograms:** Are neurons alive or saturated?
    * **Gradient histograms:** Is information flowing backward?
    * **Update-to-data ratio:** `|param.update| / |param.data|` should be ~1e-3. Too small = slow learning, too large = unstable.
    * These visualizations are how senior engineers *know* their network is healthy, not just *hope* it is.

### Key Insights

**Karpathy's Warning About BatchNorm:**

"I have shot myself in the foot with this layer over and over... Avoid if possible."

The problem: BatchNorm couples examples in a batch together. Your prediction on one example depends on what else is in the batch. This causes:
- Subtle bugs during inference if you forget to switch modes
- Non-deterministic behavior that's hell to debug
- Issues with small batches

Modern alternatives (LayerNorm, GroupNorm) fix this. Transformers use LayerNorm exclusively. But BatchNorm is still everywhere in CNNs.

**The Connection to Bengio 2003:**

Yesterday I read about why embeddings exist. Today I learned about why we can actually train deep networks with those embeddings. 

Bengio's paper showed embeddings enable generalization. BatchNorm (2015) showed how to keep those embeddings from destroying themselves during training. These papers are 12 years apart, but they're part of the same story.

### 🏁 Status & Reflections

* **Current Milestone:** Deep conceptual understanding of the BatchNorm layer and why it exists. Ready to implement tomorrow.

## Day 11: Jan 7, 2026
**Focus:** Backpropagation Ninja Training - Manual Gradient Calculation Through Batch Normalization

**Code:** [Makemore Part 4: Backprop Exercise 1](https://github.com/m-mahadi/frontier-ai-365/blob/e84184e6e7b3b69fbcf17d860cffd173df051864/01-foundations-nn/build_makemore_backprop_ninja%20_exercise_1.ipynb)

### Today's Progress

* **The Challenge:** Manually computed gradients for every single intermediate variable in the forward pass - no PyTorch autograd shortcuts allowed.
* **What I Actually Did:**
    * Solved Exercise 1 from Karpathy's "Becoming a Backprop Ninja".
    * Manually backpropagated through 26 different tensors, from the loss all the way back to the character embeddings.
    * Got ~80% of the gradients correct on my own. The remaining 20% required checking Karpathy's solution to understand where my chain rule application broke down.
* **Validation:**
    * Used the `cmp()` utility function to compare my manual gradients against PyTorch's autograd.
    * Got "exact: True" for the first ~10 variables (logprobs through logits).
    * Got "approximate: True" with maxdiff ~1e-9 for BatchNorm variables - good enough, just floating-point precision differences.

### Key Insights

**Batch Normalization Backprop is Hell:**

The forward pass has 7 intermediate variables just for BatchNorm. The backward pass requires carefully tracking how gradients flow through:
- The variance calculation (with Bessel's correction n-1)
- The standard deviation (with the 1e-5 epsilon for stability)
- The mean subtraction
- The normalization
- The scale (gamma) and shift (beta)

Every single one of these steps has a different local derivative. Miss one and your entire gradient is wrong.

The parts I struggled with:
1. **The logit_maxes gradient:** Forgot that when you subtract the max for numerical stability, the gradient has to flow back through the max operation. This requires `F.one_hot` indexing to correctly distribute gradients only to the maximum element in each row.
2. **The BatchNorm mean backward:** Got confused about how `dbnmeani` needs to be broadcast back to `dhprebn`. The mean is computed over the batch dimension, so its gradient needs to be distributed equally across all batch elements.
3. **The embedding gradient accumulation:** Used a double loop instead of a more efficient scatter operation. It works but it's slow.
   
### The Reality Check

**Honest reflection:** I spent 3+ hours on this. Some gradients took 10 minutes each. The BatchNorm backward pass alone probably took 45 minutes of staring at equations and debugging tensor shapes.


### 🏁 Status & Reflections

Still grinding. One brutal exercise at a time. Alhamdulillah for progress, even when it's hard-won.





## Day 12: Jan 8, 2026
**Focus:** Backpropagation Through BatchNorm - The Calculus Reality Check

**Code:** [Makemore Part 4: Exercises 2, 3, 4](https://github.com/m-mahadi/frontier-ai-365/blob/518dfa847d647be2d8996bb7c6681f5a71e65048/01-foundations-nn/build_makemore_backprop_ninja_exercise_2%2C3%2C4.ipynb)

### Today's Progress

* **Exercise 2 - Cross Entropy in One Shot:**
    * Collapsed the entire softmax → log → NLL chain into a single backward pass.
    * The gradient simplifies beautifully: `dlogits = softmax(logits); dlogits[range(n), Yb] -= 1; dlogits /= n`
    * This is exactly what `F.cross_entropy` does under the hood - subtract 1 from the correct class probability, divide by batch size. That's it.
* **Exercise 3 - BatchNorm Forward Pass Compression:**
    * Rewrote the 7-line BatchNorm forward pass into a single line: `hpreact_fast = bngain * (hprebn - hprebn.mean(0, keepdim=True)) / torch.sqrt(hprebn.var(0, keepdim=True, unbiased=True) + 1e-5) + bnbias`
    * Maxdiff: 4.7e-7. Good enough. The forward pass makes sense now.
* **Exercise 4 - Full Manual Backprop Training Loop:**
    * Implemented the entire training loop with manual gradients - no `.backward()` shortcuts.
    * Used the one-line gradients from Exercises 2 and 3 to make it cleaner.
    * **The Problem:** Exercise 3's backward pass (BatchNorm gradient) completely broke me.

### The Harsh Truth

**I couldn't derive the backward passes myself.**

The final line looks like this:
```python
dhprebn = bngain * bnvar_inv / n * (n*dhpreact - dhpreact.sum(0) - n/(n-1)*bnraw*(dhpreact*bnraw).sum(0))
```

I stared at this for 30 minutes. I understand *what* it does (propagate gradients backward through normalization). I understand *why* it's needed (chain rule through mean and variance).

But I cannot derive it from first principles. My calculus is too weak.

### What I Learned (Conceptually)

* **Cross Entropy Gradient:** The softmax + NLL derivative is elegant because the gradient literally *is* the probability distribution, adjusted by subtracting 1 from the correct class. The model is "pulled" toward the right answer proportionally to how confident it was in the wrong answer.
  
* **BatchNorm is Hell:** Forward pass is simple. Backward pass requires tracking gradients through: mean calculation, variance calculation (with Bessel's correction), standard deviation (with epsilon for stability), normalization, scale, and shift. Every single step has a different local derivative. Miss one and everything breaks.

### Watched: Makemore Part 5

* Didn't code along. Just watched to get the concepts.
* Understood the high-level ideas but some of the advanced tricks went over my head.
* Will revisit later after building a stronger foundation.

### The Reality Check

**My calculus is too weak for deep learning right now.**

I can implement architectures. I can tune hyperparameters. I can debug training loops. But when it comes to deriving gradients for complex layers like BatchNorm, I hit a wall.

This isn't optional knowledge - it's foundational. If I can't derive these gradients, I don't truly understand what's happening. I'm just copying formulas.

### The Plan Forward

**Strategic Pivot:** Instead of grinding through more makemore exercises I don't fully understand, I'm going to:

1. **Move to "Let's Build GPT from Scratch"** - Get the big picture of how Transformers work
2. **Watch the Tokenizer video** - Understand how text actually gets fed into models
3. **Circle back to fundamentals** - Rewrite from scratch by myself(NO HELP):
    * Simple neural network (MNIST-level)
    * Manual backprop (micrograd-level)
    * Bigram model
    * Basic MLP

4. **Fix the calculus gap** - This semester, I need to strengthen my math. Partial derivatives, chain rule, matrix calculus. Non-negotiable.

5. **And also do stanford's CSE229** 

**I will NOT touch Makemore Part 5+ right now.** Those exercises assume a level of mathematical maturity I don't have yet. Coming back to them after strengthening calculus will be way more productive than bashing my head against derivations I can't follow.

### 🏁 Status & Reflections

* **Current Milestone:** Completed Makemore Part 4 exercises (with help). Identified a critical gap in my mathematical foundation.
* **Honest Assessment:** I can *implement* neural networks. I can *train* them. But I can't *derive* the math for complex layers from first principles. That's a problem.
* **Confidence Level:** Medium. I'm not demoralized - I'm calibrated. I know what I know, and I know what I don't know. The gap is clear. The path forward is clear.
* **Next Step Tomorrow (InshaAllah):** Start "Let's Build GPT from Scratch". Get the big picture before diving deeper into the details.

---

**The Mindset Shift:**

Progress isn't linear. Sometimes you need to zoom out before you can zoom in. I could keep grinding through exercises I half-understand, or I could build a broader foundation first and come back stronger.

I'm choosing the second path.

Still grinding. One honest day at a time. Alhamdulillah for the clarity to recognize my own limits.


## Day 13: Jan 9, 2026
**Focus:** Research Deep Dive - Multi-Hop Reasoning & Knowledge Graph RAG Architectures

**Reading:** 
- [HotPotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering](https://arxiv.org/abs/1809.09600)
- [From Local to Global: A GraphRAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130)

### Today's Progress

* **Shifted Gears to Research Mode:** Spent the majority of the day working on our research project architecture. Had to step away from the neural network fundamentals to focus on reading and understanding cutting-edge RAG systems.
* **HotPotQA Paper Analysis:**
    * Studied the multi-hop reasoning dataset design - how they use Wikipedia hyperlinks to create questions requiring reasoning across multiple documents.
    * Understood the bridge entity concept - intermediate entities that connect two pieces of information (e.g., "Who was the singer of Radiohead?" → Thom Yorke → his birthday).
    * Analyzed their three types of multi-hop reasoning: Type I (inferring bridge entities), Type II (locating entities by multiple properties), Type III (inferring properties through bridge entities).
    * Key insight: They provide **supporting facts** as ground truth - not just the answer, but which sentences led to it. This enables explainable AI.
* **GraphRAG Paper Analysis:**
    * Deep dive into Microsoft Research's graph-based RAG approach for global sensemaking questions.
    * Traditional RAG fails on questions like "What are the main themes in the dataset?" because it's designed for fact retrieval, not corpus-wide understanding.
    * GraphRAG's pipeline: Extract entities/relationships → Build knowledge graph → Detect communities (using Leiden algorithm) → Generate hierarchical summaries → Answer queries via map-reduce over community summaries.
    * **The breakthrough:** Using graph community detection to partition knowledge into thematic clusters, then summarizing those clusters hierarchically. This allows answering global questions without retrieving specific facts.
* **Architecture Design Work:** (Details classified for now - research in progress.)

### Why This Matters for Fundamentals

**These aren't just research papers - they're showing me what deep learning is actually *for*.**

The fundamentals I'm learning aren't abstract math - they're the building blocks for systems like this.

### The Reality Check

**I didn't write a single line of code today. Did I make progress?**

Yes. Research is part of the process. Understanding *what* has already been built and *why* certain architectural choices were made is just as important as implementing from scratch.

But I also need to be honest: reading papers is easier than coding. It's cognitively demanding, but there's no compiler yelling at you for syntax errors. No debugging. No grinding through edge cases.

**The balance:** Today was research. Tomorrow goes back to implementation. I can't let research become a procrastination tool.




## Day 14: Jan 10, 2026
**Focus:** Building GPT from Scratch - Character-Level Language Modeling & Self-Attention Foundations

**Code:** [Let's Build GPT - Part 1](https://github.com/m-mahadi/frontier-ai-365/blob/820c2e6d0b25108b9cbe6658e9986423086a011e/02-gpt-architectures/let's_build_gpt(incomplete).ipynb)

**Watching:** Andrej Karpathy's "Let's build GPT: from scratch, in code, spelled out"

### Today's Progress

* **Dataset Setup - Tiny Shakespeare:**
    * Downloaded and loaded the classic 1.1M character Shakespeare corpus.
    * Built a character-level vocabulary (65 unique characters including punctuation and newlines).
    * Implemented encode/decode functions using simple dictionary mappings (`stoi`, `itos`).
* **Batching & Context Windows:**
    * Implemented the `get_batch()` function to sample random chunks of `block_size=8` tokens.
    * **Key insight:** A single batch of shape `(4, 8)` actually contains 32 training examples (4 sequences × 8 positions), because each position predicts the next character. This is the "sliding window" trick that makes character-level LMs efficient.
    * Visualized how context grows: `[18]` predicts `47`, then `[18, 47]` predicts `56`, etc.
* **Bigram Language Model (Neural Net Version):**
    * Built the simplest possible language model: a lookup table where each token directly predicts the next token's logits.
    * Architecture: Just an embedding table of shape `(vocab_size, vocab_size)`. No hidden layers. No transformations.
    * Loss started at ~4.88 (random guessing entropy for 65 classes is `-ln(1/65) ≈ 4.17`, so close to theoretical maximum).
* **Training Loop:**
    * Implemented basic SGD with Adam optimizer (`lr=1e-1`).
    * Trained for 10,000 steps with batch size 32.
    * Final loss: **2.44** (compared to starting loss of 4.88).
    * Generated samples went from complete garbage to... slightly structured garbage. Model learned basic bigram patterns like "ch", "th", "ed" but still no long-range coherence.
* **Self-Attention Mathematical Foundation:**
    * **The Core Trick:** Computing weighted averages of past tokens using matrix multiplication.
    * Implemented three versions of the same operation:
        1. **Naive loops:** Double for-loop computing `mean(x[b, :t+1])` for each position.
        2. **Matrix multiply:** Using a lower-triangular weight matrix to aggregate past tokens in one shot.
        3. **Softmax version:** Masking future positions with `-inf`, then applying softmax to get attention weights.
    * **The revelation:** All three produce identical results (`torch.allclose() = True`), but version 3 is what Transformers actually use because it generalizes to learned attention patterns.
* **Research Work:** Continued working on the research project architecture (details classified).

### Key Insights

**Self-Attention = Learnable Weighted Averaging:**
* Version 1 (loops): Compute running average manually. Conceptually clear but slow.
* Version 2 (matrix multiply): Use a triangular matrix to encode "only attend to past tokens." Fast but fixed weights.
* Version 3 (softmax): Replace fixed weights with learned weights. This is self-attention.
* **The trick:** `masked_fill(tril == 0, float('-inf'))` ensures future tokens are ignored. After softmax, their weights become zero. Elegant as hell.

**Why This Works:**
```python
wei = torch.tril(torch.ones(T, T))  # Lower triangular matrix
wei = wei / wei.sum(1, keepdim=True)  # Normalize rows to sum to 1
xbow = wei @ x  # Weighted average of past embeddings
```
This is just computing `x[t] = average of x[0:t+1]` in parallel for all `t`. No loops needed.

**The Bridge to Transformers:**
* Right now, `wei` is uniform (every past token gets equal weight).
* In a real Transformer, `wei` comes from the Query-Key similarity: `wei = softmax(Q @ K.T / sqrt(d_k))`.
* This lets the model **learn** which past tokens are relevant, instead of averaging everything equally.

### The Mathematical Breakthrough

**Weighted aggregation via matrix multiply:**
```python
a = [[1.0, 0.0, 0.0],
     [0.5, 0.5, 0.0],
     [0.33, 0.33, 0.33]]  # Weights for positions 0, 1, 2

b = [[2, 7],
     [6, 4],
     [6, 5]]  # Token embeddings

c = a @ b  # Weighted sum of embeddings
# c[0] = 1.0*b[0] = [2, 7]
# c[1] = 0.5*b[0] + 0.5*b[1] = [4, 5.5]
# c[2] = 0.33*b[0] + 0.33*b[1] + 0.33*b[2] = [4.67, 5.33]
```

This is **the** core operation in Transformers. 

### Time Allocation Today

* **Research work:** ~4 hours (architecture design, paper review)
* **GPT tutorial:** ~3 hours (coding along, rewinding, taking notes)
* **Total deep work:** ~7 hours


## Day 15: Jan 11, 2026
**Focus:** Completing Karpathy's Let's build GPT

**Watching:** Andrej Karpathy's "Let's build GPT: from scratch, in code, spelled out" (Completion)

### Today's Progress

* **Self-Attention Version 4 - The Real Deal:**
    * Implemented full self-attention with Query, Key, Value projections.
    * **The architecture:** Three linear transformations (`head_size = 16`):
        - `key = x @ W_k` - What do I contain?
        - `query = x @ W_q` - What am I looking for?
        - `value = x @ W_v` - What do I communicate if I'm relevant?
    * Computed attention weights: `wei = softmax(query @ key.T / sqrt(head_size))`
    * Applied weights to values: `out = wei @ value`
    * **Why this works:** Tokens with high query-key similarity get high attention weights, and we aggregate their values.
* **Watched Through the Complete Architecture:**
    * Multi-head attention (running multiple attention heads in parallel)
    * Feed-forward networks (MLP after attention)
    * Residual connections (`x = x + attention(x)`)
    * Layer normalization
    * Positional embeddings
    * The full GPT decoder stack
* **Understanding vs. Implementation:**
    * Didn't write much code today - mostly watched and took notes.
    * Focused on understanding *why* each component exists rather than just copying code.
    * The lecture moves fast in the second half, covering the complete architecture.

### Key Insights

**Query-Key-Value Finally Makes Sense:**

**Why separate Key and Value?**
- Key is used for *routing* (deciding attention weights)
- Value is used for *aggregation* (what gets mixed together)
- They can encode different aspects of the same token

**The Scaling Factor `sqrt(head_size)`:**
- Without it, `query @ key.T` can have very large magnitudes
- Large magnitudes → softmax saturates → gradients vanish
- Dividing by `sqrt(d_k)` keeps the dot products in a reasonable range
- This is **critical** for stable training - not just a minor detail

**Multi-Head Attention = Multiple Perspectives:**
- Instead of one 512-dim attention, run 8 heads of 64-dim each
- Each head learns to attend to different patterns (syntax, semantics, position, etc.)
- Concatenate all heads, then project back to model dimension
- This is parallelizable - all heads run simultaneously

**Residual Connections = Training Deep Networks:**
- Without residuals: gradients vanish in deep networks
- With residuals: gradients have a "highway" directly to earlier layers
- Formula: `x = x + attention(x)` instead of `x = attention(x)`
- Allows training 100+ layer networks (e.g., GPT-3 has 96 layers)

**Layer Norm vs Batch Norm:**
- Batch Norm: normalize across the batch dimension (what I learned in makemore)
- Layer Norm: normalize across the feature dimension
- Layer Norm is better for Transformers because:
    - No coupling between examples in a batch
    - Works with batch_size=1 during inference
    - More stable for variable-length sequences

**Positional Embeddings:**
- Self-attention is permutation-invariant - it doesn't know token order
- "The cat sat on the mat" and "mat the on sat cat The" look identical
- Solution: add positional encodings to input embeddings
- Two approaches: learned (what GPT uses) or sinusoidal (original Transformer paper)

### What I Learned (Conceptually)

**The Full Transformer Block:**
```
x = x + self_attention(layer_norm(x))  # Attention with residual
x = x + feed_forward(layer_norm(x))    # MLP with residual
```

This pattern repeats N times (e.g., 12 layers in GPT-2 small, 96 in GPT-3).

**Why the MLP after attention?**
- Attention is linear (just weighted averaging)
- MLP adds non-linearity and point-wise transformations
- Think of it as: attention = communication, MLP = computation

**The Training Setup:**
- Cross-entropy loss on next-token prediction
- Adam optimizer with learning rate decay
- Gradient clipping to prevent explosions
- Dropout for regularization

### What I Still Don't Understand Deeply

* **Why does multi-head attention work better than single-head with higher dimension?**
* **The exact mechanics of masked self-attention:**
* **How does the model learn long-range dependencies?**
* **Positional embeddings - learned vs sinusoidal:**
  
### The Reality Check

I can explain the components:
- Query, Key, Value projections ✓
- Multi-head attention concept ✓
- Residual connections ✓
- Layer normalization ✓

But I couldn't implement a full Transformer from scratch right now. Not even close.


## Day 16: Jan 12, 2026
**Focus:** Understanding Tokenization - From Character-Level to Byte Pair Encoding

**Watching:** Andrej Karpathy's "Let's build the GPT Tokenizer" (Part 1 - up to BPE algorithm)

### Today's Progress

* **Watched: Character-Level Tokenization:**
    * Reviewed why character-level modeling (what we did in GPT tutorial) is inefficient
    * 1 character = 1 token means very long sequences
    * "Hello" = 5 tokens at character level vs 1 token at word level
    * Long sequences → more computation, longer context needed
* **Word-Level Tokenization Problems:**
    * Naive approach: split on spaces, build vocabulary
    * **Problem 1:** Vocabulary explosion (100k+ words in English)
    * **Problem 2:** Can't handle unseen words (OOV = out of vocabulary)
    * **Problem 3:** No morphological understanding ("run", "running", "runs" are completely separate)
* **Byte Pair Encoding (BPE) - The Solution:**
    * Start with character-level vocabulary
    * Iteratively merge the most frequent pair of tokens
    * Example: "l" + "o" appears often → create "lo" token
    * "lo" + "w" appears often → create "low" token
    * Continues until vocabulary reaches target size (e.g., 50k tokens)
* **Why BPE is Genius:**
    * Balances between character-level (flexible but inefficient) and word-level (efficient but rigid)
    * Common words get single tokens ("the", "and", "is")
    * Rare words get broken into subwords ("unhappiness" → "un" + "happiness")
    * Can represent ANY text (no OOV problem) because it falls back to characters
    * Vocabulary size is tunable (trade-off between sequence length and vocab size)

### Key Insights

**The Tokenization Hierarchy:**
1. **Character-level:** 1 char = 1 token. Flexible but inefficient. ~100 token vocab.
2. **Word-level:** 1 word = 1 token. Efficient but rigid. ~100k+ token vocab.
3. **BPE (subword):** Between chars and words. Best of both worlds. ~50k token vocab.

Ran out of time due to other work commitments. Will continue tomorrow.

## Day 17: Jan 13, 2026
**Focus:** The Engine Room - Attention Mechanics & The KV Cache

**Watching:**
* [Query, Key and Value Matrix for Attention Mechanisms](https://www.youtube.com/watch?v=UPtG_38Oq8o)
* [The KV Cache: Memory Usage in Transformers](https://www.youtube.com/watch?v=80bIUggRJf4)

### Today's Progress

* **The "Zero-Motivation" Day:**
    * Woke up with absolutely no drive. Brain felt like mush.
    * Considered taking a zero day. Then remembered the rule: **no zero days.**
    * Couldn't handle coding today. So I pivoted - focused on conceptual videos instead.
    * Sometimes showing up at 30% is better than not showing up at all.

* **Attention Mechanics Deep Dive:**
    * Finally deconstructed what Query, Key, and Value **actually mean** beyond the abstract math.
    * **The Intuition:**
        - **Query:** "What am I looking for?" (e.g., the word "Bank" asking "what context should I use?")
        - **Key:** "What do I contain?" (e.g., "River" saying "I'm a geographical feature")
        - **Value:** "What information do I pass along?" (e.g., "River" providing its semantic embedding)
    * **The Disambiguation Example:**
        - Input: "The bank of the river"
        - Query from "bank" computes dot product with all Keys
        - High score with "river" (geographical context)
        - Low score with any financial terms
        - Attention weights pull the Value from "river" → "bank" now means riverbank, not financial institution
    * This is how Transformers resolve ambiguity. It's not magic - it's learned similarity matching.

* **The KV Cache - Why Inference is Expensive:**
    * Watched Umar Jamil's explanation of why long-context generation kills VRAM.
    * **The Problem Without Caching:**
        - Generating token 1: compute attention over 0 tokens (instant)
        - Generating token 2: compute attention over 1 token
        - Generating token 100: compute attention over 99 tokens
        - Generating token 1000: compute attention over 999 tokens
        - This is **quadratic waste** - you're recomputing the same Keys and Values over and over
    * **The Solution - KV Cache:**
        - After computing Key and Value for token `t`, store them in GPU memory
        - When generating token `t+1`, reuse the cached K and V matrices
        - Only compute new K and V for the single new token
        - Concatenate with cached tensors: `K_cached = [K_old, K_new]`
    * **The Trade-off:**
        - **Without cache:** Low memory, but quadratic compute (slow)
        - **With cache:** Linear compute (fast), but massive memory usage
        - For a 7B model with 4096 context length, the KV cache can be **larger than the model weights themselves**
    * **Why OpenAI Charges Per Token:**
        - Every token you generate requires storing K and V tensors in VRAM
        - Long conversations = huge memory footprint
        - This is why ChatGPT "forgets" - they have to truncate context to save memory
        - This is why long-context models (GPT-4 Turbo 128k) cost more - the KV cache scales with context length
          
### The Reality Check

**Coding today:** Literally zero.

**Why?**
- Mental fatigue. Motivation was completely dead.
- Had two options: (1) Take a zero day, or (2) Do *something* productive
- Chose option 2 - watched conceptual videos instead of coding

**Is this a problem?**

Maybe. This is now Day 4 of minimal/no coding:
- Day 14: Some coding (GPT Part 1)
- Day 15: Minimal coding (GPT Part 2)
- Day 16: Zero coding (Tokenizer lecture)
- Day 17: Zero coding (Attention theory)

**The pattern is clear:** I'm in passive learning mode. Watching videos is my procrastination.

## Day 18: Jan 14, 2026
**Focus:** Byte Pair Encoding Implementation - Building a Tokenizer from Scratch

**Code:** [BPE Tokenizer Implementation](https://github.com/m-mahadi/frontier-ai-365/blob/79752055930ff1b843138f859a1701833f3e415d/01-foundations-nn/tokenizer%20unfinished.ipynb)

**Watching:** Andrej Karpathy's "Let's build the GPT Tokenizer" (Continuation - BPE Implementation)

### Today's Progress

* **BPE Algorithm Implementation:**
    * Built the core `get_stats()` function - counts frequency of all adjacent byte pairs
    * Implemented `merge()` function - replaces most frequent pair with a new token ID
    * **The Training Loop:**
        - Start with UTF-8 bytes (vocab size = 256)
        - Find most frequent byte pair in corpus
        - Merge that pair into a new token (ID = 256, 257, 258...)
        - Repeat until reaching target vocab size (276 in this example)
    * Tracked all merges in a dictionary for later use during encoding

* **Encoding & Decoding:**
    * **Decoding:** Built vocab dictionary mapping token IDs to their byte sequences, then concatenated and decoded to UTF-8
    * **Encoding:** Given new text, iteratively apply merge rules learned during training
    * **The Challenge:** Need to apply merges in the **correct order** (the order they were learned)
    * **The Solution:** `min(stats, key=lambda p: merges.get(p, float("inf")))` - finds the pair that was merged earliest

* **Compression Results:**
    * Original text: 24,597 bytes
    * After BPE (vocab size 276): 19,438 tokens
    * **Compression ratio: 1.27x**
    * With larger vocab (e.g., 50k like GPT-2), compression would be ~4x

* **Verification:**
    * `decode(encode(text)) == text` ✓ - Round-trip works perfectly
    * The tokenizer can encode *any* UTF-8 text (no OOV problem) because it falls back to individual bytes

### Key Insights

**BPE is Greedy Compression:**
- At each step, merge the most frequent pair
- This is a greedy algorithm - not guaranteed to be optimal
- But it's fast and works well in practice
- Similar to Huffman coding but for pairs instead of individual symbols

**The Merge Order Problem:**
When encoding new text, you can't just merge any pair you see. You need to apply merges in the **same order** they were learned during training.

Example:
```
Training learned: (101, 32) → 256, then (256, 116) → 257
Encoding "e t": First merge (101, 32) → [256, 116]
                Then merge (256, 116) → [257]
```

If you applied merges out of order, the encoding would be inconsistent with training.

**Why UTF-8 Bytes as Base:**
- GPT tokenizers start with UTF-8 byte-level encoding (256 base tokens)
- This means they can represent **any** Unicode text without OOV errors
- Alternative: start with characters (65k+ vocab for Unicode) - wastes vocab space
- Byte-level is more efficient and handles rare scripts automatically

**The Vocabulary Dictionary:**
```python
vocab = {idx: bytes([idx]) for idx in range(256)}  # Base: single bytes
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]  # Build up merged tokens
```
Each token ID maps to its byte sequence. Token 256 might be `b"e "`, token 257 might be `b"in"`, etc.

**The Encoding Loop:**
```python
while len(tokens) >= 2:
    stats = get_stats(tokens)
    pair = min(stats, key=lambda p: merges.get(p, float("inf")))
    if pair not in merges:
        break  # No more valid merges
    idx = merges[pair]
    tokens = merge(tokens, pair, idx)
```
Keep merging until no more learned pairs exist in the sequence.

### What I Learned (Implementation Details)

**The `zip(ids, ids[1:])` Trick:**
```python
for pair in zip(ids, ids[1:]):
    counts[pair] = counts.get(pair, 0) + 1
```
This iterates over all adjacent pairs in one pass. Elegant.

**The Merge Logic:**
```python
while i < len(ids):
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
        newids.append(idx)
        i += 2  # Skip both elements of the pair
    else:
        newids.append(ids[i])
        i += 1
```
Can't use a for-loop because we need to skip 2 positions when we merge. While-loop with manual index management.

### The Reality Check

**Coding today:** About 2 hours of active coding (following along with Karpathy).

**Did I code from scratch?** No. I followed the lecture step-by-step.

**Could I implement this without the video?** Probably not. I understand each function now, but I wouldn't have known the overall structure without guidance.

## Day 19: Jan 15, 2026
**Focus:** The Regex Hell - Research Work Day

**Code:** Research project implementation (details classified)

### Today's Progress

* **Zero deep learning work today.** Full day dedicated to research project.
* **The Task:** Building a parser for structured documents with complex hierarchical patterns.
* **The Reality:** Spent 8+ hours writing, debugging, and fixing regex patterns.
* **What I Built:**
    - Pattern matching system for hierarchical document structure
    - Text extraction and layout detection logic
    - Inline parsing for nested elements
    - Output serialization to structured format

### Key Insights (The Painful Kind)

**Regex is a Beautiful Nightmare:**

There's a special kind of pain that comes from staring at this:
```python
re.compile(r'^(\d+[A-Z]?)\.\s+(.+)')
```

And realizing it matches "Section 1. Text" but also matches "Explanation 1. Text" when you explicitly DON'T want that.

Then you modify it to:
```python
re.compile(r'^(?!Explanation)(\d+[A-Z]?)\.\s+(.+)')
```

And it still breaks because now it doesn't handle "499A. Some Text" properly.

**The Regex Debugging Loop:**

1. Write pattern
2. Test on sample data
3. Works!
4. Test on real data
5. Breaks spectacularly
6. Add edge case handling
7. Now the original case breaks
8. Curse programming
9. Rewrite pattern from scratch
10. Repeat 50 times

**Edge Cases Are Infinite:**

The document structure I'm parsing has:
- Numbered sections: "1. Title"
- Lettered sections: "499A. Title"
- Subsections: "(1) Text"
- Clauses: "(a) Text"
- Illustrations that LOOK like clauses but aren't
- Multi-line marginal notes
- Page breaks in the middle of sections
- Headers and footers disguised as content
- Unicode characters that break `.split()`
- Nested hierarchies that require recursive parsing

Each one required 30+ minutes of debugging.

**The False Sense of Progress:**

Hour 1: "I'll have this done in 2 hours"
Hour 3: "Why is this so hard?"
Hour 5: "I've made a terrible mistake"
Hour 8: "It works! ...I think?"
Hour 8.5: "It doesn't work."

**Text Extraction is a Lie:**

The PDF has two columns: marginal notes (left) and main text (right).

Sounds simple, right? Just split at X-coordinate 135.

Except:
- Sometimes the layout shifts
- Sometimes text bleeds between columns
- Sometimes page numbers appear in random locations
- Sometimes words are split across lines mid-character
- Unicode rendering is inconsistent
- "Simple" whitespace is actually 17 different Unicode space characters

**The Worst Part:**

After 8 hours, I have a working parser. It handles 95% of cases correctly.

But I KNOW there are edge cases I haven't encountered yet. The code will break on some random document, and I'll have to debug regex hell all over again.

**This is "Research Work":**

Academia romanticizes research as "pursuing knowledge" and "solving hard problems."

The reality: 80% of the time is wrestling with data formats, broken libraries, and regex patterns.

The actual research (the thinking, the novel contributions)? Maybe 20% of the time.

The rest is just... this.

### Time Allocation Today

* **Regex hell:** ~5 hours
* **Testing + debugging:** ~2.5 hours
* **Documentation:** ~30 minutes
* **Deep learning:** 0 hours
* **Crying internally:** Immeasurable

Total deep work: ~8 hours (all research, zero fundamentals)

### The Reality Check

**Days since last fundamentals work:** 2 days (Day 17 was theory, Day 18 was coding along, today was research)

**I'm slipping.**

Thinking of starting Stanford's CS299 from tomorrow direct. spend few hours then all research. 


## Day 20: Jan 16, 2026
**Focus:** Matrix Calculus Foundations - The Math Behind Backpropagation

**Study:** Multivariable Calculus & Matrix Calculus problem set + Andrew Ng's ML Course Intro

### Today's Progress

* **Zero coding today.** Full day on mathematical foundations.
* **The Mission:** Understand gradient and Hessian calculations for neural network optimization.
* **What I Worked Through:**
    - Gradient vectors (∇f(x)) - the direction of steepest ascent
    - Hessian matrices (∇²f(x)) - the curvature of the loss landscape
    - Quadratic forms (x^T Ax) - how we approximate complex loss functions
    - Chain rule for multivariable functions - the foundation of backpropagation
    - Outer products (aa^T) - why Hessians are matrices, not scalars

* **Watched:** Andrew Ng's ML Course Introduction
    - Supervised learning (regression vs classification)
    - Unsupervised learning (clustering, dimensionality reduction)
    - The formal definition of machine learning (Task, Experience, Performance)

### Key Insights

**The Gradient is a Vector Field:**

In single-variable calculus, the derivative is just a slope (one number).

In multivariable calculus, the gradient is a **vector** that points in the direction of steepest ascent.
```
∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ
```

**Why this matters for deep learning:** Backpropagation is literally computing this gradient, then stepping in the opposite direction (steepest *descent*) to minimize loss.

**The Hessian is a Curvature Map:**

The Hessian is a matrix of second-order partial derivatives. It tells you the **shape** of the loss landscape:
- Positive definite: You're in a valley (local minimum)
- Negative definite: You're on a peak (local maximum)  
- Indefinite: You're at a saddle point (up in one direction, down in another)

**The Derivative Rules I Had to Learn:**

1. **Linear term:** ∇(c^T x) = c
   - If you have a dot product with a constant vector, the gradient is just that vector

2. **Quadratic term:** ∇(x^T Ax) = 2Ax (if A is symmetric)
   - This is like the scalar rule (x²)' = 2x, but in matrix form
   - The "2" appears because x appears twice in the product

**The Outer Product (aa^T) Mystery - SOLVED:**

This was the hardest part. Why does the Hessian need aa^T instead of a^T a?

**a^T a:** (1×n) × (n×1) = scalar (1×1)
- You get one number
- You've collapsed all the interaction information

**aa^T:** (n×1) × (1×n) = matrix (n×n)
- You get a grid of interactions
- Cell (i,j) tells you how variable i interacts with variable j

Example with a = [1, 2]:
```
a·a^T = [1]·[1 2] = [1×1  1×2] = [1  2]
        [2]         [2×1  2×2]   [2  4]
```

**Why this matters:** The Hessian MUST be a matrix to encode all pairwise interactions between variables. A scalar destroys this information.

### What I Learned from Andrew Ng's Lecture

**The Formal Definition of ML (Tom Mitchell):**

A well-posed learning problem has three components:
- **Task (T):** What are you trying to do? (e.g., classify emails)
- **Experience (E):** What data do you learn from? (e.g., labeled spam/not-spam)
- **Performance (P):** How do you measure success? (e.g., accuracy)

**Supervised Learning:**
- **Regression:** Predict continuous values (e.g., house prices)
- **Classification:** Predict discrete categories (e.g., malignant/benign tumor)
- Reality: Most real problems are high-dimensional (not 2D plots in textbooks)

**Unsupervised Learning:**
- **Clustering:** Find hidden structure in unlabeled data (e.g., Google News grouping articles)
- **Dimensionality reduction:** Compress high-dimensional data
- **Cocktail party problem:** Separate mixed audio sources (ICA - Independent Component Analysis)

**Reinforcement Learning:**
- Learn by trial and error with rewards/penalties
- Example: Autonomous helicopter learning to fly by maximizing a stability reward function

**The Junior vs Senior Engineer Gap:**

Junior: Randomly tweaks hyperparameters hoping for improvements
Senior: Diagnoses exactly why the model is failing (bias vs variance) and applies surgical fixes

This is what I need to become - someone who can debug models systematically, not just throw more data/compute at problems.

## Day 21: Jan 17, 2026
**Focus:** Positive Definite Matrices & GraphRAG Implementation

**Study:** Linear Algebra - Positive Definite Matrices
**Code:** Research project implementation (details classified)

### Today's Progress

* **Math Work:** Positive definite matrices problem set
* **Research Work:** System implementation (not the focus of this log)
* **Zero deep learning fundamentals coding**

### The Math: Positive Definite Matrices

**The Definition:**

A matrix A ∈ ℝⁿˣⁿ is **positive semidefinite** (PSD), denoted A ≽ 0, if:
- A = Aᵀ (symmetric)
- xᵀAx ≥ 0 for all x ∈ ℝⁿ

A matrix is **positive definite** (PD), denoted A ≻ 0, if:
- A = Aᵀ (symmetric)  
- xᵀAx > 0 for all x ≠ 0 (strictly greater than zero for non-zero vectors)

**The Simplest Example:**

The identity matrix I (diagonal matrix with 1s on diagonal, 0s elsewhere):

xᵀIx = xᵀx = Σᵢ xᵢ² = ||x||²

This is always ≥ 0, and equals 0 only when x = 0.

### The Math: Key Tricks I Learned

**Problem (a): Proving A = zzᵀ is positive semidefinite**

**The Trick:** Use associativity to rearrange the product.

xᵀ(zzᵀ)x = (xᵀz)(zᵀx) = α · α = α²

Since any number squared is ≥ 0, the matrix is PSD. Simple.

**Problem (b): Finding null-space and rank of A = zzᵀ**

**The Trick:** Compute Ax directly, then factor.

Ax = (zzᵀ)x = z(zᵀx)

For this to equal zero, need zᵀx = 0.

**Result:** 
- Null-space = all vectors orthogonal to z (dimension n-1)
- Rank = 1 (every row/column is a multiple of every other)

**Verification:** rank + nullity = 1 + (n-1) = n ✓

**Problem (c): Rank inequalities with PSD matrices**

**Got stuck here.** Tried multiple approaches:
- Column space argument
- Row space argument  
- Couldn't figure out where the PSD property matters

Ran out of cognitive energy after 30 minutes. Need fresh brain to revisit.

### Key Process Insights

**The Outer Product Trick:**

When you see zzᵀ, remember:
- It's always rank 1 (only one independent direction)
- It's always PSD (because xᵀ(zzᵀ)x is a squared dot product)
- Its null-space is all vectors perpendicular to z

**The Associativity Trick:**

When proving xᵀAx ≥ 0, try rearranging using parentheses:
- xᵀ(zzᵀ)x = (xᵀz)(zᵀx) reveals the structure
- Dot products commute: xᵀz = zᵀx
- Result: something squared, which is always ≥ 0

For now: accept slow math progress. Research has priority.

## Day 22: Jan 18, 2026
**Focus:** Eigenvalues, Eigenvectors, and the Spectral Theorem

**Study:** Linear Algebra - Spectral decomposition and diagonalization

### Today's Progress

* **Math Work:** Eigenvalue problem set (problems on diagonalization and spectral theorem)
* **Research Work:** Minimal - mostly math focus today
* **Zero deep learning coding**

### The Math: What I Learned

**Eigenvectors and Eigenvalues:**

An eigenvector v of matrix A satisfies: Av = λv

The scalar λ is the eigenvalue. The vector just gets scaled, not rotated.

**Diagonalization:**

If A is diagonalizable: A = TΛT⁻¹

Where:
- T has eigenvectors as columns: T = [v⁽¹⁾ ... v⁽ⁿ⁾]
- Λ is diagonal with eigenvalues: Λ = diag(λ₁,...,λₙ)

**The Spectral Theorem:**

For symmetric matrices (A = Aᵀ):
- Always diagonalizable by orthogonal matrix U
- A = UΛUᵀ where UᵀU = I
- All eigenvalues are real
- Eigenvectors are orthogonal

**Problem (b): Orthogonal eigenvectors**

Showed that if U = [u⁽¹⁾ ... u⁽ⁿ⁾] is orthogonal and A = UΛUᵀ, then:
- u⁽ⁱ⁾ is an eigenvector of A
- Au⁽ⁱ⁾ = λᵢu⁽ⁱ⁾

**Problem (c): PSD eigenvalues**

If A is positive semidefinite, then all eigenvalues λᵢ ≥ 0.

Got stuck on the formal proof. Need to revisit.

### Why This Matters

**For Neural Networks:**

- Eigenvalues of Hessian tell you about loss landscape curvature
- PCA uses eigenvectors of covariance matrix
- Spectral normalization in GANs uses largest eigenvalue
- Condition number (max λ / min λ) affects optimization stability

### Time Allocation

* **Eigenvalue problems:** ~2 hours
* **Research work:** ~1 hour
* **Total:** ~3 hours

Short day. Exhausted from yesterday.

### 🏁 Status

**Math:** Partial progress on spectral theorem problems.

**Research:** On hold today.

**Mental State:** Tired. Needed a lighter day.

Still grinding. One day at a time.
