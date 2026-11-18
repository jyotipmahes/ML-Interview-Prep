# ML Interview Prep

Repository for Machine Learning Engineer (MLE) interview preparation materials and from-scratch algorithm implementations. Additional notebooks and resources will be added here.

---

## 📌 Current Status

### 🏛️ Machine Learning System Design (MLSD) Master Tracks

- **[`System_Design/README.md`](./System_Design/README.md)**: Staff-level architectural blueprints, mathematical foundations, and production failure mode analysis for Tier-1 ML System Design rounds (Meta, Google, LinkedIn, Netflix, Pinterest, ByteDance).
  - **Track 01: Recommendation Systems (6-Part Master Architecture)**:
    - **[`Part 1: Candidate Retrieval & Vector Space Intelligence`](./System_Design/01_Recommendation_Systems/Part_1_Retrieval_and_Candidate_Generation.md)**: 4-stage funnel latency SLAs ($15\text{ms}$), classical CF (Funk SVD, SVD++, Implicit ALS Spark sparsity trick $Q^T C_u Q$, BPR pairwise loss), Deep Two-Tower contrastive encoders (InfoNCE, LogQ popularity correction derivation, 80/20 hard negative curriculum, temperature $\tau$), ANN vector indexing (IVF-PQ residual math, HNSW graphs, Google SCaNN MIPS loss, 6-way Vector DB matrix), the "Metadata Filtering Dilemma" (LinkedIn GPU Exhaustive + RAR in $<1.5\text{ms}$), Pinterest PinnerSage/PinSage & LinkedIn PYMK, and Generative Retrieval with Semantic IDs via RQ-VAE (TIGER) & YouTube PLUM (2025).
    - **[`Part 2: Pre-Ranking (L1) & Heavy Ranking (L2) Scoring Engine`](./System_Design/01_Recommendation_Systems/Part_2_Ranking_PreRanking_and_Feature_Engineering.md)**: Pre-Ranking L1 architectural dilemmas & the "Feature Availability Trap" ($30-60\text{ms}$ network fetch), Pinterest Root-Leaf serving topology, high-cardinality terabyte embedding tables, Facebook 2014 GBDT+LR, Google Wide & Deep, Deep & Cross Network (DCN-v1 rank-1 bottleneck vs. DCN-v2 full-rank matrix math, Stacked vs. Parallel architectures, Low-Rank MoE), Meta DLRM distributed scaling (Model Parallelism on GPU VRAM with All-to-All vs. Data Parallelism with All-Reduce), Multi-Task Learning (Shared-Bottom $\to$ MMoE $\to$ PLE), decoupling training loss weights from real-time serving Value Functions ($V = \sum w_k \hat{p}_k$), Group AUC (GAUC), and Expected Calibration Error (ECE).
    - *Upcoming:* Part 3 (Sequential Transformers & Loss Engineering), Part 4 (Re-Ranking, Diversity & Slate Optimization), Part 5 (Generative RecSys HSTU & Real-Time Ops), Part 6 (Industrial Architectures & 45-Min Staff Interview Playbook).

### Available Notebooks

- **[`MLE_Coding_Interview_Guide.ipynb`](./MLE_Coding_Interview_Guide.ipynb)**: From-scratch implementations of core MLE coding interview algorithms with mathematical formulations, interview tips, and verification tests:
  1. **Linear Regression** (Vectorized Gradient Descent & Normal Equation)
  2. **Logistic Regression** (Vectorized Gradient Descent & Numerically Stable Sigmoid)
  3. **K-Means Clustering** (Random Initialization, Vectorized Broadcasting & List Comprehension)
  4. **K-Nearest Neighbors (KNN)** (Unified Pairwise Distance Matrix & $\mathcal{O}(N)$ Top-K)
  5. **2-Layer MLP with Backprop** (Manual Multivariate Chain Rule in pure NumPy)
  6. **Layer Normalization** (NumPy & PyTorch implementations)
  7. **Multi-Head Self-Attention** (Scaled Dot-Product, Causal Masking & Projections in PyTorch)
  8. **Low-Rank Adaptation (LoRA)** (PEFT module with initialization trap handling & weight merging)
  9. **Byte-Pair Encoding (BPE)** (Subword tokenizer with pair-frequency statistics and greedy merging)

- **[`ML_LeetCode_Interview_Guide.ipynb`](./ML_LeetCode_Interview_Guide.ipynb)**: LeetCode-style algorithmic questions intersecting with Machine Learning, Probability, Retrieval, and Computer Vision in **pure Python** (no NumPy/PyTorch, standard library only):
  1. **Weighted Random Sampling** (Prefix Sums + Binary Search / LC 528)
  2. **Reservoir Sampling** (Uniform Streaming with $\mathcal{O}(1)$ Memory / LC 382 & LC 398)
  3. **Sort Transformed Array** (Convex Quadratic Parabola & Two Pointers / LC 360)
  4. **Find Minimum of a Convex Function** (Ternary Search & Divide-and-Conquer)
  5. **Expected Calibration Error (ECE)** (Bucket Binning & Reliability Gaps)
  6. **Exact ROC-AUC with Tie Handling** (Wilcoxon Rank-Sum Statistic & Fractional Ties)
  7. **Precision-Recall Curve & Average Precision** (Sweep-Line Threshold Integration)
  8. **Dot Product of Two Sparse Vectors** (Two Pointers & Binary Search for Imbalance / LC 1570)
  9. **Top-K Cosine Similarity Retrieval** (Min-Heap with `heapq` in $\mathcal{O}(N \log K)$)
  10. **TF-IDF & Inverted Index Document Retrieval** (Lexical Search & Smoothed IDF)
  11. **Streaming Running Median** (Dual Heaps in $\mathcal{O}(\log N)$ / LC 295)
  12. **Edit Distance & Word Error Rate (WER)** (1D Space-Optimized DP / LC 72)
  13. **Image Rotation by Arbitrary Angle** (Inverse Coordinate Mapping & Bilinear Interpolation)

- **[`AI_Assisted_Coding_Interview_Guide.ipynb`](./AI_Assisted_Coding_Interview_Guide.ipynb)**: Practical implementation and AI-assisted coding patterns (asked in companies like **Shopify**, **LinkedIn**, **Meta**, and similar tech organizations, with many other companies adopting comparable pair-programming or progressive implementation formats) featuring stateful simulation, robot navigation, game state machines, and progressive class design in **pure Python** (standard library only):
  1. **Mars Rover / Grid Robot Command Engine** (Cardinal directions, ray-step obstacle collision avoidance, jump distance, and undo action history)
  2. **Blind-Grid Robot Room Cleaner** (Sensory API exploration, virtual coordinate tracking, and blind DFS backtracking with physical orientation restoration)
  3. **Cleaning Robot with Cycle Detection** (Deterministic matrix navigation, obstacle/boundary reflection, and state-space cycle detection)
  4. **Snake Game Engine with Growth & Collision** (`deque` body representation, sequential food ingestion, and tail-vacation vs. head collision order)
  5. **Connect Four / Tic-Tac-Toe with $\mathcal{O}(1)$ Win Checking** (Dynamic counter vectors for rows, columns, and diagonals with $\mathcal{O}(1)$ win evaluation)
  6. **Battleship Fleet Coordinator** (Grid coordinate validation, multi-deck ship geometry, shot hit/miss tracking, and fleet sinking coordinator)
  7. **In-Memory Key-Field Database with TTL & Backup/Restore** (Progressive 4-level architecture: CRUD, prefix scan, temporal TTL expiration, and remaining TTL restore)
  8. **Cloud Storage Manager with User Quotas & Auto-Eviction** (File indexing, user quota enforcement, largest-first tie-broken auto-eviction, and file compression)
  9. **Text Editor with Undo/Redo & Find/Replace** (Cursor navigation, buffer slicing, regex pattern replace count, and transactional command dual-stack undo/redo)
  10. **Multi-Account Financial Ledger with Atomic Rollbacks** (Double-entry transaction ledger, point-in-time binary search balance queries, and atomic transfer rollback)
  11. **Sliding Window Log & Token Bucket Rate Limiter** (Per-client timestamp window log pruning and continuous fractional refill token bucket)
  12. **DAG Task Execution Engine with Dependency Resolution** (Directed acyclic graph, Kahn's topological sort, circular cycle detection, and task output piping)

### Available Guides & Playbooks

- **[`LeetCode_Patterns_and_Constraint_Shifts.md`](./LeetCode_Patterns_and_Constraint_Shifts.md)**: A comprehensive reference guide and quick-review playbook detailing algorithmic problem patterns, constraint shifts, and complexity trade-offs for technical coding interviews:
  - **Constraint-to-Budget Engine**: Determining target Big-O from input size `N` (`10^8` operations/sec rule, `N <= 20` exponential vs. `N >= 100,000` linear/logarithmic).
  - **The Master Invariant Principle**: How breaking or granting core assumptions (sortedness, monotonicity, non-negativity, history-freeness, bounded values) transforms optimal data structures and complexity classes.
  - **Master Constraint-Shift Reference Matrix**: Direct transition mappings across searching, two-sum, subarray sums, sequences, dynamic queries, knapsack, shortest paths, and streaming.
  - **Pattern Deep Dives**: Subarrays vs. substrings vs. subsequences vs. subsets, binary search on the answer space, monotonic stacks/deques, DFS vs. BFS selection and space/memory trade-offs (balanced vs. skewed tree geometry, unweighted shortest paths vs. exhaustive backtracking), the graph shortest-path ladder, node reuse vs. simple paths, DP loop inversions and memoization soundness, interval sweep-line, memory-bounded streaming sketches, and non-comparison sorting.
  - **P, NP, NP-Complete & The 5 Escape Hatches**: Practical strategies when facing NP-hard problems (small-`N` bitmask DP, pseudo-polynomial DP, approximations, heuristics, structural restrictions).
  - **Hidden Complexity Traps & 60-Second Triage Funnel**: Common pitfalls inside loops and a repeatable framework to analyze and articulate solutions under interview pressure.

---

## 🚀 Upcoming Additions
More topic-specific notebooks, ML system design walkthroughs, and practice problems will be added as preparation continues.
