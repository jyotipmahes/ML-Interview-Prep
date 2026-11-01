# ML Interview Prep

Repository for Machine Learning Engineer (MLE) interview preparation materials and from-scratch algorithm implementations. Additional notebooks and resources will be added here.

---

## 📌 Current Status

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

---

## 🚀 Upcoming Additions
More topic-specific notebooks, ML system design walkthroughs, and practice problems will be added as preparation continues.
