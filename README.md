## I. Machine Learning Fundamentals & Workflow
* Total Error Formulation: Error = Bias(θ)² + Var(θ) + σ²
* Bias: Bias(θ̂) = E[θ̂] - θ
* Variance: Var(θ̂) = E[(θ̂ - E[θ̂])²]
* Perceptron Learning Rule: w_t+1 = w_t + η y_t x_t

## II. Evaluation & Performance Metrics
* Ranking Accuracy: 1 - err / (P * N)
* Precision: Precision = TP / (TP + FP)
* Recall / TPR: Recall = TP / (TP + FN)
* FPR: FPR = FP / (FP + TN)
* F1-Score: F1 = 2 * (Precision * Recall) / (Precision + Recall)

## III. Parametric Models (Classification & Regression)
* Linear Regression (OLS): J(w) = (1 / 2n) * Σ [from i=1 to n] (wᵀ x_i - y_i)²
* Normal Equation: w = (Xᵀ X)⁻¹ Xᵀ y
* Ridge Regression (L2): J(w) = (1 / 2n) * Σ [from i=1 to n] (wᵀ x_i - y_i)² + λ ||w||₂²
* Lasso Regression (L1): J(w) = (1 / 2n) * Σ [from i=1 to n] (wᵀ x_i - y_i)² + λ ||w||₁
* Logistic Regression Loss: J(w) = -(1 / n) * Σ [from i=1 to n] [y_i log(σ(wᵀ x_i)) + (1 - y_i) log(1 - σ(wᵀ x_i))]
* Sigmoid Activation: σ(z) = 1 / (1 + e⁻ᶻ)

## IV. Support Vector Machines & Kernels
* Primal SVM Objective (Soft Margin): min_{w, b, ξ} (1 / 2) ||w||₂² + C * Σ [from i=1 to n] ξ_i  where  y_i(wᵀ x_i + b) ≥ 1 - ξ_i  and  ξ_i ≥ 0
* Dual SVM Formulation: max_α Σ [from i=1 to n] α_i - (1 / 2) * Σ [from i=1 to n] Σ [from j=1 to n] α_i α_j y_i y_j (x_iᵀ x_j)  where  0 ≤ α_i ≤ C  and  Σ [from i=1 to n] α_i y_i = 0
* Kernel Dual SVM Optimization: max_α Σ [from i=1 to n] α_i - (1 / 2) * Σ [from i=1 to n] Σ [from j=1 to n] α_i α_j y_i y_j κ(x_i, x_j)  where  0 ≤ α_i ≤ C  and  Σ [from i=1 to n] α_i y_i = 0
* Kernel Prediction Equation: f(x) = sign( Σ [from i ∈ SV] α_i y_i κ(x_i, x) + b )
* Common Kernels:
  * Polynomial: κ(x, x') = (xᵀ x' + c)ᵈ
  * RBF / Gaussian: κ(x, x') = exp( -||x - x'||² / 2σ² ) = exp( -γ ||x - x'||² )
  * Linear: κ(x, x') = xᵀ x'

## V. Distance Metrics & Clustering
* Metric Axioms for D(x,y):
  1. D(x,x) = 0
  2. If x ≠ y, D(x,y) > 0
  3. D(x,y) = D(y,x)
  4. D(x,z) ≤ D(x,y) + D(y,z)
* Scatter Matrix Decomposition: S = Σ [from j=1 to K] S_j + B
  * Within-Cluster Scatter: S_j = Σ [from x ∈ D_j] (x - μ_j)(x - μ_j)ᵀ
  * Between-Cluster Scatter: B = Σ [from j=1 to K] |D_j|(μ_j - μ)(μ_j - μ)ᵀ
* Total Scatter Trace: Scat(D) = Σ [from j=1 to K] Scat(D_j) + Σ [from j=1 to K] |D_j| ||μ_j - μ||₂²
* K-Medoids Optimization Target: μ_j = argmin_{x ∈ D_j} Σ [from x' ∈ D_j] Dis(x, x')
* Kernel K-Means Distance Function: Dis_κ(x, y) = √[ κ(x,x) - 2κ(x,y) + κ(y,y) ]

## VI. Deep Learning Fundamentals
* Layer Operations:
  * Pre-activation: a_i(x) = b_i + W_i h_i-1(x)
  * Activation: h_i(x) = g(a_i(x))
* Activation Functions:
  * ReLU: g(z) = max(0, z)
  * Tanh: g(z) = tanh(z) = (eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ)
  * Softmax: ŷ⁽ᵏ⁾ = e^(a⁽ᵏ⁾) / Σ [from j=1 to K] e^(a⁽ʲ⁾)
* Multi-Class Cross-Entropy Loss: L(ŷ, y) = -Σ [from k=1 to K] y⁽ᵏ⁾ log ŷ⁽ᵏ⁾
* Gradient Update Implementations:
  * GD: w_t+1 = w_t - η * (1 / n) * Σ [from i=1 to n] ∇_w L(f(x_i), y_i)
  * SGD: w_t+1 = w_t - η * ∇_w L(f(x_t), y_t)
  * Mini-Batch GD: w_t+1 = w_t - η * (1 / |B|) * Σ [from i ∈ B] ∇_w L(f(x_i), y_i)
* Dropout Execution:
  * Training: h = h ⊙ m  where  m ~ Bernoulli(p)
  * Inference: h_test = p * h

## VII. Transformers & Language Models
* Scaled Dot-Product Attention: Attention(Q, K, V) = softmax( QKᵀ / √(d_k) )V
* Multi-Head Attention (MHA): MultiHead(Q, K, V) = Concat(head_1, ..., head_h)Wᴼ
  * where head_i = Attention(QW_iᵠ, KW_iᴷ, VW_iⱽ)
* Position-Wise Feed-Forward Network (FFN): FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
* Architectural Trajectories:
  * BERT (Encoder): Bidirectional Context | Masked Language Modeling (MLM) + Next Sentence Prediction (NSP) | Special Tokens: [CLS], [SEP]
  * GPT (Decoder): Causal/Masked Attention | Autoregressive Next Token Prediction
* Prompting Typologies:
  * Chain-of-Thought (CoT): Step-by-step sequential breakdown execution.
  * Tree-of-Thoughts (ToT): Tree-structured node generation combined with search algorithms (BFS/DFS).
  * RAG: Vector space or string edit distance matching against external databases for sequence augmentation.

## VIII. Ensemble Methods
* Bagging (Bootstrap Aggregating): f_bag(x) = (1 / T) * Σ [from t=1 to T] f_t(x)  or  MajorityVote(f_1(x), ..., f_t(x))
* Random Forest Definition: Bagging + Subspace Feature Sampling
* Boosting Pipeline Sequence: α_t assignment via weak learner error rates → w_t+1 amplification of incorrect vectors → F(x) = sign( Σ [from t=1 to T] α_t f_t(x) )

## IX. Probability Distributions & Bayes
* Bernoulli: E[X] = p, Var(X) = p(1-p)
* Gaussian: E[X] = μ, Var(X) = σ²
* Bayes' Rule: p(θ | D) = p(D | θ) p(θ) / p(D)
