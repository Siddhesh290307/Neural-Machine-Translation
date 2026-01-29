# Neural Machine Translation: A Comparative Study of Attention and Beam Search

## Project Overview
This project investigates the performance of the standard Sequence-to-Sequence (Seq2Seq) framework and evaluates the impact of **Bahdanau Attention** and **Beam Search** on translation quality using the **BLEU Score** metric.

The study compares three distinct configurations:
1. **Baseline:** Vanilla Encoder-Decoder LSTM.
2. **Attention-Enabled:** LSTM with Bahdanau (Additive) Attention.
3. **Enhanced Inference:** LSTM with Bahdanau Attention + Beam Search Decoding.

The objective was to analyze how attention mechanisms and beam search help the Seq2Seq model perform more effectively on a large-scale machine translation dataset.

---

## Dataset
- **Source Language:** English
- **Target Language:** French
- **Dataset Size:** 175,621 sentence pairs
- **Source:** [Kaggle: English-French Language Translation](https://www.kaggle.com/datasets/devicharith/language-translation-englishfrench)
- **Preprocessing:**
    - Lowercasing and punctuation removal.
    - Adding `<SOS>` and `<EOS>` tokens.
    - Tokenization and vocabulary generation.
    - Sequence padding for uniform input length.

---

## Model Architectures

### 1. Vanilla Seq2Seq (Encoder-Decoder LSTM)
The baseline model follows the architecture proposed by **Sutskever et al. (2014)**. It consists of an **Encoder** that processes the input sequence into a fixed-length **Context Vector** (the final hidden state), and a **Decoder** that uses this vector to generate the output.

![Vanilla Seq2Seq Architecture](https://media.geeksforgeeks.org/wp-content/uploads/20250529130320642115/Seq2Seq-Model.webp)

* **Key Characteristic:** The entire meaning of the source sentence is "squeezed" into a single vector.
* **Limitation:** As sequence length increases, the fixed-length vector becomes a bottleneck, leading the model to "forget" the beginning of long sentences.

### 2. LSTM with Bahdanau Attention
Based on **Bahdanau et al. (2014)**, this architecture introduces an **Additive Attention** mechanism. Instead of a single bottleneck vector, the decoder creates a dynamic context vector at each time step.



The alignment score $e_{ij}$ is calculated as:
$$e_{ij} = v_a^\top \tanh(W_a [s_{i-1}; h_j])$$

* **Mechanism:** $s_{i-1}$ is the previous decoder state, and $h_j$ is the $j$-th encoder hidden state.
* **Impact:** This allows the model to learn an explicit alignment between source and target words, solving the long-range dependency issue.

### 3. Beam Search Implementation
This project contrasts standard **Greedy Search** with **Beam Search** ([Wu et al., 2016](https://arxiv.org/abs/1609.08144)).



* **Greedy Search:** Selects the single highest-probability token at each step ($k=1$). This often leads to sub-optimal local maxima.
* **Beam Search:** Maintains the top **$k$** most likely partial sequences (hypotheses) at each step.
* **Result:** Reduces incoherent or repetitive sentences by exploring a wider search space.

---

## Training Setup
- **Optimizer:** Adam
- **Loss Function:** Masked Categorical Crossentropy
- **Embedding Dim:** 128
- **Hidden Units:** 128
- **Epochs:** 10
- **Teacher Forcing Ratio:** 1.0

---

## Model Comparison

| Configuration | BLEU Score |
| :--- | :--- |
| **Vanilla Seq2Seq (Baseline)** | 0.1449 |
| **LSTM with Bahdanau Attention (Greedy)** | 0.2496 |
| **LSTM with Bahdanau Attention (Beam Search)** | **0.2760** |

---

## Observations
* **The Bottleneck Effect:** The Baseline model showed a significant drop in performance as sentence length increased beyond 6 tokens. The Attention mechanism effectively improved the BLEU Score and the quality of generated sentences.
* **Decoding Quality:** While Attention improved the *understanding* of the source text, **Beam Search** improved the *fluency* of the output, preventing the model from getting stuck in repetitive loops.

---

## Conclusion
This confirms that **Bahdanau Attention** is essential for handling variable-length sequences by providing a dynamic context. Furthermore, the choice of **decoding strategy** (Beam Search) is equally critical for generating high-quality, human-like translations.

---

## References
* **Sutskever, I., et al. (2014):** [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
* **Bahdanau, D., et al. (2014):** [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
* **Wu, Y., et al. (2016):** [Google's Neural Machine Translation System](https://arxiv.org/abs/1609.08144)
