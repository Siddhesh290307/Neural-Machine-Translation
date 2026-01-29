### Neural Machine Translation: A Comparative Study of Attention and Beam Search

## Project Overview

This project investigates the performance of the standard Sequence-to-Sequence (Seq2Seq) framework and evaluates the impact of Bahdanau Attention and Beam Search on translation quality with the help of Bleu-Score

The study compares three distinct configurations:

- Baseline: Vanilla Encoder-Decoder LSTM.

- Attention-Enabled: LSTM with Bahdanau (Additive) Attention.

- Enhanced Inference: LSTM with Bahdanau Attention + Beam Search Decoding.

The objective was to analyze how attention mechanisms and beam search helps the Sequence-to-Sequence (Seq2Seq) model to perform better on a machine translation dataset.


## Dataset

- Source Language: English

- Target Language: French

- Dataset Size: 175621

- Dataset Source: https://www.kaggle.com/datasets/devicharith/language-translation-englishfrench

- Preprocessing:
    - Lowercasing and punctuation removal.
    - Adding <S0S> and <EOS> tags
    - Tokenization.
    - Generating vocabulary
    - Padding sequences


## Model Architectures

### 1. Vanilla Seq2Seq (Encoder-Decoder LSTM)
The baseline model follows the architecture proposed by **Sutskever et al. (2014)** in ["Sequence to Sequence Learning with Neural Networks"](https://arxiv.org/abs/1409.3215). It consists of an **Encoder** that processes the input sequence into a fixed-length **Context Vector** (the final hidden state), and a **Decoder** that uses this vector to generate the output.

* **Key Characteristic:** The entire meaning of the source sentence is "squeezed" into a single vector.
* **Limitation:** As sequence length increases, the fixed-length vector becomes a bottleneck, leading the model to forget the beginning of long sentences.


![Vanilla Seq2Seq Architecture](https://media.geeksforgeeks.org/wp-content/uploads/20250529130320642115/Seq2Seq-Model.webp)


### 2. LSTM with Bahdanau Attention
Based on the seminal paper **"Neural Machine Translation by Jointly Learning to Align and Translate"** ([Bahdanau et al., 2014](https://arxiv.org/abs/1409.0473)), this architecture introduces an **Additive Attention** mechanism. Instead of a single bottleneck vector, the decoder creates a dynamic context vector at each time step.

The alignment score $e_{ij}$ is calculated as:
$$e_{ij} = v_a^\top \tanh(W_a [s_{i-1}; h_j])$$

* **Mechanism:** $s_{i-1}$ is the previous decoder state, and $h_j$ is the $j$-th encoder hidden state.
* **Impact:** This allows the model to learn an explicit alignment between source and target words, effectively solving the long-range dependency issue.



### 3. Beam Search Implementation
While the model outputs a probability distribution over the vocabulary at each step, the **Decoding Strategy** determines which word is actually chosen. This project contrasts standard **Greedy Search** with **Beam Search** ([Wu et al., 2016](https://arxiv.org/abs/1609.08144)).

* **Greedy Search:** Selects the single highest-probability token at each step ($k=1$). This often leads to local optima and sub-optimal overall translations.
* **Beam Search:** Maintains the top **$k$** most likely partial sequences (hypotheses) at each step.
* **Result:** Significantly reduces the likelihood of generating incoherent or repetitive sentences by exploring a wider search space.



---

## Training Setup

- **Optimizer:** Adam
- **Loss Function:** Masked Categorical Crossentropy
- **Embedding Dim:** [128]
- **Hidden Units:** [128]
- **Epochs:** [10]
- **Teacher Forcing Ratio:** [1.0]

---

## Comaprison

- Vanilla Seq2Seq (Encoder-Decoder LSTM) achieved a Bleu score of 0.1449
- LSTM with Bahdanau Attention achieved a Bleu score of 0.2496
- LSTM with Bahdanau Attention achieved a Bleu score of 0.2760


---

## Observations

* **The Bottleneck Effect:** The Baseline model showed a significant drop in performance as sentence length increased beyond 6 tokens. The Attention mechanism effectively improved the Bleu Score and quality of sentences generated.
* **Decoding Quality:** While Attention improved the *understanding* of the source, **Beam Search** improved the *fluency* of the output, preventing the model from getting stuck in repetitive loops.

---

## Conclusion
This research confirms that **Bahdanau Attention** is essential for handling variable-length sequences. Furthermore, while the attention mechanism improves the underlying model, the choice of **decoding strategy** (Beam Search) is equally critical for generating high-quality, human-like translations.


## References

### Core Architectures
* **Vanilla Seq2Seq:** Sutskever, I., Vinyals, O., & Le, Q. V. (2014). *Sequence to Sequence Learning with Neural Networks*. [arXiv:1409.3215](https://arxiv.org/abs/1409.3215)
* **Bahdanau Attention:** Bahdanau, D., Cho, K., & Bengio, Y. (2014). *Neural Machine Translation by Jointly Learning to Align and Translate*. [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)
* **Beam Search Optimization:** Wu, Y., et al. (2016). *Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation*. [arXiv:1609.08144](https://arxiv.org/abs/1609.08144)

