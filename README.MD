# Objective
Develop a system to understand semantic relationships between words, comparing ground truth passages with the outputs of a language model (LLM). This system should identify semantic consistencies, contradictions, or neutrality.

# Methodology

## Natural Language Inference using a Cross-Encoder
Natural Language Inference is the task of determining whether a “hypothesis” is true (entailment), false (contradiction), or undetermined (neutral) given a “premise”.
+ Cross-Encoder
    + A Cross-Encoder encodes two texts simultaneously — producing output values that directly compare the two texts
    + Cross-Encoders are slower and computationally expensive, but more presise when comparing texts — especially when the texts depend on each other and require modeling of detailed interactions
+ Bi-Encoder
    + Bi-Encoders encode texts individually, generating embeddings
    + Generated embeddings are then compared using functions such as cosine similarity
    + Bi-Encoders are faster and more scalable, however, they do not model the direct interactions between texts

[nli-deberta-v3-base](https://huggingface.co/cross-encoder/nli-deberta-v3-base) — a Cross-Encoder trained on the [SNLI](https://nlp.stanford.edu/projects/snli/) and [MultiNLI](https://cims.nyu.edu/~sbowman/multinli/) datasets is used in this demo to model the relationship between texts. This model acheived accuracies of: 92.38 and 90.04 on the SNLI-test set and the MNLI mismatched set. This model is trained to output three scores for a sentence pair: contradiction, entailment, and neutral.

## Sentiment Analysis using RoBERTa
Sentiment Analysis is the process of analyzing digital text to determine if the emotional tone of the message is positive, negative, or neutral.

[twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) — a RoBERTa-base model trained on ~124M tweets is used in this demo to compare the tone of the GT and the model's response.

# Run Demo
To run the demo, run all cells in this [notebook](https://colab.research.google.com/drive/1aw1ie2KO5JUjrFNqk6YoRWTsEaQxeNbS?usp=sharing)
