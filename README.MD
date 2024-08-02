# Objective
Develop a system to understand semantic relationships between words, comparing ground truth passages with the outputs of a language model (LLM). This system should identify semantic consistencies, contradictions, or neutrality.

# Idea
+ Natural Language Inference (NLI)—a task whereby a system determines whether a pair of sentences instantiates in an entailment, a contradiction, or neutral relation—has been useful for training and evaluating models on sentential reasoning [(Jeretic et al., 2020)](https://ai.meta.com/research/publications/are-natural-language-inference-models-imppressive-learning-implicature-and-presupposition/)
+ Evaluating Consistency with NLI
    + Ensure that the generated output aligns with, and does not contradict the Ground Truth (GT)
    + Identify hallucination, checking to see if the generated output is related to the GT
    + Provides a more nuanced evaluation metric compared to [ROUGE](https://huggingface.co/spaces/evaluate-metric/rouge) / [BLEU](https://huggingface.co/spaces/evaluate-metric/bleu) which check for aligning n-grams
        + Instead of checking for directly matching strings, NLI makes sure the contextual meanings align

# Related Studies
+ FENICE: Factuality Evaluation of summarization based on Natural language Inference and Claim Extraction [(Scire et al., 2024)](https://arxiv.org/html/2403.02270v1)
    + Uses NLI to verify claims against original text
+ BERTScore: Evaluating Text Generation with BERT [(Zhang et al., 2019)](https://arxiv.org/abs/1904.09675)
    + Utilizes BERT and computes token similarity

# Methodology

## Natural Language Inference using a Cross-Encoder
Natural Language Inference is the task of determining whether a “hypothesis” is true (entailment), false (contradiction), or undetermined (neutral) given a “premise”.
+ DeBERTa Cross-Encoder
    + A Cross-Encoder encodes two texts simultaneously — producing output values that directly compare the two texts
        + Input texts are concatenated, special tokens are added to separate them
        + Embeddings within each texts are added as well, to differentiate the two texts
        + Position embeddings are added within each text
    + DeBERTa is designed to efficiently handle long sequence inputs
        + “in theory, the maximum sequence length that can be handled is 24,528” [(DeBERTa Paper)](https://arxiv.org/pdf/2006.03654)
        + However, it struggles to capture detailed relationships between long texts
        + Hence, a seperate algorithm is necessary to split longer texts

[nli-deberta-v3-base](https://huggingface.co/cross-encoder/nli-deberta-v3-base) — a Cross-Encoder trained on the [SNLI](https://nlp.stanford.edu/projects/snli/) and [MultiNLI](https://cims.nyu.edu/~sbowman/multinli/) datasets is used in this demo to model the relationship between texts. This model acheived accuracies of: 92.38 and 90.04 on the SNLI-test set and the MNLI mismatched set. This model is trained to output three scores for a sentence pair: contradiction, entailment, and neutral.


### Cross-Encoder VS Bi-Encoder
+ Cross-Encoder
    + A Cross-Encoder encodes two texts simultaneously
    + Cross-Encoders are slower and computationally expensive, but more presise when comparing texts — especially when the texts depend on each other and require modeling of detailed interactions
+ Bi-Encoder
    + Bi-Encoders encode texts individually, generating embeddings
    + Generated embeddings are then compared using functions such as cosine similarity
    + Bi-Encoders are faster and more scalable, however, they do not model the direct interactions between texts

## Sentiment Analysis using RoBERTa
Sentiment Analysis is the process of analyzing digital text to determine if the emotional tone of the message is positive, negative, or neutral.

[twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) — a RoBERTa-base model trained on ~124M tweets and fine-tuned on the [TweetEval dataset](https://aclanthology.org/2020.findings-emnlp.148/) is used in this demo to compare the tone of the GT and the model's response.

# Run Demo
[Huggingface Spaces Demo](https://huggingface.co/spaces/ethanlim04/semantic-relationships-gt-model)
