from sentence_transformers import CrossEncoder
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# 90.04% accuracy on MNLI mismatched set
nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')

def compute_metric(ground_truth: str, inference: str) -> dict:
    scores = nli_model.predict([ground_truth, inference], apply_softmax=True)
    label = ['contradiction', 'entailment', 'neutral'][scores.argmax()]
    return {
        'label': label,
        'contradiction': scores[0],
        'entailment': scores[1],
        'neutral': scores[2],
    }

def _compare_tone(text: str) -> dict:
    # Trained on ~124M Tweets for sentiment analysis
    model_name = r"cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    result = {}
    for i in range(scores.shape[0]):
        l = config.id2label[ranking[i]]
        s = scores[ranking[i]]
        result[l] = np.round(float(s), 4)

    return result

def compare_tone(ground_truth: str, inference: str) -> dict:
    gt = _compare_tone(ground_truth)
    model_res = _compare_tone(inference)
    return {"gt": gt, "model": model_res}
    
if __name__ == "__main__":
    print(compute_metric("Foxes are closer to dogs than they are to cats. Therefore, foxes are not cats.", "Foxes are not cats."))
    print(compute_metric("Foxes are closer to dogs than they are to cats. Therefore, foxes are not cats.", "Foxes are cats."))
    print(compare_tone("This is neutural", "Wtf"))
