from review_processing.fine_gain import get_word_sentiment_score
from helper.utils import softmax, word_segment, sigmoid
import torch


def get_coarse_simtiment_score(text, word2vec_model):
    word_seg = word_segment(text)
    sim_word = []
    sim_word_weight = []
    for e in word2vec_model.wv.most_similar(positive=word_seg, topn=10):
        sim_word.append(e[0])
        sim_word_weight.append(e[1])
    return sim_word, softmax(sim_word_weight)


def get_coarse_sentiment_score(model, tokenizer, text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    
    sentiment_scale = [(0.0, 0.19), (0.2, 0.39), (0.4, 0.59), (0.6, 0.79), (0.8, 0.99)]
    highest_prob = probabilities[0][predicted_class].item()
    
    lower_bound, upper_bound = sentiment_scale[predicted_class]
    sentiment_score = lower_bound + highest_prob * (upper_bound - lower_bound)
    
    return sentiment_score

# Get coarse-grained sentiment score
def get_coarse_score(text, word2vec_model):
    word_seg = word_segment(text)
    sim_word, sim_word_weight = get_coarse_simtiment_score(text, word2vec_model)
    score = 0
    for i, j in zip(sim_word, sim_word_weight):
        score += get_word_sentiment_score(i) * j
    return sigmoid(score)

