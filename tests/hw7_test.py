import nltk
from smoothed_ngrams import fit_ngram_model
from lexical_uniformity import lexical_uniformity
from odd_one_out import odd_one_out

from nltk.corpus import gutenberg
from nltk.corpus import wordnet as wn
from nltk.lm.preprocessing import padded_everygram_pipeline

from pytest import approx

def test_lexical_uniformity():
    test_data = ["It", "was", "the", "best", "of", "times", "it", "was", "the", "worst", "of", "times"]
    uniformity = lexical_uniformity(test_data)
    assert uniformity == 1.5

def test_smoothed_ngrams_defaults():
    nltk.download('gutenberg')
    nltk.download('punkt_tab')
    training_data = gutenberg.sents('melville-moby_dick.txt')
    test_sents = [
        ["Queequeg", "loves", "salt", "."],
        ["Captain", "Ahab", "."]
    ]
    test_sent_grams, _ = padded_everygram_pipeline(2, test_sents)
    test_grams_list = list(test_sent_grams)
    model = fit_ngram_model(training_data)
    assert model.perplexity(test_grams_list[0]) == float('inf')
    assert model.perplexity(test_grams_list[1]) == approx(48.29138290283018)

def test_smoothed_ngrams_trigram():
    nltk.download('gutenberg')
    nltk.download('punkt_tab')
    training_data = gutenberg.sents('melville-moby_dick.txt')
    test_sents = [
        ["Queequeg", "loves", "salt", "."],
        ["Captain", "Ahab", "."]
    ]
    test_sent_grams, _ = padded_everygram_pipeline(3, test_sents)
    test_grams_list = list(test_sent_grams)
    model = fit_ngram_model(training_data, n=3)
    assert model.perplexity(test_grams_list[0]) == float('inf')
    assert model.perplexity(test_grams_list[1]) == approx(17.260634264039922)

def test_smoothed_ngrams_smoothed_bigram():
    nltk.download('gutenberg')
    nltk.download('punkt_tab')
    training_data = gutenberg.sents('melville-moby_dick.txt')
    test_sents = [
        ["Queequeg", "loves", "salt", "."],
        ["Captain", "Ahab", "."]
    ]
    test_sent_grams, _ = padded_everygram_pipeline(2, test_sents)
    test_grams_list = list(test_sent_grams)
    model = fit_ngram_model(training_data, smoothed=True)
    assert model.perplexity(test_grams_list[0]) == approx(1138.794952109668)
    assert model.perplexity(test_grams_list[1]) == approx(159.99056335551376)

def test_smoothed_ngrams_smoothed_trigram():
    nltk.download('gutenberg')
    nltk.download('punkt_tab')
    training_data = gutenberg.sents('melville-moby_dick.txt')
    test_sents = [
        ["Queequeg", "loves", "salt", "."],
        ["Captain", "Ahab", "."]
    ]
    test_sent_grams, _ = padded_everygram_pipeline(3, test_sents)
    test_grams_list = list(test_sent_grams)
    model = fit_ngram_model(training_data, n=3, smoothed=True)
    assert model.perplexity(test_grams_list[0]) == approx(545.5142057909777)
    assert model.perplexity(test_grams_list[1]) == approx(113.28791415497945)

def test_odd_one_out():
    nltk.download('wordnet')
    right = wn.synset('right_whale.n.01')
    orca = wn.synset('orca.n.01')
    minke = wn.synset('minke_whale.n.01')
    tortoise = wn.synset('tortoise.n.01')

    result = odd_one_out([right, orca, minke, tortoise])
    assert result == tortoise
