from nltk import ngrams

n_ngrams = 4

def calc_ngram_sim(row):
    test_features_splitted = row.true_feature.split(',')
    test_set = set()
    pred_set = set()
    for test_feature in test_features_splitted:
        for ngram in ngrams(test_feature, n_ngrams):
            test_set.add(ngram)

    pred_features_splitted = row.pred_feature.split(',')
    for pred_feature in pred_features_splitted:
        for ngram in ngrams(pred_feature, n_ngrams):
            pred_set.add(ngram)
            
    count = 0
    for true_feature_gram in test_set:
        if true_feature_gram in pred_set:
            count += 1

    coverage = count / len(test_set) * 100
    return coverage

def calc_feature_sim(row):
    test_features_splitted = row.true_feature.split(';')
    pred_features_splitted = row.pred_feature.split(';')
    count = 0
    for true_feature in test_features_splitted:
        if true_feature in pred_features_splitted:
            count += 1

    coverage = count / len(test_features_splitted) * 100
    return coverage