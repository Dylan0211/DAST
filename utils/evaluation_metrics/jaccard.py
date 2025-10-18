
def jaccard_similarity_coefficient(rank_1, rank_2, k):
    """
    larger -> better
    :param rank_1:
    :param rank_2:
    :param k:
    :return:
    """
    temp_rank_1 = rank_1[:k]
    temp_rank_2 = rank_2[:k]
    numerator = set(temp_rank_1).intersection(set(temp_rank_2))
    denominator = set(temp_rank_1).union(set(temp_rank_2))
    return '{:.4f}'.format(len(numerator) / len(denominator))