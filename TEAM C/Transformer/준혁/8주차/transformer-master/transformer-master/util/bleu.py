import math
from collections import Counter

import numpy as np

# BLEU 점수를 계산하기 위한 통계치를 구하는 함수
def bleu_stats(hypothesis, reference):
    """BLEU를 위한 통계치 계산."""
    stats = []
    stats.append(len(hypothesis))  # 가설 문장의 길이 추가
    stats.append(len(reference))  # 참조 문장의 길이 추가
    for n in range(1, 5):
        # 가설 문장의 n-그램 추출
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )
        # 참조 문장의 n-그램 추출
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
        )
        # 가설 문장과 참조 문장에서 중복되는 n-그램의 개수와 가설 문장의 n-그램 개수를 계산하여 추가
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats

# 통계치를 바탕으로 BLEU 점수를 계산하는 함수
def bleu(stats):
    """n-그램 통계치를 바탕으로 BLEU 계산."""
    # 통계치 중 0이 하나라도 있으면 BLEU 점수는 0
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    (c, r) = stats[:2]  # 가설 문장과 참조 문장의 길이를 가져옴
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.  # n-그램의 로그 정확도 합산
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)  # BLEU 점수 계산

# 개발 세트의 BLEU 점수를 계산하는 함수
def get_bleu(hypotheses, reference):
    """개발 세트에 대한 BLEU 점수 계산."""
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))  # 각 문장에 대한 통계치를 누적
    return 100 * bleu(stats)  # BLEU 점수를 백분율로 반환

# 인덱스를 단어로 변환하는 함수
def idx_to_word(x, vocab):
    words = []
    for i in x:
        word = vocab.itos[i]  # 인덱스를 단어로 변환
        if '<' not in word:  # 특수 토큰을 제외
            words.append(word)
    words = " ".join(words)  # 단어들을 공백으로 연결하여 문장으로 반환
    return words