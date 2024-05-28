import math
from collections import Counter

import numpy as np


def bleu_stats(hypothesis, reference):
    """Compute statistics for BLEU."""
    stats = []
    stats.append(len(hypothesis)) # 번역된 문장
    stats.append(len(reference)) # 정답 문장
    for n in range(1, 5):
        # hypothesis 문장에서 n-gram 개수 세기
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )
        # reference 문장에서 n-gram 개수 세기
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
        )
        # 일치하는 n-gram 개수 (교집합)
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        # 가능한 n-gram 개수
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats


def bleu(stats):
    """Compute BLEU given n-gram statistics."""
    # 입력: status -  bleu_stats 함수에서 반환된 통계 목록 
    if len(list(filter(lambda x: x == 0, stats))) > 0: # 통계값에 0 값이 있으면 0 반환
        return 0
    (c, r) = stats[:2] # 첫 두 통계 추출 (hypothesis 길이, reference 길이)
    # n-gram 정확도 계산
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)


def get_bleu(hypotheses, reference):
    """Get validation BLEU score for dev set."""
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    # 모든 문장 쌍에 대해 통계 계산 및 누적
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    return 100 * bleu(stats)


def idx_to_word(x, vocab):
    # 인덱스 시퀀스를 단어 시퀀스로 변환하는 함수
    # x: 단어 인덱스 시퀀스 (정스 리스트)
    # vocab: 단어 사전 (인덱스 -> 단어 매핑)
    words = []
    for i in x:
        word = vocab.itos[i]
        if '<' not in word:
            words.append(word)
    words = " ".join(words)
    return words # 단어 시퀀스 