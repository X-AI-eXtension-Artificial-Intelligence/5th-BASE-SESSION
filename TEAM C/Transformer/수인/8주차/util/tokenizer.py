import spacy

class Tokenizer:

    def __init__(self):
        # 독일어와 영어 토큰화나이저를 위한 spacy 모델 로딩
        self.spacy_de = spacy.load('de_core_news_sm') # 독일어 모델 로딩
        self.spacy_en = spacy.load('en_core_web_sm') # 영어 모델 로딩

    def tokenize_de(self, text):
        """
        Tokenizes German text from a string into a list of strings
        입력: 문자열 형태의 독일어 텍스트
        출력: 분리된 독일어 토큰들의 리스트 (문자열)
        """
        return [tok.text for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        """
        Tokenizes English text from a string into a list of strings
        입력: 문자열 형태의 영어 텍스트
        출력: 분리된 영어 토큰들의 리스트 (문자열)
        """
        return [tok.text for tok in self.spacy_en.tokenizer(text)]