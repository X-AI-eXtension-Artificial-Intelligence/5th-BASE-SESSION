import spacy


# 토크나이저 클래스 정의
class Tokenizer:

    # 초기화 함수
    def __init__(self):
        self.spacy_de = spacy.load('de_core_news_sm')  # 독일어 모델 로드
        self.spacy_en = spacy.load('en_core_web_sm')  # 영어 모델 로드

    # 독일어 텍스트를 토큰화하는 함수
    def tokenize_de(self, text):
        """
        문자열 형태의 독일어 텍스트를 리스트 형태로 토큰화합니다.
        """
        return [tok.text for tok in self.spacy_de.tokenizer(text)]

    # 영어 텍스트를 토큰화하는 함수
    def tokenize_en(self, text):
        """
        문자열 형태의 영어 텍스트를 리스트 형태로 토큰화합니다.
        """
        return [tok.text for tok in self.spacy_en.tokenizer(text)]