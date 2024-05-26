from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets.translation import Multi30k

# 데이터 로더 클래스
class DataLoader:
    source: Field = None  # 원본 언어에 대한 필드
    target: Field = None  # 대상 언어에 대한 필드

    # 초기화 함수
    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        self.ext = ext  # 데이터셋 파일 확장자 설정
        self.tokenize_en = tokenize_en  # 영어 토크나이저 함수
        self.tokenize_de = tokenize_de  # 독일어 토크나이저 함수
        self.init_token = init_token  # 시작 토큰
        self.eos_token = eos_token  # 종료 토큰
        print('dataset initializing start')

    # 데이터셋을 생성하는 함수
    def make_dataset(self):
        if self.ext == ('.de', '.en'):
            # 독일어-영어 데이터셋 설정
            self.source = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)

        elif self.ext == ('.en', '.de'):
            # 영어-독일어 데이터셋 설정
            self.source = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)

        # Multi30k 데이터셋 불러오기 (학습, 검증, 테스트 데이터셋 생성)
        train_data, valid_data, test_data = Multi30k.splits(exts=self.ext, fields=(self.source, self.target))
        return train_data, valid_data, test_data

    # 어휘 사전을 생성하는 함수
    def build_vocab(self, train_data, min_freq):
        self.source.build_vocab(train_data, min_freq=min_freq)  # 원본 언어 어휘 사전 생성
        self.target.build_vocab(train_data, min_freq=min_freq)  # 대상 언어 어휘 사전 생성

    # 데이터셋을 반복자로 만드는 함수
    def make_iter(self, train, validate, test, batch_size, device):
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train, validate, test),
                                                                              batch_size=batch_size,
                                                                              device=device)
        print('dataset initializing done')
        return train_iterator, valid_iterator, test_iterator  # 반복자 반환