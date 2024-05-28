from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets.translation import Multi30k

class DataLoader:
    source: Field = None # 소스 데이터 필드
    target: Field = None # 타겟 데이터 필드

    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        self.ext = ext # 데이터 파일 확장자 ('.de', '.en')
        self.tokenize_en = tokenize_en # 영어 토크나이저
        self.tokenize_de = tokenize_de # 독일어 토크나이저
        self.init_token = init_token # 시작 토큰
        self.eos_token = eos_token # 종료 토큰
        print('dataset initializing start')

    def make_dataset(self):
        if self.ext == ('.de', '.en'):
            self.source = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True) # 독일어 필드 설정
            self.target = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True) # 영어 필드 설정

        elif self.ext == ('.en', '.de'):
            self.source = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True) # 영어 필드 설정
            self.target = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True) # 독일어 필드 설정

        train_data, valid_data, test_data = Multi30k.splits(exts=self.ext, fields=(self.source, self.target))
        return train_data, valid_data, test_data 

    def build_vocab(self, train_data, min_freq):
        self.source.build_vocab(train_data, min_freq=min_freq) # 입력 언어 vocab 생성
        self.target.build_vocab(train_data, min_freq=min_freq) # 출력 언어 vocab 생성

    def make_iter(self, train, validate, test, batch_size, device):
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train, validate, test),
                                                                              batch_size=batch_size,
                                                                              device=device)
        print('dataset initializing done')
        return train_iterator, valid_iterator, test_iterator