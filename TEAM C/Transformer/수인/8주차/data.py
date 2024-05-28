from conf import *
from util.data_loader import DataLoader
from util.tokenizer import Tokenizer

tokenizer = Tokenizer()

# DataLoader 객체 생성, 파일 확장자를 영어와 독일어로 지정
## 영어와 독일어에 대한 토크나이즈 함수를 전달하고, 시작 토큰과 종료 토큰 설정
loader = DataLoader(ext=('.en', '.de'),
                    tokenize_en=tokenizer.tokenize_en,
                    tokenize_de=tokenizer.tokenize_de,
                    init_token='<sos>', 
                    eos_token='<eos>')

# DataLoader를 사용하여 학습, 검증, 테스트 데이터 생성
train, valid, test = loader.make_dataset()
# 학습 데이터를 기반으로 어휘 사전 구축 (토큰이 포함되기 위한 최소 빈도는 2로 설정)
loader.build_vocab(train_data=train, min_freq=2)
# 학습, 검증, 테스트 데이터셋에 대한 iterator 생성
## 지정된 배치 크기와 device를 사용하여 데이터를 처리
train_iter, valid_iter, test_iter = loader.make_iter(train, valid, test,
                                                     batch_size=batch_size,
                                                     device=device)
# 소스 언어의 vocab에서 패딩 토큰의 인덱스 가져오기
src_pad_idx = loader.source.vocab.stoi['<pad>']
# 타겟 언어의 vocab에서 패딩 토큰의 인덱스 가져오기
trg_pad_idx = loader.target.vocab.stoi['<pad>']
# 타겟 언어의 vocab에서 시작 토큰의 인덱스 가져오기
trg_sos_idx = loader.target.vocab.stoi['<sos>']

# 소스 언어의 vocab 크기 가져오기
enc_voc_size = len(loader.source.vocab)
# 타겟 언어의 vocab 크기 가져오기
dec_voc_size = len(loader.target.vocab)