import math
import time
import torch
from torch import nn, optim
from torch.optim import Adam

from data import *  # 데이터 관련 모듈 import
from models.model.transformer import Transformer  # Transformer 모델 import

from util.bleu import idx_to_word, get_bleu  # BLEU 점수 계산 및 인덱스를 단어로 변환하는 함수 import
from util.epoch_timer import epoch_time  # 에폭 시간 측정 함수 import


# 모델의 학습 가능한 파라미터의 총 개수를 계산하는 함수
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 모델의 가중치를 초기화하는 함수
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)

# Transformer 모델 인스턴스를 생성하고, gpu설정.
model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)

# 모델의 학습 가능한 파라미터 수를 출력
print(f'The model has {count_parameters(model):,} trainable parameters')

# 모델의 가중치를 초기화
model.apply(initialize_weights)

# Adam 옵티마이저 설정
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

# 학습률 스케줄러 설정 (성능 향상이 없을 경우 학습률 감소)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

# 손실 함수 정의 (패딩 인덱스를 무시하도록 설정)
criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

# 모델 학습 함수
def train(model, iterator, optimizer, criterion, clip):
    model.train()  # 모델을 학습 모드로 전환
    epoch_loss = 0  # 에폭 손실 초기화
    for i, batch in enumerate(iterator):
        src = batch.src  # 입력 시퀀스
        trg = batch.trg  # 목표 시퀀스

        optimizer.zero_grad()  # 옵티마이저의 기울기 초기화
        output = model(src, trg[:, :-1])  # 모델 예측 (마지막 토큰 제외)
        output_reshape = output.contiguous().view(-1, output.shape[-1])  # 출력 텐서를 2D로 변경
        trg = trg[:, 1:].contiguous().view(-1)  # 목표 시퀀스를 1D로 변경 (시작 토큰 제외)

        loss = criterion(output_reshape, trg)  # 손실 계산
        loss.backward()  # 역전파
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # 기울기 클리핑
        optimizer.step()  # 옵티마이저 스텝

        epoch_loss += loss.item()  # 손실 합산
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())  # 진행률과 손실 출력

    return epoch_loss / len(iterator)  # 평균 손실 반환

# 모델 평가 함수
def evaluate(model, iterator, criterion):
    model.eval()  # 모델을 평가 모드로 전환
    epoch_loss = 0  # 에폭 손실 초기화
    batch_bleu = []  # BLEU 점수 저장 리스트 초기화
    with torch.no_grad():  # 평가 시에는 기울기 계산 안 함
        for i, batch in enumerate(iterator):
            src = batch.src  # 입력 시퀀스
            trg = batch.trg  # 목표 시퀀스
            output = model(src, trg[:, :-1])  # 모델 예측 (마지막 토큰 제외)
            output_reshape = output.contiguous().view(-1, output.shape[-1])  # 출력 텐서를 2D로 변경
            trg = trg[:, 1:].contiguous().view(-1)  # 목표 시퀀스를 1D로 변경 (시작 토큰 제외)

            loss = criterion(output_reshape, trg)  # 손실 계산
            epoch_loss += loss.item()  # 손실 합산

            total_bleu = []  # 배치의 BLEU 점수 저장 리스트 초기화
            for j in range(batch_size):  # 각 배치에 대해 BLEU 점수 계산
                try:
                    trg_words = idx_to_word(batch.trg[j], loader.target.vocab)  # 목표 시퀀스를 단어로 변환
                    output_words = output[j].max(dim=1)[1]  # 모델 출력에서 예측된 단어 인덱스 추출
                    output_words = idx_to_word(output_words, loader.target.vocab)  # 예측된 단어 인덱스를 단어로 변환
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())  # BLEU 점수 계산
                    total_bleu.append(bleu)  # BLEU 점수 리스트에 추가
                except:
                    pass

            total_bleu = sum(total_bleu) / len(total_bleu)  # 배치의 평균 BLEU 점수 계산
            batch_bleu.append(total_bleu)  # 배치 BLEU 점수 리스트에 추가

    batch_bleu = sum(batch_bleu) / len(batch_bleu)  # 전체 배치의 평균 BLEU 점수 계산
    return epoch_loss / len(iterator), batch_bleu  # 평균 손실과 BLEU 점수 반환

# 모델 학습 및 평가 실행 함수
def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []  # 손실 및 BLEU 점수 저장 리스트 초기화
    for step in range(total_epoch):
        start_time = time.time()  # 에폭 시작 시간 기록
        train_loss = train(model, train_iter, optimizer, criterion, clip)  # 모델 학습
        valid_loss, bleu = evaluate(model, valid_iter, criterion)  # 모델 평가
        end_time = time.time()  # 에폭 종료 시간 기록

        if step > warmup:
            scheduler.step(valid_loss)  # 일정 에폭 이후 학습률 조정

        train_losses.append(train_loss)  # 학습 손실 기록
        test_losses.append(valid_loss)  # 평가 손실 기록
        bleus.append(bleu)  # BLEU 점수 기록
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)  # 에폭 소요 시간 계산

        if valid_loss < best_loss:
            best_loss = valid_loss  # 최저 평가 손실 갱신
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))  # 모델 저장

        # 학습 손실, BLEU 점수, 평가 손실을 파일로 저장
        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open('result/bleu.txt', 'w')
        f.write(str(bleus))
        f.close()

        f = open('result/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        # 에폭 결과 출력
        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')

# 메인 실행 부분
if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)  # 지정된 에폭 동안 학습 및 평가 실행
