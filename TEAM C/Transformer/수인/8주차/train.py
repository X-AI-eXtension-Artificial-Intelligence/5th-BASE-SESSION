import math
import time

from torch import nn, optim
from torch.optim import Adam

from data import *
from models.model.transformer import Transformer
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time


def count_parameters(model): # 모델의 학습 가능한 매개 변수 총 개수 계산
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m): # Kaiming uniform 초기화를 사용하여 모델 레이어 가중치 초기화
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)

# Transformer 모델 생성 (하이퍼파리미터 설정)
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

# 모델의 학습 가능한 파라미터 개수 출력
print(f'The model has {count_parameters(model):,} trainable parameters')
# 모델 가중치 초기화 적용
model.apply(initialize_weights)
# Adam optimizer
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)
# ReduceLROnPlateau 스케줄러 설정 (검증 손실 개선 멈출때 learning rate 감소)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)
# 손실함수
criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

# 모델 학습 함수
def train(model, iterator, optimizer, criterion, clip):
    model.train() # 학습 모드
    epoch_loss = 0
    # 데이터 배치 반복
    for i, batch in enumerate(iterator):
        src = batch.src # 소스 문장
        trg = batch.trg # 타겟 문장

        optimizer.zero_grad() # 기울기 초기화
        output = model(src, trg[:, :-1]) # 모델에 입력 데이터 전달
        # 출력 데이터 재구성
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1) # 타겟-실제문장
        # 손실 계산
        loss = criterion(output_reshape, trg)
        loss.backward() # 기울기 계산
        # exploding gradients 방지를 위해 기울기 클리핑
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() # 가중치 업데이트

        epoch_loss += loss.item()
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval() # 평가 모드
    epoch_loss = 0 # 에포크 단위 손실 값 누적
    batch_bleu = [] # 배치별 BLEU 점수 저장 
    with torch.no_grad(): # 그래디언트 계산 X
        for i, batch in enumerate(iterator):
            src = batch.src # 소스 시퀀스 추출
            trg = batch.trg # 타겟 시퀀스 추출 
            output = model(src, trg[:, :-1]) # 모델을 통해 예측된 타겟 시퀀스 생성
            output_reshape = output.contiguous().view(-1, output.shape[-1]) # 손실 계산을 위한 텐서 모양 조정
            trg = trg[:, 1:].contiguous().view(-1) # 타겟 시퀀스 모양 조정

            loss = criterion(output_reshape, trg) # 손실 계산
            epoch_loss += loss.item() # 에포크 총 손시에 배치 손실 누적

            total_bleu = []
            for j in range(batch_size):
                try:
                    trg_words = idx_to_word(batch.trg[j], loader.target.vocab) # 타겟 시퀀스 단어로 변환
                    output_words = output[j].max(dim=1)[1] # 예측된 타겟 시퀀스 단어로 변환
                    output_words = idx_to_word(output_words, loader.target.vocab) # 예측된 타겟 시퀀스를 단어로 변환 
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split()) # BLEU 점수 계산
                    total_bleu.append(bleu) # 배치 BLEU 점수 리스트에 추가
                except:
                    pass

            total_bleu = sum(total_bleu) / len(total_bleu) # 배치 BLEU 점수 평균 계산
            batch_bleu.append(total_bleu) # 배치 BLEU 점수 리스트에 추가

    batch_bleu = sum(batch_bleu) / len(batch_bleu) # 평균 BLEU 점수 리스트에 추가
    return epoch_loss / len(iterator), batch_bleu # 평균 에포크 손실과 평균 BLEU 점수 반환


def run(total_epoch, best_loss):
    # 학습 및 평가 
    train_losses, test_losses, bleus = [], [], []
    for step in range(total_epoch): # 에포크 수만큼 반복
        start_time = time.time() # 시작 시간 기록 
        train_loss = train(model, train_iter, optimizer, criterion, clip) # 모델 학습
        valid_loss, bleu = evaluate(model, valid_iter, criterion) # 모델 평가
        end_time = time.time() # 종료 시간 기록

        if step > warmup: # warmup 이후에만 학습률 스케줄러 업데이트 
            scheduler.step(valid_loss) # 검증 손실값을 기반으로 학습률 조정

        train_losses.append(train_loss) # 학습 손실값을 리스트에 저장
        test_losses.append(valid_loss) # 검증 손실값을 리스트에 저장
        bleus.append(bleu) # BLEU 점수를 리스트에 저장
        epoch_mins, epoch_secs = epoch_time(start_time, end_time) # 에포크 수행 시간 계산

        if valid_loss < best_loss: # 현재 검증 손실값이 최고값보다 작으면
            best_loss = valid_loss # 최고 검증 손실값 업데이트
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss)) # 최고 성능 모델 저장
        # 학습, 검증 손실밧, BLEU 점수 리스트에 텍스트 파일로 저장
        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open('result/bleu.txt', 'w')
        f.write(str(bleus))
        f.close()

        f = open('result/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')


if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)