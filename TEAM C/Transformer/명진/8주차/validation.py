import torch
import torch.nn as nn

import os
from tqdm import tqdm
import torchmetrics
from dataset import causal_mask


# validation 진행
def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    '''source_texts = []
    expected = []
    predicted = []
'''
    '''try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default'''
    # control window 사이즈
    console_width = 80


    with torch.no_grad():
        # val data 배치
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # batch size 1인지 확인
            assert encoder_input.size(0) == 1

            # 인코더 아웃풋 한번만 계산하고, 모든 토큰을 디코더에서 그대로 사용(다시계산x)하도록
            # 인코더 한번만 돌릴 수 있도록 greedy decode 메서드 추가
            
            # 모델 아웃풋 구함
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            # label과 비교
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            # output token -> text
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            '''source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)'''
            
            # Print source, target, model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
            
    # 텐서보드에 추가하는 부분
    '''if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()'''



# Greedy decode
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    # sos, eos id 가져오기
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # encoder output 미리 계산하고 매 step마다 reuse
    encoder_output = model.encode(source, source_mask)
    
    # decoder input을 sos token으로 초기화(1st iteration)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    
    while True:
        if decoder_input.size(1) == max_len: # max_len 넘어서면 break
            break

        # casual mask for target (decoder input)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # output 계산
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # 다음 토큰 받음
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1) # token with maximun probability
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx: # 다음토큰이 eos면 break
            break

    return decoder_input.squeeze(0)