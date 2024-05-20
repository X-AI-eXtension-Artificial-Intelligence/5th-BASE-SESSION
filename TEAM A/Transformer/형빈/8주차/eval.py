from sentence_tokenization import *
from transformer_ori import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--pth", type=str)
parser.add_argument("--sentence", type=str, default="hello world!")

args = parser.parse_args()


pth_name=args.pth,
sentence = args.sentence

d_model = 512
batch_size = 22
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 5
max_sequence_length = 300
NEG_INFTY = -1e9
train_loader = DataLoader(dataset, batch_size)
iterator = iter(train_loader)
kn_vocab_size = len(korean_vocabulary)

transformer = Transformer(d_model, 
                          ffn_hidden,
                          num_heads, 
                          drop_prob, 
                          num_layers, 
                          max_sequence_length,
                          kn_vocab_size,
                          english_to_index,
                          korean_to_index,
                          START_TOKEN, 
                          END_TOKEN, 
                          PADDING_TOKEN)

transformer.load_state_dict(torch.load(f'./checkpoint/{pth_name}'))
transformer.eval()

def translate(eng_sentence):
  eng_sentence = (eng_sentence,)
  kn_sentence = ("",)
  for word_counter in range(max_sequence_length):
    encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask= create_masks(eng_sentence, kn_sentence)
    predictions = transformer(eng_sentence,
                              kn_sentence,
                              encoder_self_attention_mask.to(device), 
                              decoder_self_attention_mask.to(device), 
                              decoder_cross_attention_mask.to(device),
                              enc_start_token=False,
                              enc_end_token=False,
                              dec_start_token=True,
                              dec_end_token=False)
    next_token_prob_distribution = predictions[0][word_counter]
    next_token_index = torch.argmax(next_token_prob_distribution).item()
    next_token = index_to_kannada[next_token_index]
    kn_sentence = (kn_sentence[0] + next_token, )
    if next_token == END_TOKEN:
      break
  return kn_sentence[0]

def translation(sentence):
    translation = translate(sentence)
    print(translation)

if __name__ == '__main__':
    translation(sentence)
