from sentence_tokenization import *
from transformer_ori import *

def create_masks(eng_batch, kn_batch):
    num_sentences = len(eng_batch)
    look_ahead_mask = torch.full([max_sequence_length, max_sequence_length] , True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    encoder_padding_mask = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
    decoder_padding_mask_self_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
    decoder_padding_mask_cross_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)

    for idx in range(num_sentences):
      eng_sentence_length, kn_sentence_length = len(eng_batch[idx]), len(kn_batch[idx])
      eng_chars_to_padding_mask = np.arange(eng_sentence_length + 1, max_sequence_length)
      kn_chars_to_padding_mask = np.arange(kn_sentence_length + 1, max_sequence_length)
      encoder_padding_mask[idx, :, eng_chars_to_padding_mask] = True
      encoder_padding_mask[idx, eng_chars_to_padding_mask, :] = True
      decoder_padding_mask_self_attention[idx, :, kn_chars_to_padding_mask] = True
      decoder_padding_mask_self_attention[idx, kn_chars_to_padding_mask, :] = True
      decoder_padding_mask_cross_attention[idx, :, eng_chars_to_padding_mask] = True
      decoder_padding_mask_cross_attention[idx, kn_chars_to_padding_mask, :] = True

    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
    decoder_self_attention_mask =  torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)
    #print(f"encoder_self_attention_mask {encoder_self_attention_mask.size()}: {encoder_self_attention_mask[0, :10, :10]}")
    #print(f"decoder_self_attention_mask {decoder_self_attention_mask.size()}: {decoder_self_attention_mask[0, :10, :10]}")
    #print(f"decoder_cross_attention_mask {decoder_cross_attention_mask.size()}: {decoder_cross_attention_mask[0, :10, :10]}")
    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask


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


criterian = nn.CrossEntropyLoss(ignore_index=korean_to_index[PADDING_TOKEN],
                                reduction='none')

# When computing the loss, we are ignoring cases when the label is the padding token
for params in transformer.parameters():
    if params.dim() > 1:
        nn.init.xavier_uniform_(params)

optim = torch.optim.Adam(transformer.parameters(), lr=1e-4)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


transformer.train()
transformer.to(device)
total_loss = 0
num_epochs = 30

def train():
  for epoch in range(num_epochs):
      print(f"Epoch {epoch}")
      iterator = iter(train_loader)
      for batch_num, batch in enumerate(iterator):
          transformer.train()
          eng_batch, kn_batch = batch
          encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(eng_batch, kn_batch)
          optim.zero_grad()
          kn_predictions = transformer(eng_batch,
                                      kn_batch,
                                      encoder_self_attention_mask.to(device), 
                                      decoder_self_attention_mask.to(device), 
                                      decoder_cross_attention_mask.to(device),
                                      enc_start_token=False,
                                      enc_end_token=False,
                                      dec_start_token=True,
                                      dec_end_token=True)
          labels = transformer.decoder.sentence_embedding.batch_tokenize(kn_batch, start_token=False, end_token=True)
          loss = criterian(
              kn_predictions.view(-1, kn_vocab_size).to(device),
              labels.view(-1).to(device)
          ).to(device)
          valid_indicies = torch.where(labels.view(-1) == korean_to_index[PADDING_TOKEN], False, True)
          loss = loss.sum() / valid_indicies.sum()
          loss.backward()
          optim.step()
          #train_losses.append(loss.item())
          if batch_num % 100 == 0:
              print(f"Iteration {batch_num} : {loss.item()}")
              print(f"English: {eng_batch[0]}")
              print(f"korean Translation: {kn_batch[0]}")
              kn_sentence_predicted = torch.argmax(kn_predictions[0], axis=1)
              predicted_sentence = ""
              for idx in kn_sentence_predicted:
                if idx == korean_to_index[END_TOKEN]:
                  break
                predicted_sentence += index_to_korean[idx.item()]
              print(f"korean Prediction: {predicted_sentence}")


              transformer.eval()
              kn_sentence = ("",)
              eng_sentence = ("should we go to the mall?",)
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
                  next_token_prob_distribution = predictions[0][word_counter] # not actual probs
                  next_token_index = torch.argmax(next_token_prob_distribution).item()
                  next_token = index_to_korean[next_token_index]
                  kn_sentence = (kn_sentence[0] + next_token, )
                  if next_token == END_TOKEN:
                    break
              
              print(f"Evaluation translation (should we go to the mall?) : {kn_sentence}")
              print("-------------------------------------------")


  torch.save(transformer.state_dict(), f'./checkpoint/transformer_{num_epochs}.pth')


if __name__ == '__main__':
  train()