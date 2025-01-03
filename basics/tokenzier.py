import re
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

with open("the-verdict.txt", "r", encoding="utf-8") as f: 
    raw_text = f.read()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text) 
preprocessed = [item.strip() for item in preprocessed if item.strip()] 
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
all_words.extend(["<|endoftext|>", "<|unk|>"])


#v1
class Tokenizer:
    def __init__(self,vocab):
        self.str_to_int={word:id for id,word in enumerate(vocab)}
        self.int_to_str={id:word for word,id in self.str_to_int.items()}

    def encode(self,text):
        preprocessed=re.split(r'([,.?_!"()\']|--|\s)', text) 
        preprocessed=[item.strip() for item in preprocessed if item.strip()]
        preprocessed=[item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids=[self.str_to_int[token] for token in preprocessed]
        return ids
    
    def decode(self,text):
        tokens=[self.int_to_str[id] for id in text]
        return re.sub(r'\s+([,.:;?!"()\'])', r'\1', " ".join(tokens)) #remove spaces befroe punc




#v2
tokenizer=tiktoken.get_encoding("gpt2")
enc_text = tokenizer.encode(raw_text)


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride): 
            input_chunk = token_ids[i:i + max_length] 
            target_chunk = token_ids[i + 1: i + max_length + 1] 
            self.input_ids.append(torch.tensor(input_chunk)) 
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

dataloader = create_dataloader_v1(raw_text, batch_size=2, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
second_batch = next(data_iter)
print(second_batch)

vocab_size = 50257
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
   stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

context_length = max_length
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim) 
pos_embeddings = pos_embedding_layer(torch.arange(context_length)) 
print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)