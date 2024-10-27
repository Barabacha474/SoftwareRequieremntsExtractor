import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# file_path = '/kaggle/input/longtaskdescription-en/long_task_descriptions_en.csv'
file_path = 'long_task_descriptions_en.csv'
data = pd.read_csv(file_path)
data = [(row['fulltext'], row['essence']) for _, row in data.iterrows()]

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
vocab_size = tokenizer.vocab_size  # Use tokenizer's vocabulary size for embedding layer


class RequirementsDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, summary = self.data[idx]
        inputs = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True,
                                return_tensors="pt")
        targets = self.tokenizer(summary, max_length=self.max_length, padding="max_length", truncation=True,
                                 return_tensors="pt")

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze()
        }


class TransformerRequirementsExtractor(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_encoder_layers, num_decoder_layers, dim_feedforward,
                 dropout=0.1):
        super(TransformerRequirementsExtractor, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(512, embed_size)  # Position embeddings for sequence length up to 512
        self.embedding_dim = embed_size

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=num_heads,
                                                        dim_feedforward=dim_feedforward, dropout=dropout,
                                                        batch_first=True)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.embedding_dim, nhead=num_heads,
                                                        dim_feedforward=dim_feedforward, dropout=dropout,
                                                        batch_first=True)

        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)

        self.fc_out = nn.Linear(self.embedding_dim, vocab_size)

    def forward(self, src, trg, src_mask=None, trg_mask=None):
        src_positions = torch.arange(0, src.size(1), device=src.device).unsqueeze(0).expand(src.size(0), -1)
        trg_positions = torch.arange(0, trg.size(1), device=trg.device).unsqueeze(0).expand(trg.size(0), -1)

        src_emb = self.embedding(src) + self.position_embedding(src_positions)
        trg_emb = self.embedding(trg) + self.position_embedding(trg_positions)

        memory = self.encoder(src_emb, src_key_padding_mask=src_mask)
        output = self.decoder(trg_emb, memory, tgt_key_padding_mask=trg_mask)

        return self.fc_out(output)


def train(model, dataloader, criterion, optimizer, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        with tqdm(dataloader, desc=f"Training Epoch {epoch + 1}") as pbar:
            for batch in pbar:
                src, trg = batch["input_ids"].to(device), batch["labels"].to(device)
                src_mask, trg_mask = batch["attention_mask"].to(device), batch["attention_mask"].to(device)

                trg_input = trg[:, :-1]
                trg_mask = trg_mask[:, :-1]
                trg_target = trg[:, 1:]

                src_mask = src_mask.bool()
                trg_mask = trg_mask.bool()

                optimizer.zero_grad()
                output = model(src, trg_input, src_mask=src_mask, trg_mask=trg_mask)
                output = output.reshape(-1, output.shape[-1])
                trg_target = trg_target.reshape(-1)

                loss = criterion(output, trg_target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                avg_loss = total_loss / (pbar.n + 1)
                pbar.set_postfix({"Loss": avg_loss})
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")


def predict(model, tokenizer, text, max_length=50, num_beams=4):
    model.eval()
    src = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)["input_ids"].to(
        device)
    src_mask = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)[
        "attention_mask"].to(device).bool()

    trg = torch.tensor([[tokenizer.cls_token_id]], dtype=torch.long).to(device)
    beam_candidates = [(trg, 0)]

    for _ in range(max_length):
        new_candidates = []
        for candidate, score in beam_candidates:
            output = model(src, candidate, src_mask=src_mask)
            next_token_logits = output[:, -1, :].softmax(dim=-1)
            topk_tokens = next_token_logits.topk(num_beams)

            for k in range(num_beams):
                next_token = topk_tokens.indices[0, k].unsqueeze(0).unsqueeze(0)
                new_candidate = torch.cat([candidate, next_token], dim=1)
                new_score = score + topk_tokens.values[0, k].item()

                if next_token.item() == tokenizer.sep_token_id:
                    return tokenizer.decode(new_candidate.squeeze(0).tolist(), skip_special_tokens=True)

                new_candidates.append((new_candidate, new_score))

        beam_candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)[:num_beams]

    best_sequence = beam_candidates[0][0]
    return tokenizer.decode(best_sequence.squeeze(0).tolist(), skip_special_tokens=True)


embed_size = 512
num_heads = 8
num_encoder_layers = 4
num_decoder_layers = 4
dim_feedforward = 1024

model = TransformerRequirementsExtractor(vocab_size, embed_size, num_heads, num_encoder_layers, num_decoder_layers,
                                         dim_feedforward).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

train_dataset = RequirementsDataset(train_data, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
train(model, train_dataloader, criterion, optimizer)

sample_text = "To ensure effective operation within our team, we need a system that handles task assignments, tracks deadlines, and generates regular progress reports."
print("Predicted Summary:", predict(model, tokenizer, sample_text))
