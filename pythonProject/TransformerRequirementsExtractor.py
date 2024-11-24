import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


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
        self.position_embedding = nn.Embedding(512, embed_size)
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


class RequirementExtractorNN:
    def __init__(self, embed_size=512, num_heads=8, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=1024,
                 dropout=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.vocab_size = self.tokenizer.vocab_size

        self.model = TransformerRequirementsExtractor(
            vocab_size=self.vocab_size,
            embed_size=embed_size,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def train_from_dataset(self, file_path="long_task_descriptions_en.csv", batch_size=8, epochs=3):
        data = pd.read_csv(file_path)
        data = [(row['fulltext'], row['essence']) for _, row in data.iterrows()]

        train_data, _ = train_test_split(data, test_size=0.2, random_state=42)
        train_dataset = RequirementsDataset(train_data, self.tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            with tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}") as pbar:
                for batch in pbar:
                    src, trg = batch["input_ids"].to(self.device), batch["labels"].to(self.device)
                    src_mask, trg_mask = batch["attention_mask"].to(self.device), batch["attention_mask"].to(self.device)

                    trg_input = trg[:, :-1]
                    trg_mask = trg_mask[:, :-1]
                    trg_target = trg[:, 1:]

                    src_mask = src_mask.bool()
                    trg_mask = trg_mask.bool()

                    self.optimizer.zero_grad()
                    output = self.model(src, trg_input, src_mask=src_mask, trg_mask=trg_mask)
                    output = output.reshape(-1, output.shape[-1])
                    trg_target = trg_target.reshape(-1)

                    loss = self.criterion(output, trg_target)
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()
                    avg_loss = total_loss / (pbar.n + 1)
                    pbar.set_postfix({"Loss": avg_loss})
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}")

        self.save_model()

    def save_model(self, path="./best_model.pt"):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path="./best_model.pt"):
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()

    def predict(self, text, max_length=50, num_beams=4):
        if not self.model:
            raise ValueError("Model is not loaded. Please load a trained model first.")

        self.model.eval()
        src = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)[
            "input_ids"].to(self.device)
        src_mask = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)[
            "attention_mask"].to(self.device).bool()

        trg = torch.tensor([[self.tokenizer.cls_token_id]], dtype=torch.long).to(self.device)
        beam_candidates = [(trg, 0)]

        for _ in range(max_length):
            new_candidates = []
            for candidate, score in beam_candidates:
                output = self.model(src, candidate, src_mask=src_mask)
                next_token_logits = output[:, -1, :].softmax(dim=-1)
                topk_tokens = next_token_logits.topk(num_beams)

                for k in range(num_beams):
                    next_token = topk_tokens.indices[0, k].unsqueeze(0).unsqueeze(0)
                    new_candidate = torch.cat([candidate, next_token], dim=1)
                    new_score = score + topk_tokens.values[0, k].item()

                    if next_token.item() == self.tokenizer.sep_token_id:
                        return self.tokenizer.decode(new_candidate.squeeze(0).tolist(), skip_special_tokens=True)

                    new_candidates.append((new_candidate, new_score))

            beam_candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)[:num_beams]

        best_sequence = beam_candidates[0][0]
        return self.tokenizer.decode(best_sequence.squeeze(0).tolist(), skip_special_tokens=True)