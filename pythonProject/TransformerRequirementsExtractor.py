import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load Dataset
# file_path = '/kaggle/input/longtaskdescription-en/long_task_descriptions_en.csv'
file_path = 'long_task_descriptions_en.csv'
data = pd.read_csv(file_path)

# Use "fulltext" as input and "essence" as the target summary
data = [(row['fulltext'], row['essence']) for _, row in data.iterrows()]

# Train-test split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Initialize a pre-trained tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# Custom Dataset with Pre-trained Tokenizer
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


# Custom Transformer Model
class TransformerRequirementsExtractor(nn.Module):
    def __init__(self, embed_size, num_heads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerRequirementsExtractor, self).__init__()

        # Use BERT model as an embedding layer
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.embedding_dim = embed_size

        # Encoder and Decoder Layers with batch_first=True
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=num_heads,
                                                        dim_feedforward=dim_feedforward, dropout=dropout,
                                                        batch_first=True)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.embedding_dim, nhead=num_heads,
                                                        dim_feedforward=dim_feedforward, dropout=dropout,
                                                        batch_first=True)

        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)

        # Final linear layer to project back to vocabulary size
        self.fc_out = nn.Linear(self.embedding_dim, self.bert.config.vocab_size)

    def forward(self, src, trg, src_mask=None, trg_mask=None):
        # Use BERT embeddings for the encoder
        src_emb = self.bert(src, attention_mask=src_mask).last_hidden_state
        trg_emb = self.bert(trg, attention_mask=trg_mask).last_hidden_state

        # Encoding and Decoding with key padding masks for src and trg
        memory = self.encoder(src_emb, src_key_padding_mask=src_mask)
        output = self.decoder(trg_emb, memory, tgt_key_padding_mask=trg_mask)

        return self.fc_out(output)


# Training and evaluation functions
def train(model, dataloader, criterion, optimizer, epochs=1):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        with tqdm(dataloader, desc=f"Training Epoch {epoch + 1}") as pbar:
            for batch in pbar:
                src, trg = batch["input_ids"].to(device), batch["labels"].to(device)
                src_mask, trg_mask = batch["attention_mask"].to(device), batch["attention_mask"].to(device)

                # Adjust target input and mask for teacher forcing
                trg_input = trg[:, :-1]
                trg_mask = trg_mask[:, :-1]  # Adjust mask to match trg_input
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


# def predict(model, tokenizer, text, max_length=50):
#     model.eval()
#     src = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)["input_ids"].to(
#         device)
#     src_mask = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)[
#         "attention_mask"].to(device).bool()

#     trg = torch.tensor([tokenizer.encode('<sos>')], dtype=torch.long).to(device)

#     for _ in range(max_length):
#         output = model(src, trg, src_mask=src_mask)
#         pred_token = output.argmax(dim=-1)[:, -1]
#         trg = torch.cat([trg, pred_token.unsqueeze(0)], dim=1)
#         if pred_token.item() == tokenizer.eos_token_id:
#             break

#     return tokenizer.decode(trg.squeeze(0).tolist())

def predict(model, tokenizer, text, max_length=50, num_beams=4):
    model.eval()

    # Encode the input text
    src = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)["input_ids"].to(
        device)
    src_mask = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)[
        "attention_mask"].to(device).bool()

    # Initial target sequence with <sos> token
    trg = torch.tensor([[tokenizer.cls_token_id]], dtype=torch.long).to(device)

    for _ in range(max_length):
        output = model(src, trg, src_mask=src_mask)
        next_token_logits = output[:, -1, :]

        # Get the next token using beam search
        next_token = next_token_logits.softmax(dim=-1).topk(num_beams).indices[:,
                     0]  # Pick top token if greedy decoding

        # Append the token to the sequence
        trg = torch.cat([trg, next_token.unsqueeze(0)], dim=1)

        # Stop generation on <eos> token or [SEP]
        if next_token.item() in [tokenizer.sep_token_id, tokenizer.eos_token_id]:
            break

    # Decode the generated tokens to text, skipping special tokens
    return tokenizer.decode(trg.squeeze(0).tolist(), skip_special_tokens=True)


# Instantiate and use dataset and model as before
if __name__ == '__main__':
    # Create datasets with pre-trained tokenizer
    train_dataset = RequirementsDataset(train_data, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Model parameters
    embed_size = 768  # BERT embedding size
    num_heads = 8
    num_encoder_layers = 4
    num_decoder_layers = 4
    dim_feedforward = 512

    # Initialize model, criterion, and optimizer
    model = TransformerRequirementsExtractor(embed_size, num_heads, num_encoder_layers, num_decoder_layers,
                                             dim_feedforward).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Train model
    train(model, train_dataloader, criterion, optimizer)

    # Predict on a sample
    sample_text = "To ensure effective operation within our team, we need a system that handles task assignments, tracks deadlines, and generates regular progress reports."
    print("Predicted Summary:", predict(model, tokenizer, sample_text))