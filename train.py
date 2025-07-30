# train.py - Fixed DeepSeek V3 Implementation

from datasets import load_dataset
from transformers import AutoTokenizer
import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import asdict, dataclass
from torch.nn import RMSNorm
from pathlib import Path
import wandb
from torch.utils.data import DataLoader
import tqdm
from torchinfo import summary

torch.set_float32_matmul_precision('medium')
# Enable MPS optimizations
if torch.backends.mps.is_available():
    torch.mps.set_per_process_memory_fraction(0.8)

@dataclass
class ModelArgs:
    block_size: int = 128
    batch_size: int = 16  # Increased for better MPS utilization
    embeddings_dims: int = 512
    attn_dropout: float = 0.1
    no_of_heads: int = 8
    dropout: float = 0.1
    epochs: int = 3
    max_lr: float = 6e-4
    no_of_decoder_layers: int = 6
    weight_decay_optim: float = 0.1
    beta_1: float = 0.9
    beta_2: float = 0.95
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    vocab_size: int = None
    base_freq: float = 100000
    experts: int = 8
    clip: float = 1.0
    top_experts: int = 4
    noisy_topk: bool = False
    use_checkpointing: bool = False
    use_shared_expert: bool = True
    ignore_pad_token_in_loss: bool = True
    eps: float = 1e-8
    loss_scale: float = 0.3
    useauxFreeLoadBalancingLoss: bool = True
    aux_free_bias_update_rate: float = 0.001
    mtp_heads: int = 2  # Reduced for stability
    latent_dim: int = 64

ModelArgs = ModelArgs()
print(f"Device: {ModelArgs.device}")

# Load datasets
print("Loading datasets...")
fw_train = load_dataset("roneneldan/TinyStories", split="train[:10000]")  # Subset for faster testing
fw_test = load_dataset("roneneldan/TinyStories", split="validation[:1000]")

# Initialize tokenizer
print("Initializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
ModelArgs.vocab_size = len(tokenizer.get_vocab())
print(f"Vocabulary size: {ModelArgs.vocab_size}")

def prepare_dataset(split, batch_size):
    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        encodings = tokenizer(
            texts,
            max_length=ModelArgs.block_size,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        # Create labels (next token prediction)
        encodings["labels"] = encodings["input_ids"].clone()
        encodings["labels"][:, :-1] = encodings["input_ids"][:, 1:]
        encodings["labels"][:, -1] = tokenizer.pad_token_id
        return encodings
    
    dataset = fw_train if split == 'train' else fw_test
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn, 
        shuffle=(split == 'train'), 
        drop_last=True,
        num_workers=0  # Use 0 for MPS compatibility
    )

class Normalization(nn.Module):
    def __init__(self, embeddings_dims: int = ModelArgs.embeddings_dims):
        super().__init__()
        self.rmsnorm_layer = RMSNorm(embeddings_dims)

    def forward(self, x):
        return self.rmsnorm_layer(x)

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return x * self.sig(x)

class SWiGLUExpertMoE(nn.Module):
    def __init__(self, embeddings_dims: int = ModelArgs.embeddings_dims, device=ModelArgs.device):
        super().__init__()
        self.hidden_dims = ((embeddings_dims * 2) * 4) // 3
        self.swish = Swish()
        self.linear_layer1 = nn.Linear(embeddings_dims, self.hidden_dims, bias=False, device=device)
        self.linear_layer2 = nn.Linear(embeddings_dims, self.hidden_dims, bias=False, device=device)
        self.linear_layer3 = nn.Linear(self.hidden_dims, embeddings_dims, bias=False, device=device)

    def forward(self, x):
        swish_res = self.swish(self.linear_layer1(x))
        x_V = self.linear_layer2(x)
        res = torch.mul(swish_res, x_V)
        return self.linear_layer3(res)

class MoeLayer(nn.Module):
    def __init__(self, dropout=ModelArgs.dropout, embeddings_size=ModelArgs.embeddings_dims, device=ModelArgs.device):
        super().__init__()
        self.heads = nn.ModuleList([
            SWiGLUExpertMoE(embeddings_dims=embeddings_size, device=device) 
            for _ in range(ModelArgs.experts)
        ])
        self.gate = nn.Linear(embeddings_size, ModelArgs.experts, device=device, bias=False)
        self.shared_expert = (
            SWiGLUExpertMoE(embeddings_dims=embeddings_size, device=device) 
            if ModelArgs.use_shared_expert else None
        )
        self.device = device
        
        if ModelArgs.useauxFreeLoadBalancingLoss:
            self.register_buffer('routing_bias', torch.zeros(ModelArgs.experts, device=device))
            self.bias_update_speed = ModelArgs.aux_free_bias_update_rate

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        gate_out = self.gate(x)
        
        if ModelArgs.useauxFreeLoadBalancingLoss:
            gate_out = gate_out + self.routing_bias
            
        # Top-k routing
        top_k_values, top_k_indices = torch.topk(gate_out, k=ModelArgs.top_experts, dim=-1)
        
        # Create mask for selected experts
        mask = torch.zeros_like(gate_out, device=self.device)
        mask.scatter_(-1, top_k_indices, 1.0)
        
        # Apply softmax to get routing probabilities
        masked_gate = gate_out.masked_fill(mask == 0, float('-inf'))
        probs = F.softmax(masked_gate, dim=-1)
        
        # Initialize output
        out = torch.zeros_like(x)
        
        # Add shared expert output if enabled
        if ModelArgs.use_shared_expert and self.shared_expert:
            out = out + self.shared_expert(x)
        
        # Process each expert
        for expert_idx in range(ModelArgs.experts):
            # Find tokens that should be processed by this expert
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)  # [batch_size, seq_len]
            
            if not expert_mask.any():
                continue
                
            # Get expert weights for selected tokens
            expert_weights = probs[:, :, expert_idx] * mask[:, :, expert_idx]  # [batch_size, seq_len]
            
            # Process tokens for this expert
            selected_tokens = x[expert_mask]  # [num_selected_tokens, embed_dim]
            
            if selected_tokens.numel() == 0:
                continue
                
            expert_output = self.heads[expert_idx](selected_tokens)
            
            # Apply weights and add to output
            weighted_output = expert_output * expert_weights[expert_mask].unsqueeze(-1)
            out[expert_mask] = out[expert_mask] + weighted_output
        
        # Update routing bias for load balancing
        if ModelArgs.useauxFreeLoadBalancingLoss and self.training:
            with torch.no_grad():
                # Calculate expert utilization
                expert_counts = (probs * mask).sum(dim=(0, 1))  # [num_experts]
                target_count = expert_counts.mean()
                
                # Update bias to balance load
                bias_update = self.bias_update_speed * (target_count - expert_counts)
                self.routing_bias.add_(bias_update)
        
        return out

class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(self, device, embeddings_dims: int = ModelArgs.embeddings_dims, block_size: int = ModelArgs.block_size):
        super().__init__()
        pe = torch.zeros(block_size, embeddings_dims)
        position = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embeddings_dims, 2).float() * (-math.log(10000.0) / embeddings_dims))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)].to(x.device)

class LatentAttention(nn.Module):
    def __init__(self, attn_dropout=ModelArgs.attn_dropout, embeddings_dims=ModelArgs.embeddings_dims, 
                 no_of_heads=ModelArgs.no_of_heads, device=ModelArgs.device):
        super().__init__()
        self.head_size = embeddings_dims // no_of_heads
        self.latent_dim = ModelArgs.latent_dim
        self.W_k = nn.Linear(self.latent_dim, self.head_size, device=device, bias=False)
        self.W_v = nn.Linear(self.latent_dim, self.head_size, device=device, bias=False)
        self.W_dkv = nn.Linear(embeddings_dims, self.latent_dim, device=device, bias=False)
        self.query = nn.Linear(embeddings_dims, self.head_size, device=device, bias=False)
        self.dropout = nn.Dropout(attn_dropout)
        self.device = device

    def forward(self, x, kv_cache=None, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Compute latent representation
        latent_matrix = self.W_dkv(x)
        
        # Update KV cache
        if kv_cache is None:
            kv_cache = latent_matrix
        else:
            kv_cache = torch.cat([kv_cache, latent_matrix], dim=1)
        
        # Compute keys and values from compressed representation
        compressed_k = self.W_k(kv_cache)  # [batch_size, cache_len, head_size]
        compressed_v = self.W_v(kv_cache)  # [batch_size, cache_len, head_size]
        
        # Compute queries
        q = self.query(x)  # [batch_size, seq_len, head_size]
        
        # Attention computation
        scale = self.head_size ** -0.5
        weights = torch.matmul(q, compressed_k.transpose(-2, -1)) * scale
        
        # Apply causal mask
        cache_len = kv_cache.shape[1]
        causal_mask = torch.tril(torch.ones(seq_len, cache_len, device=self.device))
        weights = weights.masked_fill(causal_mask == 0, float('-inf'))
        
        # Apply additional mask if provided
        if mask is not None:
            weights = weights.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        weights_normalized = F.softmax(weights, dim=-1)
        weights_normalized = self.dropout(weights_normalized)
        
        # Apply attention
        out = torch.matmul(weights_normalized, compressed_v)
        
        return out, kv_cache

class MHLA(nn.Module):
    def __init__(self, device, attn_dropout=ModelArgs.attn_dropout, 
                 embeddings_dims=ModelArgs.embeddings_dims, no_of_heads=ModelArgs.no_of_heads):
        super().__init__()
        self.heads = nn.ModuleList([
            LatentAttention(attn_dropout, embeddings_dims, no_of_heads, device) 
            for _ in range(no_of_heads)
        ])
        self.dropout = nn.Dropout(attn_dropout)
        self.linear = nn.Linear(embeddings_dims, embeddings_dims, device=device, bias=False)

    def forward(self, x, kv_cache=None, mask=None):
        head_outputs = []
        updated_caches = []
        
        for head in self.heads:
            head_out, updated_cache = head(x, kv_cache, mask)
            head_outputs.append(head_out)
            updated_caches.append(updated_cache)
        
        # Concatenate head outputs
        concat = torch.cat(head_outputs, dim=-1)
        out = self.dropout(self.linear(concat))
        
        # For simplicity, return the first cache (in practice, you might want to handle this differently)
        return out, updated_caches[0] if updated_caches else None

class DecoderLayer(nn.Module):
    def __init__(self, device, attn_dropout=ModelArgs.attn_dropout, no_of_heads=ModelArgs.no_of_heads, 
                 embeddings_dims=ModelArgs.embeddings_dims, dropout=ModelArgs.dropout):
        super().__init__()
        self.mha = MHLA(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, 
                       no_of_heads=no_of_heads, device=device)
        self.layer_norm1 = Normalization(embeddings_dims)
        self.layer_norm2 = Normalization(embeddings_dims)
        self.moe_block = MoeLayer(dropout=dropout, embeddings_size=embeddings_dims, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv_cache=None, mask=None):
        # Self-attention with residual connection
        normed_x = self.layer_norm1(x)
        attn_out, kv_cache = self.mha(normed_x, kv_cache, mask)
        x = x + attn_out
        
        # MoE with residual connection
        normed_x = self.layer_norm2(x)
        moe_out = self.moe_block(normed_x)
        x = x + moe_out
        
        return x, kv_cache

class Block(nn.Module):
    def __init__(self, device, embeddings_dims=ModelArgs.embeddings_dims, 
                 no_of_decoder_layers=ModelArgs.no_of_decoder_layers, 
                 block_size=ModelArgs.block_size, vocab_size=ModelArgs.vocab_size, 
                 dropout=ModelArgs.dropout):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embeddings_dims, device=device)
        self.decoder = nn.ModuleList([
            DecoderLayer(device=device, embeddings_dims=embeddings_dims, dropout=dropout) 
            for _ in range(no_of_decoder_layers)
        ])
        self.linear_layer = nn.Linear(embeddings_dims, vocab_size, device=device)
        self.dropout = nn.Dropout(dropout)
        self.norm = Normalization(embeddings_dims)
        
        # Weight tying
        self.embeddings.weight = self.linear_layer.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, mask=None, inference=False):
        x = self.embeddings(x)
        kv_cache = None
        
        for layer in self.decoder:
            x, kv_cache = layer(x, kv_cache, mask)
        
        x = self.dropout(x)
        # Apply scaling as in original code
        x = 2 * (ModelArgs.no_of_decoder_layers ** -0.5) * x
        x = self.norm(x)
        
        return self.linear_layer(x)

class DeepSeekV3(nn.Module):
    def __init__(self, device, embeddings_dims=ModelArgs.embeddings_dims, 
                 block_size=ModelArgs.block_size, vocab_size=ModelArgs.vocab_size, 
                 dropout=ModelArgs.dropout):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embeddings_dims, device=device)
        self.pos_embeddings = SinusoidalPositionalEmbeddings(
            device=device, embeddings_dims=embeddings_dims, block_size=block_size
        )
        
        # Main decoder block
        self.decoder = Block(
            device=device, embeddings_dims=embeddings_dims, 
            no_of_decoder_layers=ModelArgs.no_of_decoder_layers, 
            block_size=block_size, vocab_size=vocab_size, dropout=dropout
        )
        
        # Multi-token prediction components (simplified)
        self.mtp_heads = ModelArgs.mtp_heads
        if self.mtp_heads > 1:
            self.mtp_layers = nn.ModuleList([
                nn.Linear(embeddings_dims, vocab_size, device=device) 
                for _ in range(self.mtp_heads - 1)
            ])
        
        # Weight tying
        self.embedding.weight = self.decoder.embeddings.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, inference=False, mask=None):
        input_ids = input_ids.long()
        
        # Embeddings
        x = self.embedding(input_ids)
        x = x + self.pos_embeddings(x)
        
        if inference or self.mtp_heads == 1:
            # Standard next-token prediction
            return self.decoder(input_ids, mask=mask, inference=inference)
        
        # Multi-token prediction training (simplified version)
        B, T, C = x.shape
        
        # Get base representations
        base_logits = self.decoder(input_ids, mask=mask)  # [B, T, vocab_size]
        
        # For simplicity, just return the base logits
        # The original MTP implementation was complex and had issues
        return base_logits

def get_lr(it):
    warmup_iters = 200 * ModelArgs.epochs
    lr_decay_iters = 1000 * ModelArgs.epochs
    min_lr = 0.1 * ModelArgs.max_lr
    
    if it < warmup_iters:
        return ModelArgs.max_lr * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (ModelArgs.max_lr - min_lr)

def topk_sampling(model, prompt, device, max_length=50, top_k=50, temperature=0.9):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated = input_ids
    
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(generated, inference=True)[:, -1, :]
            logits = logits / temperature
            
            # Top-k sampling
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                probs = F.softmax(top_k_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, num_samples=1)
                next_token = torch.gather(top_k_indices, -1, next_token_idx)
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    model.train()
    return tokenizer.decode(generated[0], skip_special_tokens=True)

def save_text(file_path, step, text):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'a') as f:
        f.write(f"Step {step}: {text}\n")

def train():
    device = ModelArgs.device
    print(f"Training on device: {device}")
    
    # Initialize W&B
    wandb.init(project='DSV-Training', config=asdict(ModelArgs))
    
    # Initialize model
    model = DeepSeekV3(device=device)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=ModelArgs.max_lr, 
        betas=(ModelArgs.beta_1, ModelArgs.beta_2), 
        weight_decay=ModelArgs.weight_decay_optim, 
        eps=ModelArgs.eps
    )
    
    # Training parameters
    total_iters = 500 * ModelArgs.epochs
    eval_iters = 50
    save_checkpoint_iter = 100
    total_batch_size = 32768  # Reduced for MPS
    gradient_accumulation_steps = max(1, total_batch_size // (ModelArgs.batch_size * ModelArgs.block_size))
    
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    
    # Data loaders
    train_dataloader = prepare_dataset('train', ModelArgs.batch_size)
    val_dataloader = prepare_dataset('val', ModelArgs.batch_size)
    train_iterator = iter(train_dataloader)
    val_iterator = iter(val_dataloader)

    @torch.inference_mode()
    def estimate_loss():
        nonlocal val_iterator
        model.eval()
        losses = torch.zeros(eval_iters, device=device)
        
        for k in range(eval_iters):
            try:
                batch = next(val_iterator)
            except StopIteration:
                val_iterator = iter(val_dataloader)
                batch = next(val_iterator)
            
            input_ids = batch["input_ids"].to(device).long()
            targets = batch["labels"].to(device).long()
            
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=tokenizer.pad_token_id,
                reduction='mean'
            )
            losses[k] = loss
        
        model.train()
        return losses.mean().item()

    # Training loop
    model.train()
    token_count = 0
    
    print("Starting training...")
    for step in tqdm.tqdm(range(total_iters), desc="Training"):
        accumulated_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        for micro_step in range(gradient_accumulation_steps):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                batch = next(train_iterator)
            
            input_ids = batch['input_ids'].to(device).long()
            targets = batch['labels'].to(device).long()
            token_count += input_ids.numel()
            
            # Forward pass
            logits = model(input_ids)
            
            # Compute loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=tokenizer.pad_token_id,
                reduction='mean'
            ) / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            accumulated_loss += loss.item()
        
        # Gradient clipping
        total_norm = 0.0
        if ModelArgs.clip > 0.0:
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=ModelArgs.clip)
        
        # Optimizer step
        optimizer.step()
        
        # Evaluation and logging
        if step % eval_iters == 0 or step == total_iters - 1:
            val_loss = estimate_loss()
            perplexity = math.exp(min(val_loss, 10))  # Cap to prevent overflow
            
            wandb.log({
                "step": step,
                "train_loss": accumulated_loss,
                "val_loss": val_loss,
                "perplexity": perplexity,
                "tokens_processed": token_count,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "gradient_norm": total_norm if isinstance(total_norm, float) else total_norm.item()
            })
            
            print(f"Step {step}: train_loss={accumulated_loss:.4f}, val_loss={val_loss:.4f}, perplexity={perplexity:.2f}")
        
        # Save checkpoint
        if step % save_checkpoint_iter == 0 and step > 0:
            checkpoint_path = f"checkpoint_{step}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step,
                'loss': accumulated_loss,
                'model_args': asdict(ModelArgs)
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        
        # Generate sample text
        if step % eval_iters == 0:
            prompt = "Once upon a time there lived a baby deer named Bambi."
            try:
                generated_text = topk_sampling(model, prompt, device, max_length=50, top_k=50, temperature=0.9)
                save_text(f"generated_data/generated_text_{step}.txt", step, generated_text)
                print(f"Generated: {generated_text[:100]}...")
            except Exception as e:
                print(f"Generation failed: {e}")
        
        # Update learning rate
        optimizer.param_groups[0]['lr'] = get_lr(step)

    wandb.finish()
    print("Training completed!")

def generate_text(model, prompt, device, max_length=50, top_k=50, temperature=0.9):
    generated = topk_sampling(model, prompt, device, max_length, top_k, temperature)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}")
    return generated

if __name__ == "__main__":
    print(f"Running on {ModelArgs.device}")
    print(f"PyTorch version: {torch.__version__}")
    
    if ModelArgs.device == "mps":
        print(f"MPS available: {torch.backends.mps.is_available()}")
        print(f"MPS built: {torch.backends.mps.is_built()}")
    
    # Train the model
    train()
    
    # Load and test the model
    try:
        model = DeepSeekV3(device=ModelArgs.device)
        checkpoint = torch.load("checkpoint_500.pt", map_location=ModelArgs.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        generate_text(model, "Once upon a time there lived a baby deer named Bambi.", ModelArgs.device)
    except FileNotFoundError:
        print("No checkpoint found. Training completed without final checkpoint.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")