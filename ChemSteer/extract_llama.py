import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import sacrebleu
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import gc
import os

class SteeringVectorExtractor:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        d_prime: int = None,
        inject_layer: int = 24,
        inject_timestep: str = 'all',
        inject_mlp: bool = True,
        inject_every_layer: bool = False,
        inject_embedding: bool = False,
        reg_lambda: float = 5e-3,
        device: torch.device = None
    ):
        self.device = device or (
            torch.device('cuda') if torch.cuda.is_available()
            else torch.device('mps') if torch.backends.mps.is_available()
            else torch.device('cpu')
        )
        print(f"Using device: {self.device}")
        print(f"injecting into layer {inject_layer}")

        self.model = model.to(self.device)
        self.model.eval()
        self.model.config.pad_token_id = self.model.config.eos_token_id
        for p in self.model.parameters(): p.requires_grad = False

        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.d = self.model.config.hidden_size
        self.d_prime = d_prime or self.d
        self.inject_layer = inject_layer
        self.inject_timestep = inject_timestep
        self.inject_mlp = inject_mlp
        self.inject_every_layer = inject_every_layer
        self.inject_embedding = inject_embedding
        self.reg_lambda = reg_lambda

        if self.d_prime < self.d:
            print(f"Creating projection: {self.d_prime}→{self.d}")
            W = torch.randn(self.d_prime, self.d, device=self.device)
            q, _ = torch.linalg.qr(W.t())
            self.W_steer = q.t().detach()
        else:
            self.W_steer = None

    def _inject(self, hidden: torch.Tensor, zsteer: torch.Tensor) -> torch.Tensor:
        vec = (self.W_steer.t() @ zsteer) if self.W_steer is not None else zsteer
        # Normalize steering vector for consistent injection
        vec = vec / (vec.norm() + 1e-8)
        if self.inject_timestep == 'first':
            hidden[:, 0, :] += vec
        else:
            hidden += vec.unsqueeze(0).unsqueeze(1)
        return hidden

    def extract_zsteer(
        self,
        sentence: str,
        steps: int = 500,
        lr: float = 1.0,
    ) -> torch.Tensor:
        # print(f"\n[extract] Target: '{sentence}'")
        tokens = self.tokenizer(
            sentence,
            return_tensors='pt',
            return_attention_mask=True,
            padding=True
        ).to(self.device)
        input_ids = tokens.input_ids
        attention_mask = tokens.attention_mask

        zsteer = nn.Parameter(torch.empty(self.d_prime, device=self.device))
        nn.init.xavier_normal_(zsteer.unsqueeze(0))
        optimizer = torch.optim.Adam([zsteer], lr=lr)

        handles = []
        if self.inject_embedding:
            # print("Injecting at embedding layer.")
            def embed_hook(module, inp, out):
                # out is likely a tensor (embedding output)
                return self._inject(out, zsteer)
            handles.append(self.model.model.embed_tokens.register_forward_hook(embed_hook))

        layers = (
            list(self.model.model.layers) if self.inject_every_layer
            else [self.model.model.layers[self.inject_layer]]
        )
        # print(f"Injecting into {'ALL' if self.inject_every_layer else 'layer '+str(self.inject_layer)}.")
        for layer in layers:
            def attn_hook(module, inp, out, zsteer=zsteer):
                if isinstance(out, tuple):
                    attn_out, *rest = out
                    modified = self._inject(attn_out, zsteer)
                    return (modified, *rest)
                else:
                    return self._inject(out, zsteer)
            handles.append(layer.self_attn.register_forward_hook(attn_hook))

            if self.inject_mlp:
                def mlp_hook(module, inp, out, zsteer=zsteer):
                    if isinstance(out, tuple):
                        mlp_out, *rest = out
                        modified = self._inject(mlp_out, zsteer)
                        return (modified, *rest)
                    else:
                        return self._inject(out, zsteer)
                handles.append(layer.mlp.register_forward_hook(mlp_hook))

        for i in range(1, steps+1):
            optimizer.zero_grad()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            loss = nn.CrossEntropyLoss()(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1)
            )
            loss.backward()
            optimizer.step()
            # if i % max(1, steps//10) == 0 or i == 1:
            #     print(f"  Step {i}/{steps}  Loss: {loss.item():.4f}  Norm: {zsteer.norm().item():.2f}  LR: {optimizer.param_groups[0]['lr']:.4f}")

        for h in handles:
            h.remove()
        z_cpu = zsteer.detach().cpu()
        # print(f"Extracted zsteer first10: {z_cpu.numpy()[:10]}")
        return z_cpu

    def steer_generate(
        self,
        zsteer: torch.Tensor,
        max_new_tokens: int = 20
    ) -> str:
        print(f"\n[generate] norm(zsteer)={zsteer.norm().item():.2f}")
        orig_device = next(self.model.parameters()).device
        self.model.to('cpu')
        z_cpu = zsteer.cpu()
        input_ids = torch.tensor([[self.tokenizer.bos_token_id]])
        attention_mask = torch.ones_like(input_ids)

        handles = []
        if self.inject_embedding:
            def embed_hook(module, inp, out):
                if isinstance(out, tuple):
                    emb_out, *rest = out
                    modified = self._inject(emb_out.to(orig_device), z_cpu.to(orig_device))
                    return (modified.to('cpu'), *rest)
                else:
                    return self._inject(out.to(orig_device), z_cpu.to(orig_device)).to('cpu')
            handles.append(self.model.model.embed_tokens.register_forward_hook(embed_hook))

        layers = (
            list(self.model.model.layers) if self.inject_every_layer
            else [self.model.model.layers[self.inject_layer]]
        )
        for layer in layers:
            def attn_hook(module, inp, out, z_cpu=z_cpu, orig_device=orig_device):
                if isinstance(out, tuple):
                    attn_out, *rest = out
                    modified = self._inject(attn_out.to(orig_device), z_cpu.to(orig_device))
                    return (modified.to('cpu'), *rest)
                else:
                    return self._inject(out.to(orig_device), z_cpu.to(orig_device)).to('cpu')
            handles.append(layer.self_attn.register_forward_hook(attn_hook))
            if self.inject_mlp:
                def mlp_hook(module, inp, out, z_cpu=z_cpu, orig_device=orig_device):
                    if isinstance(out, tuple):
                        mlp_out, *rest = out
                        modified = self._inject(mlp_out.to(orig_device), z_cpu.to(orig_device))
                        return (modified.to('cpu'), *rest)
                    else:
                        return self._inject(out.to(orig_device), z_cpu.to(orig_device)).to('cpu')
                handles.append(layer.mlp.register_forward_hook(mlp_hook))

        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=self.model.config.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        for h in handles:
            h.remove()
        self.model.to(orig_device)

        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Generated: '{text}'")
        return text

    def steer_transfer(
        self,
        zsteer: torch.Tensor,
        prompt: str,
        max_new_tokens: int = 50
    ) -> str:
        print(f"\nTransferring zsteer onto prompt: '{prompt}'")
        zsteer = zsteer.to(self.device)
        handles = []
        if self.inject_embedding:
            handles.append(self.model.model.embed_tokens.register_forward_hook(lambda m, i, out: self._inject(out, zsteer)))
        layers = list(self.model.model.layers) if self.inject_every_layer else [self.model.model.layers[self.inject_layer]]
        for layer in layers:
            handles.append(layer.self_attn.register_forward_hook(
                lambda m, i, out, z=zsteer: (self._inject(out, z) if not isinstance(out, tuple) else (self._inject(out[0], z), *out[1:]))
            ))
            if self.inject_mlp:
                handles.append(layer.mlp.register_forward_hook(lambda m, i, out, z=zsteer: (self._inject(out, z) if not isinstance(out, tuple) else (self._inject(out[0], z), *out[1:]))))

        tokens = self.tokenizer(prompt, return_tensors='pt', return_attention_mask=True, padding=True).to(self.device)
        out_ids = self.model.generate(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        for h in handles:
            h.remove()
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)


def compute_bleu(pred: str, ref: str) -> float:
    return sacrebleu.sentence_bleu(pred, [ref]).score


if __name__ == '__main__':
    model_name = "THGLab/Llama-3.1-8B-SmileyLlama-1.1"
    print(f"Loading model {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    tok.pad_token = tok.eos_token
    layer = 31

    extractor = SteeringVectorExtractor(
        mdl, tok,
        d_prime=None,
        inject_layer=layer,
        inject_timestep='all',
        inject_mlp=True,
        inject_every_layer=False,
        inject_embedding=False,
    )

    df = pd.read_parquet("steering_vectors_all_zinc.parquet")
    
    if 'smile' not in df.columns:
        raise ValueError("The input file must contain a 'smile' column.")

    output_dir = 'steering_vectors_zinc_layer_31_batches'
    os.makedirs(output_dir, exist_ok=True)
    batch_size = 1000
    
    for batch_start in range(0, len(df), batch_size):
        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        
        vectors = []
        smiles_list = []
        affinity_list = []
        sa_list = []
        qed_list = []
        logp_list = []
        scaffold_list = []
        
        for idx, (_, row) in enumerate(batch_df.iterrows()):
            smiles = row['smile']
            global_idx = batch_start + idx
            print(f"Processing SMILES {global_idx+1}/{len(df)}: {smiles}")
            z = extractor.extract_zsteer(smiles, steps=100, lr=0.1)
            
            vectors.append(z.cpu().numpy())
            smiles_list.append(smiles)
            # affinity_list.append(row['affinity'])
            sa_list.append(row['sa'])
            qed_list.append(row['qed'])
            logp_list.append(row['logp'])
            # scaffold_list.append(row.get('scaffold', ''))
        
        # Create batch DataFrame
        batch_result_df = pd.DataFrame({
            'smile': smiles_list,
            'steering_vector': vectors,
            # 'affinity': affinity_list,
            'sa': sa_list,
            'qed': qed_list,
            'logp': logp_list,
            # 'scaffold': scaffold_list
        })
        
        # Save batch to its own file
        batch_num = batch_start // batch_size
        batch_file = os.path.join(output_dir, f'batch_{batch_num:04d}.parquet')
        batch_result_df.to_parquet(batch_file, index=False)
        print(f"  ✓ Saved batch {batch_num} to {batch_file} ({batch_end}/{len(df)} total rows)")
    
    print(f"\n✓ All batches saved to {output_dir}")