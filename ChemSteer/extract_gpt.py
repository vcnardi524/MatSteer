import os
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import sacrebleu
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from transformers import GPT2TokenizerFast, GPT2LMHeadModel


class SteeringVectorExtractor:
    def __init__(
        self,
        model: GPT2LMHeadModel,
        tokenizer: GPT2TokenizerFast,
        d_prime: int = None,
        inject_layer: int = 6,
        inject_timestep: str = 'all',
        inject_mlp: bool = True,
        inject_every_layer: bool = False,
        inject_embedding: bool = False,
        device: torch.device = None
    ):
       
        self.device = device or (
            torch.device('cuda') if torch.cuda.is_available()
            else torch.device('mps') if torch.backends.mps.is_available()
            else torch.device('cpu')
        )
        print(f"Using device: {self.device}")

        # Load and freeze model
        self.model = model.to(self.device)
        self.model.eval()
        self.model.config.pad_token_id = self.model.config.eos_token_id
        for p in self.model.parameters(): p.requires_grad = False

        # Tokenizer padding
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Settings
        self.d = self.model.config.n_embd
        self.d_prime = d_prime or self.d
        self.inject_layer = inject_layer
        self.inject_timestep = inject_timestep
        self.inject_mlp = inject_mlp
        self.inject_every_layer = inject_every_layer
        self.inject_embedding = inject_embedding

        # Projection if needed
        if self.d_prime < self.d:
            print(f"Creating projection: {self.d_prime}→{self.d}")
            W = torch.randn(self.d_prime, self.d, device=self.device)
            q, _ = torch.linalg.qr(W.t())
            self.W_steer = q.t().detach()
        else:
            self.W_steer = None

    def _inject(self, hidden: torch.Tensor, zsteer: torch.Tensor) -> torch.Tensor:
        """Inject steering vector into hidden states"""
        vec = self.W_steer.t() @ zsteer if self.W_steer is not None else zsteer
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
        # reg_lambda: float = 0.0
    ) -> torch.Tensor:
       
           # Check SMILES length and token count
        smiles_length = len(sentence)
        token_count = len(self.tokenizer.tokenize(sentence))
        # print(f"SMILES length: {smiles_length}, Token count: {token_count}")

         # Skip if too long
        if token_count > self.tokenizer.model_max_length:
            print(f"Skipping SMILES string as it exceeds maximum token length ({self.tokenizer.model_max_length}).")
            return None

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
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode='min', factor=0.9, patience=2
        # )

        handles = []
        if self.inject_embedding:
            print("Injecting at embedding layer.")
            def embed_hook(module, inp, out): return self._inject(out, zsteer)
            handles.append(self.model.transformer.wte.register_forward_hook(embed_hook))

        layers = (
            list(self.model.transformer.h) if self.inject_every_layer
            else [self.model.transformer.h[self.inject_layer]]
        )
        # print(f"Injecting into {'ALL' if self.inject_every_layer else 'layer '+str(self.inject_layer)}.")
        for layer in layers:
            def attn_hook(module, inp, out, zsteer=zsteer):
                if isinstance(out, tuple):
                    attn_out, *rest = out
                    return (self._inject(attn_out, zsteer), *rest)
                return self._inject(out, zsteer)
            handles.append(layer.attn.register_forward_hook(attn_hook))
            if self.inject_mlp:
                def mlp_hook(module, inp, out, zsteer=zsteer): return self._inject(out, zsteer)
                handles.append(layer.mlp.register_forward_hook(mlp_hook))

        for i in range(1, steps+1):
            optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            loss = nn.CrossEntropyLoss()(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
            # if reg_lambda > 0: loss = loss + reg_lambda * torch.sum(zsteer ** 2)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_([zsteer], max_norm=5.0)
            optimizer.step()
            # scheduler.step(loss)
            # if i % max(1, steps//10) == 0 or i == 1:
            #     print(f"  Step {i}/{steps}  Loss: {loss.item():.4f}  Norm: {zsteer.norm().item():.2f}  LR: {optimizer.param_groups[0]['lr']:.4f}")

        for h in handles: h.remove()
        z_cpu = zsteer.detach().cpu()
        # print(f"Extracted zsteer first10: {z_cpu.numpy()[:10]}")
        return z_cpu

    def steer_generate(
        self,
        zsteer: torch.Tensor,
        max_new_tokens: int = 20
    ) -> str:
       
        print(f"\n[generate] norm(zsteer)={zsteer.norm().item():.2f}")
        # Move model and inputs to CPU for generation
        orig_device = next(self.model.parameters()).device
        self.model.to('cpu')
        z_cpu = zsteer.cpu()
        input_ids = torch.tensor([[self.tokenizer.bos_token_id]])
        attention_mask = torch.ones_like(input_ids)

        # Register hooks on CPU model
        handles = []
        if self.inject_embedding:
            def embed_hook(module, inp, out): return self._inject(out.to(orig_device), z_cpu.to(orig_device)).to('cpu')
            handles.append(self.model.transformer.wte.register_forward_hook(embed_hook))

        layers = (list(self.model.transformer.h) if self.inject_every_layer else [self.model.transformer.h[self.inject_layer]])
        for layer in layers:
            def attn_hook(module, inp, out, z_cpu=z_cpu):
                if isinstance(out, tuple):
                    attn_out, *rest = out
                    steered = self._inject(attn_out.to(orig_device), z_cpu.to(orig_device)).to('cpu')
                    return (steered, *rest)
                return self._inject(out.to(orig_device), z_cpu.to(orig_device)).to('cpu')
            handles.append(layer.attn.register_forward_hook(attn_hook))
            if self.inject_mlp:
                def mlp_hook(module, inp, out, z_cpu=z_cpu): return self._inject(out.to(orig_device), z_cpu.to(orig_device)).to('cpu')
                handles.append(layer.mlp.register_forward_hook(mlp_hook))

        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=self.model.config.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            # no_repeat_ngram_size=2,
            # repetition_penalty=1.2
        )

        for h in handles: h.remove()
        self.model.to(orig_device)

        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Generated: '{text}'")
        return text

    def steer_transfer(self,
        zsteer: torch.Tensor,
        prompt: str,
        max_new_tokens: int = 50
    ) -> str:
        print(f"\nTransferring zsteer onto prompt: '{prompt}'")
        zsteer = zsteer.to(self.device)
        handles = []
        if self.inject_embedding:
            handles.append(self.model.transformer.wte.register_forward_hook(lambda m, i, out: self._inject(out, zsteer)))
        layers = list(self.model.transformer.h) if self.inject_every_layer else [self.model.transformer.h[self.inject_layer]]
        for layer in layers:
            handles.append(layer.attn.register_forward_hook(
                lambda m, i, out, z=zsteer: (self._inject(out[0], z), *out[1:]) if isinstance(out, tuple) else self._inject(out, z)
            ))
            if self.inject_mlp:
                handles.append(layer.mlp.register_forward_hook(lambda m, i, out: self._inject(out, zsteer)))

        tokens = self.tokenizer(prompt, return_tensors='pt', return_attention_mask=True, padding=True).to(self.device)
        out_ids = self.model.generate(
            tokens.input_ids,
            attention_mask=tokens.attention_mask,
            # pad_token_id=self.model.config.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        for h in handles: h.remove()
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)

    
    # def steer_offset_transfer(
    #     self,
    #     target: str,
    #     prompt: str,
    #     steps: int = 500,
    #     lr: float = 1.0
    # ) -> str:
    #    
    #     # 1) Extract z_target
    #     z_target = self.extract_zsteer(target, steps=steps, lr=lr)
    #     # 2) Extract z_prompt
    #     z_prompt = self.extract_zsteer(prompt, steps=steps, lr=lr)
    #     # 3) Compute offset
    #     z_offset = z_target - z_prompt
    #     # 4) Apply offset directly on prompt
    #     tgt_len = self.tokenizer(target, return_tensors='pt').input_ids.size(1)
    #     steered = self.steer_transfer(z_offset, prompt, max_new_tokens=tgt_len)
    #     print(f"Steered result: '{steered}'")
    #     return steered



def compute_bleu(pred: str, ref: str) -> float:
    return sacrebleu.sentence_bleu(pred, [ref]).score

if __name__ == '__main__':


    # checkpoint = 'unikei/bert-base-smiles'
    # tok = BertTokenizerFast.from_pretrained(checkpoint)
    # mdl = BertModel.from_pretrained(checkpoint)

    tok = GPT2TokenizerFast.from_pretrained("entropy/gpt2_zinc_87m", max_len=256)
    mdl = GPT2LMHeadModel.from_pretrained('entropy/gpt2_zinc_87m')

    # model_name = 'gpt2'
    print(f"Loading smiles model")
    # )
    
    extractor = SteeringVectorExtractor(
        mdl, tok,
        d_prime=None,
        inject_timestep='all',
        inject_mlp=True,
        inject_every_layer=False,
        inject_embedding=False,
    )

    # z = extractor.extract_zsteer(tgt, steps=500, lr=0.01)
    # rec = extractor.steer_generate(z, max_new_tokens=sentence_len)
    # print(f"BLEU (all+embed, lr=1.0): {compute_bleu(rec, tgt):.2f}")

    # # target = "The quick brown fox jumps over the lazy dog."
    # # prompt = "Hello world!"
    df = pd.read_parquet("steering_vectors_all_chembl.parquet")

    
    if 'smile' not in df.columns:
        raise ValueError("The input file must contain a 'smile' column.")

    output_dir = 'steering_vectors_chembl_gpt2_batches'
    os.makedirs(output_dir, exist_ok=True)
    batch_size = 1000
    
    for batch_start in range(0, len(df), batch_size):
        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        
        vectors = []
        smiles_list = []
        # affinity_list = []
        sa_list = []
        qed_list = []
        logp_list = []
        # scaffold_list = []
        
        for idx, (_, row) in enumerate(batch_df.iterrows()):
            smiles = row['smile']
            global_idx = batch_start + idx
            print(f"Processing SMILES {global_idx+1}/{len(df)}: {smiles}")
            z = extractor.extract_zsteer(smiles, steps=100, lr=0.1)
            
            if z is None:
                print(f"Skipping SMILES {global_idx+1} as no steering vector was generated.")
                continue

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
