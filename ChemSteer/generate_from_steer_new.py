import torch
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
from rdkit.Contrib.SA_Score.sascorer import calculateScore as sa_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
import torch.nn as nn
import sacrebleu
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import gc
import os
import sys
import argparse
import glob 

class SteeringVectorExtractor:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        d_prime: int = None,
        inject_layer: int = 24,
        inject_timestep: str = 'first',
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
        if self.inject_timestep == 'first':
            hidden[:, 0, :] += vec
        else:
            hidden += vec.unsqueeze(0).unsqueeze(1)
        return hidden

    def steer_transfer(
        self,
        zsteer: torch.Tensor,
        prompt: str,
        max_new_tokens: int = 50
    ) -> str:
        """
        Transfer steering vector onto an arbitrary prompt.
        Injection is identical to steer_generate.
        """

        zsteer = zsteer.to(self.device)
        vec = (self.W_steer.T @ zsteer) if self.W_steer is not None else zsteer

        def make_layer_hook():
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    h = out[0]
                    h = h + vec.view(1, 1, -1)
                    return (h,) + out[1:]
                else:
                    return out + vec.view(1, 1, -1)
            return hook


        handle = self.model.model.layers[self.inject_layer].register_forward_hook(make_layer_hook())

        tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True).to(self.device)

        with torch.no_grad():
            out_ids = self.model.generate(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=self.model.config.eos_token_id, 
                use_cache = False
            )

        handle.remove()
        input_len = tokens.input_ids.shape[1]
        gen_tokens = out_ids[0][input_len:]
        gen = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        return gen


    def steer_generate_similar(
        self,
        # zsteer: torch.Tensor,
        orig_smiles: str,
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        top_k: int = 20,
        top_p: float = 0.9,
    ):

        prompt = f"Original SMILES: {orig_smiles}\n Modified SMILES:"

        zsteer = zsteer.to(self.device)

        # Project steering vector if needed
        vec = (self.W_steer.T @ zsteer) if self.W_steer is not None else zsteer
        vec = vec.view(1, 1, -1)

        # Tokenize prompt
        tokens = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)

        prompt_len = tokens.input_ids.shape[1]
        token_counter = {"t": 0}

        def make_layer_hook():
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    h = out[0]
                else:
                    h = out

                # Only steer newly generated tokens
                if token_counter["t"] >= prompt_len:
                    if self.inject_timestep == "first":
                        if token_counter["t"] == prompt_len:
                            h = h + vec
                            
                    elif self.inject_timestep == "all":
                        h = h + vec

                token_counter["t"] += 1

                if isinstance(out, tuple):
                    return (h,) + out[1:]
                return h

            return hook

        handle = self.model.model.layers[self.inject_layer].register_forward_hook(
            make_layer_hook()
        )

        with torch.no_grad():
            out_ids = self.model.generate(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=self.model.config.eos_token_id,
                eos_token_id=self.model.config.eos_token_id,
                use_cache=True,
            )

        handle.remove()

        decoded = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)

        # Extract modified SMILES
        if "Modified SMILES:" in decoded:
            smiles = decoded.split("Modified SMILES:")[-1]
        else:
            smiles = decoded

        # Stop at whitespace/newline
        smiles = smiles.strip().split()[0]
        return smiles

    def generate_similar(
        self,
        orig_smiles: str,
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        top_k: int = 20,
        top_p: float = 0.9,
    ):
        """
        Prompt-based conditional generation without any steering or activation injection.
        """

        prompt = f"Original SMILES: {orig_smiles}\nModified SMILES:"

        # Tokenize prompt
        tokens = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            out_ids = self.model.generate(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=self.model.config.eos_token_id,
                eos_token_id=self.model.config.eos_token_id,
                use_cache=True,
            )

        decoded = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)

        # Extract modified SMILES
        if "Modified SMILES:" in decoded:
            smiles = decoded.split("Modified SMILES:")[-1]
        else:
            smiles = decoded

        # Stop at whitespace/newline
        smiles = smiles.strip().split()[0]
        return smiles

def compute_bleu(pred: str, ref: str) -> float:
    return sacrebleu.sentence_bleu(pred, [ref]).score


# -----------------------------
# CONFIG
# -----------------------------

MODEL_NAME = "THGLab/Llama-3.1-8B-SmileyLlama-1.1"
MAX_NEW_TOKENS = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# PARSE ARGUMENTS
# -----------------------------
parser = argparse.ArgumentParser(description="Steer molecules using a Parquet file.")
parser.add_argument("--parquet_file", type=str, required=True, help="Path to the input Parquet file.")
parser.add_argument("--output_csv", type=str, default="", help="Path to save the output CSV file.")
parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder containing steering vectors.")
args = parser.parse_args()

PARQUET_FILE = args.parquet_file
OUTPUT_CSV = args.output_csv
INPUT_FOLDER = args.input_folder

# -----------------------------
# LOAD MODEL
# -----------------------------
print(f"Loading model {MODEL_NAME} on {DEVICE}")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
tok.pad_token = tok.eos_token

extractor = SteeringVectorExtractor(
    model,
    tok,
    inject_timestep="all",
    inject_mlp=False,
    inject_every_layer=False,
    inject_embedding=False,
    device=DEVICE
)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_parquet(PARQUET_FILE, engine="pyarrow")
print(f"Loaded {len(df)} rows from {PARQUET_FILE}")

# Stack all vectors into matrix
Z = np.stack(df["steering_vector"].values)  # (num_molecules, d_prime)


sa_vec_path = glob.glob(os.path.join(INPUT_FOLDER, "*sa.npy"))[0]
qed_vec_path = glob.glob(os.path.join(INPUT_FOLDER, "*qed.npy"))[0]
logp_vec_path = glob.glob(os.path.join(INPUT_FOLDER, "*logp.npy"))[0]

print(f"Detected SA vector: {sa_vec_path}")
print(f"Detected QED vector: {qed_vec_path}")
print(f"Detected LogP vector: {logp_vec_path}")

MIN_SIZE = 5  # Set the minimum size for generated SMILES



# Set alpha for each steering vector (match rank_molecules.py logic)
ALPHA_SA = -3 # negative to match z - alpha * steer
ALPHA_QED = 3 # set as needed
ALPHA_LOGP = 3  # set as needed

# Load steering vectors
z_sa = np.load(sa_vec_path)
z_sa = z_sa / (np.linalg.norm(z_sa) + 1e-8)
z_sa = ALPHA_SA * z_sa
z_sa_tensor = torch.tensor(z_sa, dtype=torch.float16, device=DEVICE)

z_qed = np.load(qed_vec_path)
z_qed = z_qed / (np.linalg.norm(z_qed) + 1e-8)
z_qed = ALPHA_QED * z_qed
z_qed_tensor = torch.tensor(z_qed, dtype=torch.float16, device=DEVICE)

z_logp = np.load(logp_vec_path)
z_logp = z_logp / (np.linalg.norm(z_logp) + 1e-8)
z_logp = ALPHA_LOGP * z_logp
z_logp_tensor = torch.tensor(z_logp, dtype=torch.float16, device=DEVICE)

print("Steering vectors for SA, QED, and LogP dynamically loaded from input folder.")

# -----------------------------
# GENERATE STEERED MOLECULES
# -----------------------------
results = []

for i, row in df.iterrows():
    orig_smiles = row["smile"]

    # -----------------------------
    # SA-steered
    # -----------------------------
    try:
        gen_sa = extractor.steer_generate_similar(z_sa_tensor, max_new_tokens=MAX_NEW_TOKENS, orig_smiles=orig_smiles)
        # gen_sa = extractor.generate_similar(max_new_tokens=MAX_NEW_TOKENS, orig_smiles=orig_smiles)
        print(f"[{i}] Generated SA-steered SMILES: {gen_sa}, original: {orig_smiles}")
        mol_sa = Chem.MolFromSmiles(gen_sa)
        valid_sa = mol_sa is not None
        sa_new = sa_score(mol_sa) if valid_sa else np.nan
        delta_sa = sa_new - row["sa"] if valid_sa else np.nan
        print(f"[{i}] Original SA: {row['sa']:.4f}, New SA: {sa_new:.4f}, Delta: {delta_sa:.4f}")
    except Exception as e:
        print(f"[{i}] SA generation failed: {e}")
        gen_sa, sa_new, delta_sa, valid_sa = "", np.nan, np.nan, False

    # -----------------------------
    # QED-steered
    # -----------------------------
    try:
        gen_qed = extractor.steer_generate_similar(z_qed_tensor, max_new_tokens=MAX_NEW_TOKENS, orig_smiles=orig_smiles)
        # gen_qed = extractor.generate_similar(max_new_tokens=MAX_NEW_TOKENS, orig_smiles=orig_smiles)
        print(f"[{i}] Generated QED-steered SMILES: {gen_qed}, original: {orig_smiles}")
        mol_qed = Chem.MolFromSmiles(gen_qed)
        valid_qed = mol_qed is not None
        qed_new = QED.qed(mol_qed) if valid_qed else np.nan
        delta_qed = qed_new - row["qed"] if valid_qed else np.nan
        print(f"[{i}] Original QED: {row['qed']:.4f}, New QED: {qed_new:.4f}, Delta: {delta_qed:.4f}")
    except Exception as e:
        print(f"[{i}] QED generation failed: {e}")
        gen_qed, qed_new, delta_qed, valid_qed = "", np.nan, np.nan, False

    # -----------------------------
    # LogP-steered
    # -----------------------------
    # Adjust alpha for steering direction based on original LogP
    # alpha_logp_adjusted = -2 if row["logp"] >= 2.7 else 1
    # z_logp_adjusted = alpha_logp_adjusted * z_logp_tensor
    # z_logp_adjusted = z_logp_tensor

    try:
        gen_logp = extractor.steer_generate_similar(z_logp_tensor, max_new_tokens=MAX_NEW_TOKENS, orig_smiles=orig_smiles)
        # gen_logp = extractor.generate_similar( max_new_tokens=MAX_NEW_TOKENS, orig_smiles=orig_smiles)
        print(f"[{i}] Generated LogP-steered SMILES: {gen_logp}, original: {orig_smiles}")
        mol_logp = Chem.MolFromSmiles(gen_logp)
        valid_logp = mol_logp is not None
        logp_new = Descriptors.MolLogP(mol_logp) if valid_logp else np.nan
        delta_logp = logp_new - row["logp"] if valid_logp else np.nan
        print(f"[{i}] Original LogP: {row['logp']:.4f}, New LogP: {logp_new:.4f}, Delta: {delta_logp:.4f}")
    except Exception as e:
        print(f"[{i}] LogP generation failed: {e}")
        gen_logp, logp_new, delta_logp, valid_logp = "", np.nan, np.nan, False

    results.append({
        "original_smiles": orig_smiles,
        "sa_generated_smiles": gen_sa,
        "sa_valid": valid_sa,
        "sa_new": sa_new,
        "delta_sa": delta_sa,
        "qed_generated_smiles": gen_qed,
        "qed_valid": valid_qed,
        "qed_new": qed_new,
        "delta_qed": delta_qed,
        "logp_generated_smiles": gen_logp,
        "logp_valid": valid_logp,
        "logp_new": logp_new,
        "delta_logp": delta_logp,
    })

    if (i + 1) % 50 == 0:
        print(f"Processed {i+1}/{len(df)} molecules")

# -----------------------------
# SAVE RESULTS
# -----------------------------
out_df = pd.DataFrame(results)
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nResults written to {OUTPUT_CSV}")
