import os
from enum import Enum
from pathlib import Path
from typing import Optional

import re

import clip
import torch
import torch.nn.functional as nnf
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel

BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp"
GPT2_LOCAL_DIR = BASE_DIR / "gpt2_local"
MODEL_PATH = BASE_DIR / "ClipCap" / "transformer_weights.pt"
CLIP_MODEL_NAME = "RN50x4"
PREFIX_LENGTH = 40
CLIP_LENGTH = 40
PREFIX_SIZE = 640
NUM_LAYERS = 8

app = FastAPI(title="ClipCap Caption API")
TEMP_DIR.mkdir(exist_ok=True)

clip_model = None
preprocess = None
tokenizer = None
model = None
device = None
gpt2_model = None


class MappingType(Enum):
    MLP = "mlp"
    Transformer = "transformer"


class MLP(torch.nn.Module):
    def __init__(self, sizes, bias=True, act=torch.nn.Tanh):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(torch.nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MlpTransformer(torch.nn.Module):

    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=torch.nn.ReLU, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = torch.nn.Linear(in_dim, h_dim)
        self.act = act()
        self.fc2 = torch.nn.Linear(h_dim, out_d)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = torch.nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = torch.nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = torch.nn.Linear(dim_self, dim_self)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, _ = y.shape
        queries = self.to_queries(x).view(b, n, self.num_heads, c // self.num_heads)
        keys_values = self.to_keys_values(y).view(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum("bnhd,bmhd->bnmh", queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum("bnmh,bmhd->bnhd", attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(torch.nn.Module):
    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4.0, bias=False, dropout=0.0,
                 act=torch.nn.ReLU, norm_layer=torch.nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), dim_self, act=act, dropout=dropout)

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(torch.nn.Module):
    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2.0, act=torch.nn.ReLU, norm_layer=torch.nn.LayerNorm, enc_dec: bool = False):
        super().__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act,
                                              norm_layer=norm_layer))
            elif enc_dec:
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act,
                                              norm_layer=norm_layer))
            else:
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act,
                                              norm_layer=norm_layer))
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec:
                x = layer(x, y)
            elif self.enc_dec:
                x = layer(x, x, mask)
            else:
                x = layer(x, y, mask)
        return x


class TransformerMapper(torch.nn.Module):
    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int,
                 num_layers: int = 8):
        super().__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = torch.nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = torch.nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out


class ClipCaptionModel(torch.nn.Module):
    def __init__(self, prefix_length: int, clip_length: Optional[int] = None,
                 prefix_size: int = 512, num_layers: int = 8,
                 mapping_type: MappingType = MappingType.MLP):
        super().__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel(GPT2Config())
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if mapping_type == MappingType.MLP:
            self.clip_project = MLP((prefix_size,
                                     (self.gpt_embedding_size * prefix_length) // 2,
                                     self.gpt_embedding_size * prefix_length))
        else:
            self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size,
                                                  prefix_length, clip_length, num_layers)

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out


def load_gpt2_tokenizer() -> GPT2Tokenizer:
    if not GPT2_LOCAL_DIR.exists():
        raise FileNotFoundError(f"Local GPT-2 directory not found at {GPT2_LOCAL_DIR}")
    return GPT2Tokenizer.from_pretrained(str(GPT2_LOCAL_DIR))


def get_stop_token_id(tokenizer: GPT2Tokenizer, stop_token: str = ".") -> int:
    token_ids = tokenizer.encode(stop_token, add_special_tokens=False)
    if token_ids:
        return token_ids[-1]
    return 13


def generate2(
        model: torch.nn.Module,
        tokenizer: GPT2Tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67,
        top_p=0.8,
        temperature=1.0,
        stop_token: str = '.',
):
    model.eval()
    generated_list = []
    stop_token_index = get_stop_token_id(tokenizer, stop_token)
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():
        for _ in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)

            for _ in range(entry_length):
                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]


def load_model_weights(model: torch.nn.Module, checkpoint: dict) -> torch.nn.Module:
    expected = model.state_dict()
    compatible = {k: v for k, v in checkpoint.items() if k in expected}
    missing = sorted(set(expected.keys()) - set(compatible.keys()))
    if missing:
        raise RuntimeError(f"Checkpoint is missing {len(missing)} required keys, e.g. {missing[:3]}")
    model.load_state_dict(compatible, strict=True)
    return model


def setup_model() -> None:
    global clip_model, preprocess, tokenizer, model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")

    clip_model, preprocess = clip.load(CLIP_MODEL_NAME, device=device, jit=False)
    tokenizer = load_gpt2_tokenizer()

    # Load a standalone GPT-2 model once for story generation. If GPU memory
    # is constrained, this can be replaced by reusing `model.gpt` from the
    # ClipCaptionModel instance, but we load separately for clarity.
    global gpt2_model
    if not GPT2_LOCAL_DIR.exists():
        raise FileNotFoundError(f"Local GPT-2 directory not found at {GPT2_LOCAL_DIR}")
    gpt2_model = GPT2LMHeadModel.from_pretrained(str(GPT2_LOCAL_DIR))
    gpt2_model = gpt2_model.eval().to(device)

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model = ClipCaptionModel(
        PREFIX_LENGTH,
        clip_length=CLIP_LENGTH,
        prefix_size=PREFIX_SIZE,
        num_layers=NUM_LAYERS,
        mapping_type=MappingType.Transformer,
    )
    load_model_weights(model, checkpoint)
    model = model.eval().to(device)


def _generate_caption(image_path: str, entry_length: int = 40) -> str:
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prefix = clip_model.encode_image(image_tensor).to(device, dtype=torch.float32)
        prefix = prefix / prefix.norm(2, -1).item()
        prefix_embed = model.clip_project(prefix).reshape(1, PREFIX_LENGTH, -1)
        caption_text = generate2(
            model,
            tokenizer,
            embed=prefix_embed,
            entry_length=entry_length,
            temperature=1.0,
            top_p=0.95,
            stop_token=".",
        ).strip()

    if caption_text and caption_text[-1] not in ".!?":
        caption_text = caption_text.rstrip() + "."
    return caption_text


def generate_caption(image_path: str) -> str:
    return _generate_caption(image_path, entry_length=40)


def generate_caption_long(image_path: str) -> str:
    return _generate_caption(image_path, entry_length=80)


def generate_story_from_caption(caption: str, max_new_tokens: int = 100) -> str:
    """Generate a short coherent story about the caption.

    Prompt is encoded with the local tokenizer; the generated continuation
    is decoded, the prompt removed, post-processed, and returned.
    """
    global gpt2_model, tokenizer, device
    if gpt2_model is None or tokenizer is None:
        raise RuntimeError("GPT-2 model/tokenizer not loaded")

    caption = (caption or "").strip()
    if not caption:
        caption = "a scene"

    prompt = f"Write a short, coherent story about: {caption}\n\nStory:"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    gen_kwargs = dict(
        do_sample=True,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        attention_mask=attention_mask,
    )

    with torch.no_grad():
        output = gpt2_model.generate(input_ids=input_ids, **gen_kwargs)

    # Decode full output and normalize whitespace
    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    full_text = full_text.replace("\n", " ").strip()

    # Robustly remove the prompt and any 'Story:' label even if whitespace/newlines differ
    # e.g. "Write a short, coherent story about: <caption>  Story:"
    story = re.sub(r'(?i)^\s*Write a short, coherent story about:.*?Story:\s*', '', full_text, count=1)
    # If the above didn't match (no 'Story:'), also remove the leading prompt phrase alone
    if story == full_text:
        story = re.sub(r'(?i)^\s*Write a short, coherent story about:\s*', '', full_text, count=1).strip()

    # Remove any leading 'Story:' or similar label (case-insensitive)
    story = re.sub(r'^["\'\s]*story\s*[:\-]\s*', '', story, flags=re.IGNORECASE).strip()

    # Cleanup escaped quotes and extra spaces
    story = story.replace('\\"', '"').replace("\\'", "'").strip()

    # Ensure ends with sentence-ending punctuation (., !, ?)
    if story and story[-1] not in ('.', '!', '?'):
        last_end = max(story.rfind(p) for p in ('.', '!', '?'))
        if last_end != -1:
            story = story[: last_end + 1].strip()
        else:
            story = story.rstrip() + '.'

    return story


@app.on_event("startup")
def on_startup() -> None:
    setup_model()


@app.post("/caption")
async def caption(image: UploadFile = File(...)) -> dict:
    if image.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    safe_name = Path(image.filename).stem
    suffix = Path(image.filename).suffix or ".jpg"
    temp_path = TEMP_DIR / f"{safe_name}-{os.urandom(8).hex()}{suffix}"
    try:
        with temp_path.open("wb") as f:
            f.write(await image.read())

        caption_text = generate_caption(str(temp_path))
        return {"caption": caption_text}
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.post("/story")
async def story(image: UploadFile = File(...)) -> dict:
    if image.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    safe_name = Path(image.filename).stem
    suffix = Path(image.filename).suffix or ".jpg"
    temp_path = TEMP_DIR / f"{safe_name}-{os.urandom(8).hex()}{suffix}"
    try:
        with temp_path.open("wb") as f:
            f.write(await image.read())

        # First-stage: generate a longer caption for better story context
        caption_text = generate_caption_long(str(temp_path))

        # Second-stage: expand caption into a story
        story_text = generate_story_from_caption(caption_text)
        return {"story": story_text}
    finally:
        if temp_path.exists():
            temp_path.unlink()
