import torch
import argparse
import random
import numpy as np
from torch.optim import AdamW
from transformers import AutoTokenizer, get_scheduler
import string
import os
from utils import DataLoader, Batchify, now_time, MSRVTT_CATEGORIES
from module import msLoRA_Concat, msLoRA
from sklearn.metrics import classification_report, accuracy_score
from pycocoevalcap.cider.cider import Cider

parser = argparse.ArgumentParser()
parser.add_argument('-task', type=str, default='twitter17', choices=['mvsa', 'hateful', 'scienceqa', 'twitter17', 'twitter15', 'flickr', 'msrvtt'], help='Task name')
parser.add_argument('-llm_model', type=str, default="./llm/Qwen2.5-7B")
parser.add_argument('-clip_model', type=str, default="./llm/clip-vit-base-patch32/")
parser.add_argument('-wav2vec_path', type=str, default="./llm/wav2vec2-base-960h/")
parser.add_argument('-lr', type=float, default=2e-5)
parser.add_argument('-epochs', type=int, default=10)
parser.add_argument('-batch_size', type=int, default=16)
parser.add_argument('-r', type=int, default=16)
parser.add_argument('-lora_modules', type=int, default=7)
parser.add_argument('-multimodal_scaling', type=int, default=4)
parser.add_argument('-clip_norm', '--clip_norm', type=float, default=1.0, help='gradient clipping')
parser.add_argument('-noise_std', type=float, default=0.0, help='Standard deviation of Gaussian noise for images')
parser.add_argument('-gpu', type=str, default='0,1', help='GPU ID to use')
parser.add_argument('-lora_name', type=str, default='LoRA', help='LoRA name')
parser.add_argument('-save_path', type=str, default='../autodl-tmp/', help='Path for trained models.')
parser.add_argument('-seed', type=int, default=42)
args = parser.parse_args()

print(f"{now_time()} Parameters:\n")
for key, value in vars(args).items():
    print(f"{key}: {value}")

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if args.task == 'hateful':
    data_path = './datasets/HatefulMemes/'
elif args.task == 'scienceqa':
    data_path = './datasets/ScienceQA/'
elif args.task =='twitter17':
    data_path = './datasets/Twitter17/'
elif args.task =='twitter15':
    data_path = './datasets/Twitter15/'
elif args.task == 'flickr':
    data_path = './datasets/flickr_8k/'
elif args.task =='msrvtt':
    data_path = './datasets/MSR-VTT/'
else: # mvsa
    data_path = './datasets/MVSA_Single/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(args.seed)

def evaluate(data_loader):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for _ in range(data_loader.total_step):
            img_ids, aud_ids, input_ids, mask, _, labels = data_loader.next_batch(mode='eval')
            model.set_multimodal_features(img_ids, aud_ids)
            
            max_new_tokens = 30 if args.task in ['flickr', 'msrvtt'] else 5
            gen_ids = model.model.generate(
                input_ids=input_ids.to(device),
                attention_mask=mask.to(device),
                max_new_tokens=max_new_tokens, 
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
            preds = tokenizer.batch_decode(gen_ids[:, input_ids.size(1):], skip_special_tokens=True)
            for p, l in zip(preds, labels):
                p_lower = p.lower().strip()
                pred_label = -1
                
                if args.task in ['flickr', 'msrvtt']:
                    all_preds.append(p_lower)
                    all_labels.append(l) # l is string for flickr and msrvtt
                    continue
                
                if args.task == 'hateful':
                    # Hateful Memes: yes -> 1, no -> 0
                    if p_lower.startswith('yes'): pred_label = 1
                    elif p_lower.startswith('no'): pred_label = 0
                
                elif args.task == 'scienceqa':
                    # ScienceQA: Extract the first generated character A, B, C, D...
                    if len(p_lower) > 0:
                        first_char = p_lower[0].upper()
                        if 'A' <= first_char <= 'Z':
                            pred_label = ord(first_char) - ord('A')
                
                else: # mvsa, twitter17, twitter15
                    if "positive" in p_lower: pred_label = 1
                    elif "negative" in p_lower: pred_label = 2
                    elif "neutral" in p_lower: pred_label = 0

                all_preds.append(pred_label)
                all_labels.append(l.item())
                
    if args.task in ['flickr', 'msrvtt']:
        gts = {}
        res = {}
        for i, (l, p) in enumerate(zip(all_labels, all_preds)):
            if isinstance(l, list):
                gts[i] = [str(ref) for ref in l]
            else:
                gts[i] = [str(l)]
            res[i] = [str(p)]
        cider_scorer = Cider()
        score, _ = cider_scorer.compute_score(gts, res)
    else:
        score = accuracy_score(all_labels, all_preds)
    return score, all_preds, all_labels

# Initialization
model_type = "qwen" if "qwen" in args.llm_model.lower() else "llama"
print(f"{now_time()} Detected Model Type: {model_type}")

tokenizer = AutoTokenizer.from_pretrained(args.llm_model, trust_remote_code=True)

if model_type == "qwen":
    if tokenizer.pad_token is None:
        if "<|extra_0|>" in tokenizer.get_vocab():
            tokenizer.pad_token = "<|extra_0|>"
        else:
            tokenizer.pad_token = tokenizer.eos_token
else:
    # Llama 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

# Unified setting: must be left-padded during generation
tokenizer.padding_side = 'left'

# 2. Dynamically obtain the hidden layer dimension (hidden_size) of the model
#  Qwen2.5-3B: 2048, Qwen2.5-7B: 3584, Llama2-7B: 4096
from transformers import AutoConfig
config = AutoConfig.from_pretrained(args.llm_model, trust_remote_code=True)
llm_hidden_size = getattr(config, "hidden_size", getattr(config, "d_model", None))
print(f"{now_time()} LLM Hidden Size: {llm_hidden_size}")

corpus = DataLoader(data_path, tokenizer, args.clip_model, device, wav2vec_path=args.wav2vec_path, task=args.task, noise_std=args.noise_std)
train_loader = Batchify(corpus.train, tokenizer, args.batch_size, task=args.task, shuffle=True)
valid_loader = Batchify(corpus.valid, tokenizer, args.batch_size, task=args.task)

# 3. Initialize msLoRA
model = msLoRA(
    args.llm_model, 
    args.r, 
    args.lora_modules, 
    corpus.image_embeddings,
    audio_embeddings=corpus.audio_embeddings,
    multimodal_scaling=args.multimodal_scaling
    # hidden_size=llm_hidden_size
)

#model = msLoRA_Concat(args.llm_model, args.r, args.lora_modules, corpus.image_embeddings)
optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

num_training_steps = args.epochs * train_loader.total_step
lr_scheduler = get_scheduler(
    name="cosine", optimizer=optimizer, num_warmup_steps=int(num_training_steps * 0.1), num_training_steps=num_training_steps
)

# --- Early Stopping Initialization ---
best_acc = 0.0
patience = 4
patience_counter = 0
best_model_path = args.save_path + 'best_model_{args.task}.pt'

# Training
print(f"{now_time()} Starting training for task: {args.task}")
for epoch in range(args.epochs):
    model.train()
    for step in range(train_loader.total_step):
        img_ids, aud_ids, input_ids, mask, t_lens, _ = train_loader.next_batch(mode='train')
        outputs = model(input_ids.to(device), mask.to(device), img_ids, aud_ids, target_lens=t_lens)
        
        loss = outputs.loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        if step % 50 == 0:
            print(f"{now_time()} Epoch {epoch} Step {step}/{train_loader.total_step} Loss: {loss.item():.4f}")
            
    acc, _, _ = evaluate(valid_loader)
    print(f"{now_time()} Epoch {epoch} Validation Accuracy: {acc:.4f}")

    # --- Early Stopping Logic ---
    if acc > best_acc:
        best_acc = acc
        patience_counter = 0
        print(f"{now_time()} New best validation accuracy: {best_acc:.4f}. Saving model...")
        torch.save(model.state_dict(), best_model_path)
    else:
        patience_counter += 1
        print(f"{now_time()} No improvement. Patience: {patience_counter}/{patience}")

    if patience_counter >= patience:
        print(f"{now_time()} Early stopping.")
        break

# Testing
del optimizer
del lr_scheduler
torch.cuda.empty_cache()

print(f"\n{now_time()}Loading best model from {best_model_path} for final evaluation...")
model.load_state_dict(torch.load(best_model_path))
test_loader = Batchify(corpus.test, tokenizer, args.batch_size, task=args.task)
acc, preds, labels = evaluate(test_loader)

# --- Set the Report parameters according to the task ---
if args.task == 'hateful':
    t_names = ["non-hateful", "hateful"]
    target_ids = [0, 1]
elif args.task == 'scienceqa':
    max_label = max(labels) if labels else 0
    target_ids = list(range(max_label + 1))
    t_names = [f"Option {chr(65+i)}" for i in target_ids]
elif args.task in ['flickr', 'msrvtt']:
    t_names = []
    target_ids = []
else: # mvsa, twitter17, twitter15
    t_names = ["neutral", "positive", "negative"]
    target_ids = [0, 1, 2]

metric_name = "CIDEr" if args.task in ['flickr', 'msrvtt'] else "Accuracy"
print(f"{now_time()} Test {metric_name}: {acc:.4f}")
print("\nFinal Test Report:")
if args.task not in ['flickr', 'msrvtt']:
    print(classification_report(labels, preds, labels=target_ids, target_names=t_names, digits=4, zero_division=0))