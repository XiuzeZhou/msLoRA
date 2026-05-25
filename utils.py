import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC
import cv2  # opencv-python
import librosa
import math
import random
import json
import traceback
import subprocess
import tempfile
import shutil

def now_time():
    from datetime import datetime
    return f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "

def extract_audio_robust(video_path, duration=20.0, sr=16000):
    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
    try:
        ffmpeg_cmd = shutil.which('ffmpeg')
        if not ffmpeg_cmd:
            try:
                import imageio_ffmpeg
                ffmpeg_cmd = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                ffmpeg_cmd = 'ffmpeg'
                
        cmd = [
            ffmpeg_cmd, '-y', '-i', video_path, '-t', str(duration),
            '-vn', '-acodec', 'pcm_s16le', '-ar', str(sr), '-ac', '1',
            temp_wav
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, shell=True)
        y, sr_out = librosa.load(temp_wav, sr=sr)
        return y, sr_out
    except subprocess.CalledProcessError:
        # Silent Video
        return np.array([]), sr
    finally:
        if os.path.exists(temp_wav):
            try: os.remove(temp_wav)
            except: pass

# MSR-VTT lebels: 20
MSRVTT_CATEGORIES = [
    'music', 'people', 'gaming', 'sports', 'news', 'education', 'tv shows', 
    'movie', 'animation', 'vehicles', 'how-to', 'travel', 'science', 'animals', 
    'kids', 'documentary', 'food', 'cooking', 'beauty', 'ads'
]
cat_to_id = {cat: i for i, cat in enumerate(MSRVTT_CATEGORIES)}
id_to_cat = {i: cat for i, cat in enumerate(MSRVTT_CATEGORIES)}

class DataLoader:
    def __init__(self, data_dir, tokenizer, clip_path, device, wav2vec_path, task='mvsa', noise_std=0.0):
        self.tokenizer = tokenizer
        self.device = device
        self.task = task
        self.noise_std = noise_std
        self.label_map = {'neutral': 0, 'positive': 1, 'negative': 2}
        
        # 1. loading data
        if task == 'mvsa':
            df = self._load_mvsa_df(data_dir)
            self.id_to_label = {0: "neutral", 1: "positive", 2: "negative"}
        elif task == 'scienceqa':
            df = self._load_scienceqa_df(data_dir)
        elif task == 'twitter17':
            df = self._load_twitter17_df(data_dir)
        elif task == 'flickr':
            df = self._load_flickr8k_df(data_dir)
        elif task == 'msrvtt':
            df = self._load_msrvtt_df(data_dir)
        else: # hateful
            df = self._load_hateful_memes_df(data_dir)
            self.id_to_label = {0: "no", 1: "yes"}

        # 2. Automatically extract CLIP image features
        self.image_embeddings = self._generate_visual_embeddings(df, clip_path, data_dir)
        if task == 'msrvtt':
            self.audio_embeddings = self._generate_audio_embeddings(df, wav2vec_path, data_dir)
            self.asr_texts = self._generate_asr_texts(df, wav2vec_path, data_dir)
        else:
            self.audio_embeddings = None
            self.asr_texts = {}
        
        # 3. split dataset
        self.train, self.valid, self.test = self._split_data(df)

    def _load_flickr8k_df(self, data_dir):
        caption_file = os.path.join(data_dir, 'captions.txt')
        df = pd.read_csv(caption_file)
        df.rename(columns={'image': 'image_id', 'caption': 'text'}, inplace=True)
        df['image_path'] = df['image_id'].apply(lambda x: os.path.join(data_dir, 'images', x))
        df['label'] = df['text']
        return df

    def _load_hateful_memes_df(self, data_dir):
        jsonl_path = os.path.join(data_dir, 'train.jsonl')
        data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        df = pd.DataFrame(data)
        
        df['image_id'] = df['id'].astype(str)
        df['image_path'] = df['img'].apply(lambda x: os.path.join(data_dir, x))
        return df

    def _load_scienceqa_df(self, data_dir):
        json_path = os.path.join(data_dir, 'problems.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            problems = json.load(f)
        
        rows = []
        for pid, prob in problems.items():
            if prob['image'] is not None:
                choices = prob['choices']
                options = "\n".join([f"({chr(65+i)}) {c}" for i, c in enumerate(choices)])
                full_text = f"Context: {prob['hint']}\nQuestion: {prob['question']}\nOptions:\n{options}"
                
                rows.append({
                    'image_id': str(pid),
                    'image_path': os.path.join(data_dir, 'train', pid, 'image.png'),
                    'text': full_text,
                    'answer': prob['answer'] # index 0, 1, 2...
                })
        return pd.DataFrame(rows)

    def _load_twitter17_df(self, data_dir):
        """
        data_dir: data path (train.tsv, dev.tsv, test.tsv)
        """
        all_rows = []
        image_root = os.path.join(os.path.dirname(data_dir), 'twitter2017_images')
        tex_root = os.path.join(os.path.dirname(data_dir), 'twitter2017')
        
        splits = {
            'train.tsv': 'train',
            'dev.tsv': 'val',
            'test.tsv': 'test'
        }

        for file_name, split_name in splits.items():
            file_path = os.path.join(tex_root, file_name)
            if not os.path.exists(file_path):
                continue
            
            try:
                # quoting=3 (csv.QUOTE_NONE) Prevent the use of quotation marks in tweets from causing reading errors
                df_split = pd.read_csv(file_path, sep='\t', quoting=3)
                
                for _, row in df_split.iterrows():
                    if split_name in ['train', 'val']:
                        # GitHub: count, label, imgid, text, aspect
                        label = int(row.iloc[1])      # #1 Label
                        img_id_raw = str(row.iloc[2]) # #2 ImageID
                        text = str(row.iloc[3])       # #3 String (Sentence)
                        aspect = str(row.iloc[4])     # #3 String (Aspect)
                    else:
                        label = int(row.iloc[0])
                        img_id_raw = str(row.iloc[1]) # #1 ImageID
                        text = str(row.iloc[2])       # #2 String
                        aspect = str(row.iloc[3])     # #2 String

                    if not img_id_raw.endswith('.jpg'):
                        img_id_full = img_id_raw + ".jpg"
                    else:
                        img_id_full = img_id_raw
                    
                    img_path = os.path.join(image_root, img_id_full)

                    if os.path.exists(img_path):
                        all_rows.append({
                            'image_id': img_id_raw.replace('.jpg', ''),
                            'image_path': img_path,
                            'text': text,
                            'aspect': aspect,
                            'label': label,
                            'split_hint': split_name
                        })
            except Exception as e:
                print(f"Error loading {file_name}: {e}")

        return pd.DataFrame(all_rows)
                    
    
    def _load_mvsa_df(self, data_dir):
        label_file = os.path.join(data_dir, 'labelResultAll.txt')
        df = pd.read_csv(label_file, sep='\t', header=0, names=['image_id', 'annotation'])
        
        def get_sentiment(anno):
            try:
                t, i = anno.split(',')
                t, i = t.strip(), i.strip()
                if t == i: return t
                elif t == 'neutral': return i
                elif i == 'neutral': return t
                else: return 'neutral'
            except: return 'neutral'

        df['sentiment'] = df['annotation'].apply(get_sentiment)
        df['image_path'] = df['image_id'].apply(lambda x: os.path.join(data_dir, 'data', f"{x}.jpg"))
        df['text_path'] = df['image_id'].apply(lambda x: os.path.join(data_dir, 'data', f"{x}.txt"))
        
        df = df[df['image_path'].map(os.path.exists)].reset_index(drop=True)
        return df

    
    def _load_msrvtt_df(self, data_dir):
        train_val_path = os.path.join(data_dir, 'msrvtt_train_9k.json')
        test_path = os.path.join(data_dir, 'msrvtt_test_1k.json')
        video_dir = os.path.join(data_dir, 'videos')

        with open(train_val_path, 'r') as f:
            train_val_data = json.load(f)
        with open(test_path, 'r') as f:
            test_data = json.load(f)

        random.seed(1111)
        random.shuffle(train_val_data)
        n = len(train_val_data)
        train_raw = train_val_data[:int(n * 0.8888)] # About 8000 videos
        valid_raw = train_val_data[int(n * 0.8888):] # About 1000 videos

        all_rows = []
        def process_split(raw_data, split_name):
            for item in raw_data:
                captions = item.get('caption', "")
                
                if split_name == 'train' and isinstance(captions, list):
                    for cap in captions:
                        new_item = item.copy()
                        new_item['caption'] = str(cap).strip()
                        new_item['split'] = split_name
                        new_item['video_path'] = os.path.join(video_dir, f"{item['video_id']}.mp4")
                        all_rows.append(new_item)
                else:
                    new_item = item.copy()
                    if isinstance(captions, list) and len(captions) > 0:
                        new_item['caption'] = str(captions[0]).strip()
                    else:
                        new_item['caption'] = str(captions).strip()
                    new_item['split'] = split_name
                    new_item['video_path'] = os.path.join(video_dir, f"{item['video_id']}.mp4")
                    all_rows.append(new_item)

        process_split(train_raw, 'train')
        process_split(valid_raw, 'valid')
        process_split(test_data, 'test')
            
        return pd.DataFrame(all_rows)
    
    def _generate_visual_embeddings(self, df, clip_path, data_dir, num_frames=8):
        # Check if the cache files exist
        if self.noise_std > 0:
            cache_name = f"cached_clip_embeddings_noise_{self.noise_std}.pt"
        else:
            cache_name = "cached_clip_embeddings.pt"
        cache_path = os.path.join(data_dir, cache_name)

        if os.path.exists(cache_path):
            print(now_time() + f"Loading cached CLIP embeddings from {cache_path}...")
            return torch.load(cache_path)

        print(now_time() + f"Initializing CLIP for [{self.task}] feature extraction...")
        model = CLIPModel.from_pretrained(clip_path).to(self.device)
        processor = CLIPProcessor.from_pretrained(clip_path)
        embeddings = {}
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting CLIP Features"):
            try:
                if self.task != "msrvtt":
                    img_path = row['image_path']
                    if not os.path.exists(img_path): continue
                
                    image = Image.open(img_path).convert("RGB")

                    if self.noise_std > 0:
                        img_array = np.array(image).astype(np.float32) / 255.0
                        noise = np.random.normal(0, self.noise_std, img_array.shape)
                        img_array = np.clip(img_array + noise, 0, 1) # values in [0,1]
                        image = Image.fromarray((img_array * 255).astype(np.uint8))

                    inputs = processor(images=image, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        emb = model.get_image_features(**inputs).cpu()
                    embeddings[row['image_id']] = emb.squeeze(0)
                
                else: # "msrvtt"
                    v_id = row['video_id']
                    v_path = row['video_path']
                    if not os.path.exists(v_path): continue
                
                    cap = cv2.VideoCapture(v_path)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
                    frame_features = []

                    for idx in indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if ret:
                            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            inputs = processor(images=image, return_tensors="pt").to(self.device)
                            with torch.no_grad():
                                emb = model.get_image_features(**inputs).cpu()
                            frame_features.append(emb)
                    
                    cap.release()

                    if frame_features:
                        # Take the average of the features of 8 frames to obtain a single vector representing the entire video.
                        video_emb = torch.mean(torch.cat(frame_features, dim=0), dim=0)
                        embeddings[v_id] = video_emb

            except: continue
        
        del model
        torch.cuda.empty_cache()
        
        print(now_time() + f"Saving CLIP embeddings to {cache_path}...")
        torch.save(embeddings, cache_path)
        return embeddings

    
    def _generate_audio_embeddings(self, df, wav2vec_path, data_dir):
        """Extract the audio from the video and convert it into Wav2Vec2 features"""
        cache_path = os.path.join(data_dir, f"cached_msrvtt_audio.pt")
        if os.path.exists(cache_path):
            embeddings = torch.load(cache_path)
            is_all_zeros = all(torch.all(v == 0).item() for v in embeddings.values())
            if not is_all_zeros:
                print(now_time() + "Loading cached Audio embeddings...")
                return embeddings
            else:
                print(now_time() + "[DEBUG] It was found that cached_msrvtt_asr.json were all blank strings. Force re-extraction...")

        print(now_time() + "Extracting Audio & Wav2Vec2 Features...")
        processor = Wav2Vec2Processor.from_pretrained(wav2vec_path)
        model = Wav2Vec2Model.from_pretrained(wav2vec_path).to(self.device)
        embeddings = {}
        error_log_path = os.path.join(data_dir, "audio_debug_log.txt")

        with open(error_log_path, 'w', encoding='utf-8') as err_f:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating Audio"):
                v_id = row['video_id']
                v_path = row['video_path']
                
                if not os.path.exists(v_path):
                    err_f.write(f"[{v_id}] Video missing: {v_path}\n")
                    embeddings[v_id] = torch.zeros(768)
                    continue
                
                try:
                    y, sr = extract_audio_robust(v_path, duration=20.0, sr=16000)
                    if len(y) == 0:
                        raise ValueError("Audio length is 0 (This video may not have a soundtrack.)")
                        
                    input_values = processor(y, sampling_rate=sr, return_tensors="pt").input_values.to(self.device)
                    with torch.no_grad():
                        outputs = model(input_values)
                        emb = outputs.last_hidden_state.mean(dim=1).cpu()
                    embeddings[v_id] = emb.squeeze(0)
                except Exception as e:
                    err_f.write(f"\n--- Error Extracting Audio for {v_id} ({v_path}) ---\n")
                    err_f.write(traceback.format_exc())
                    err_f.write("-" * 50 + "\n")
                    embeddings[v_id] = torch.zeros(768) 
        
        torch.save(embeddings, cache_path)
        print(now_time() + f"If there is audio extraction failure, the detailed error has been saved in: {error_log_path}")
        return embeddings

    def _generate_asr_texts(self, df, wav2vec_path, data_dir):
        """Extract the audio from the video and convert it to transcribed text using Wav2Vec2."""
        cache_path = os.path.join(data_dir, f"cached_msrvtt_asr.json")
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)

            valid_count = sum(1 for v in cached_data.values() if v.strip() != "")
            if valid_count > 0:
                print(now_time() + f"Loading cached ASR texts... ({valid_count} valid transcripts)")
                return cached_data
            else:
                print(now_time() + "[DEBUG] It was found that cached_msrvtt_asr.json were all blank strings. Force re-extraction...")

        print(now_time() + "Extracting Audio & Generating ASR Texts...")
        processor = Wav2Vec2Processor.from_pretrained(wav2vec_path)
        model = Wav2Vec2ForCTC.from_pretrained(wav2vec_path).to(self.device)
        asr_texts = {}
        error_log_path = os.path.join(data_dir, "asr_debug_log.txt")

        with open(error_log_path, 'w', encoding='utf-8') as err_f:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating ASR"):
                v_id = row['video_id']
                v_path = row['video_path']
                
                if not os.path.exists(v_path):
                    err_f.write(f"[{v_id}] Video missing: {v_path}\n")
                    asr_texts[v_id] = ""
                    continue
                    
                try:
                    y, sr = extract_audio_robust(v_path, duration=20.0, sr=16000)
                    if len(y) == 0:
                        raise ValueError("Audio length is 0 (This video may not have a soundtrack.)")
                        
                    input_values = processor(y, sampling_rate=sr, return_tensors="pt").input_values.to(self.device)
                    with torch.no_grad():
                        logits = model(input_values).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription = processor.batch_decode(predicted_ids)[0]
                    asr_texts[v_id] = transcription.lower()
                except Exception as e:
                    err_f.write(f"\n--- Error Extracting ASR for {v_id} ({v_path}) ---\n")
                    err_f.write(traceback.format_exc())
                    err_f.write("-" * 50 + "\n")
                    asr_texts[v_id] = ""
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(asr_texts, f, ensure_ascii=False, indent=4)
            
        del model
        torch.cuda.empty_cache()
        print(now_time() + f"If there is an audio translation failure, the error details have been saved in: {error_log_path}")
        return asr_texts
    
    def _split_data(self, df):
        if self.task == 'msrvtt':
            train_df = df[df['split'] == 'train']
            valid_df = df[df['split'] == 'valid']
            test_df = df[df['split'] == 'test']

            train_data = train_df.to_dict('records')
            valid_data = valid_df.to_dict('records')
            test_data = test_df.to_dict('records')

            def format_item(d):
                if 'category' in d:
                    if isinstance(d['category'], int):
                        category_id = d['category']
                        category_name = id_to_cat.get(category_id, 'music')
                    else:
                        category_name = str(d['category'])
                        category_id = cat_to_id.get(category_name, 0)
                else:
                    category_name = d.get('category_name', 'music')
                    category_id = cat_to_id.get(category_name, 0)
                
                return {
                    'id': d['video_id'],
                    'text': self.asr_texts.get(d['video_id'], d.get('asr', '')),
                    'label': d.get('caption', ''),
                    'category_name': category_name
                }

            train_data = [format_item(d) for d in train_data]
            valid_data = [format_item(d) for d in valid_data]
            test_data = [format_item(d) for d in test_data]

            return train_data, valid_data, test_data


        train_data, valid_data, test_data = [], [], []
        random_pool = []

        for _, row in df.iterrows():
            img_id = row['image_id']
            if img_id not in self.image_embeddings:
                continue
            
            text = ""
            label = -1
            
            if self.task == 'twitter17':
                text = row['text']
                label = row['label']
            elif self.task == 'hateful':
                text = row['text']
                label = row['label']
            elif self.task == 'scienceqa':
                text = row['text']
                label = row['answer']
            elif self.task == 'flickr':
                text = row['text']
                label = row['label']
            else: # MVSA
                try:
                    with open(row['text_path'], 'r', encoding='utf-8', errors='replace') as f:
                        text = f.read().strip()
                    label = self.label_map[row['sentiment']]
                except:
                    continue
            
            if not text or (self.task != 'flickr' and label == -1): 
                continue

            item = {'id': img_id, 'text': text, 'label': int(label) if self.task != 'flickr' else label}

            if self.task == 'twitter17':
                twitter_item = {'id': img_id, 'text': text, 'aspect': row['aspect'], 'label': int(label)}
                if row['split_hint'] == 'train':
                    train_data.append(twitter_item)
                elif row['split_hint'] == 'val':
                    valid_data.append(twitter_item)
                else:
                    test_data.append(twitter_item)
            else:
                random_pool.append(item)

        # Flickr 8k
        if self.task == 'flickr' and random_pool:
            from collections import defaultdict
            grouped = defaultdict(list)
            for item in random_pool:
                grouped[item['id']].append(item)
            
            ids = list(grouped.keys())
            random.seed(1111)
            random.shuffle(ids)
            n = len(ids)
            train_ids = ids[:int(n*0.8)]
            valid_ids = ids[int(n*0.8):int(n*0.9)]
            test_ids = ids[int(n*0.9):]
            
            train_data = [x for i in train_ids for x in grouped[i]]
            valid_data = [x for i in valid_ids for x in grouped[i]]
            test_data = [x for i in test_ids for x in grouped[i]]

        # MVSA, Hateful, ScienceQA
        elif self.task != 'twitter17' and random_pool:
            random.seed(1111)
            random.shuffle(random_pool)
            n = len(random_pool)
            train_data = random_pool[:int(n*0.8)]
            valid_data = random_pool[int(n*0.8):int(n*0.9)]
            test_data = random_pool[int(n*0.9):]
        
        return train_data, valid_data, test_data
            

class Batchify:
    def __init__(self, data, tokenizer, batch_size, task='mvsa', shuffle=False):
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.task = task
        self.sample_num = len(data)
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0
        self.index_list = list(range(self.sample_num))
        
        self.id_to_label = {0: "neutral", 1: "positive", 2: "negative"}

    def next_batch(self, mode='train'):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle: random.shuffle(self.index_list)

        idx_batch = self.index_list[self.step*self.batch_size : (self.step+1)*self.batch_size]
        self.step += 1
        batch_data = [self.data[i] for i in idx_batch]
        
        img_ids = [d['id'] for d in batch_data]
        aud_ids = img_ids if self.task == 'msrvtt' else None
        if self.task in ['flickr', 'msrvtt']:
            raw_labels = [d['label'] for d in batch_data] # For Captioning task, label is text
        else:
            raw_labels = torch.tensor([d['label'] for d in batch_data])
        
        prompts = []
        target_lens = []
        for d in batch_data:
            # 1. Designing Prompt
            if self.task == 'hateful':
                p = f"Instruction: Does this meme contain hateful content? Answer yes or no.\nText: {d['text']}\nAnswer: "
                label_word = "yes" if d['label'] == 1 else "no"
            elif self.task == 'scienceqa':
                p = f"Instruction: Choose the correct option letter based on the image.\n{d['text']}\nAnswer: "
                label_word = chr(65 + d['label']) # 0->A, 1->B
            elif self.task == 'twitter17' or self.task == 'twitter15':
                p = f"Instruction: Identify the sentiment toward the entity '{d['aspect']}' in the following text based on the image.\nText: {d['text']}\nAnswer: "
                label_word = "positive" if d['label'] == 1 else ("negative" if d['label'] == 2 else "neutral")
            elif self.task == 'flickr':
                p = f"Instruction: Generate a caption for this image.\nAnswer: "
                label_word = d['label']
            elif self.task == 'msrvtt':
                if d['text']: # If there is a text that has been translated from speech
                    p = f"Instruction: Generate a description for this video based on the visual, audio, and transcribed text.\nTranscribed Text: {d['text']}\nAnswer: "
                else:
                    p = f"Instruction: Generate a description for this video.\nAnswer: "
                label_word = d['label']
            else: # mvsa
                p = f"Instruction: Classify sentiment as positive, negative, or neutral.\nText: {d['text']}\nSentiment: "
                label_word = self.id_to_label[d['label']]

            if mode == 'train':
                full_text = p + label_word + self.tokenizer.eos_token
                prompts.append(full_text)
                l_len = len(self.tokenizer.encode(label_word + self.tokenizer.eos_token, add_special_tokens=False))
                target_lens.append(l_len)
            else:
                prompts.append(p)
                target_lens.append(0)

        self.tokenizer.padding_side = 'left'
        encoded = self.tokenizer(prompts, padding=True, truncation=True, max_length=256, return_tensors='pt')
        
        return img_ids, aud_ids, encoded['input_ids'], encoded['attention_mask'], target_lens, raw_labels