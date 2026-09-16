import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

def print_trainable_parameters(model, label_name="Model"):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print("\n" + "="*60)
    print(f"[{label_name}] Parameter Efficiency Analysis:")
    print(f"--> Trainable Parameters: {trainable_params:,}")
    print(f"--> Total Parameters:     {all_param:,}")
    print(f"--> Tuning Ratio:         {100 * trainable_params / all_param:.4f}%")
    print("="*60 + "\n")
    return trainable_params

class MyLoraLayer(nn.Module):
    def __init__(self, base_layer, hidden_size, r=8, lora_alpha=16, multimodal_rank_buff=4, dtype=torch.bfloat16):
        super().__init__()
        self.base_layer = base_layer
        self.hidden_size = hidden_size
        self.dtype = dtype
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features

        device = base_layer.weight.device

        self.lora_A_t = nn.Parameter(torch.randn(r, in_features, dtype=dtype, device=device))
        self.lora_B_t = nn.Parameter(torch.zeros(out_features, r, dtype=dtype, device=device))
        
        self.has_multimodal = (in_features == hidden_size) # 6 modules
        # self.has_multimodal = (in_features == hidden_size or out_features == hidden_size) # All 7 modules
        
        if self.has_multimodal:
            mm_rank = r * multimodal_rank_buff
            # Image LoRA
            self.lora_A_img = nn.Parameter(torch.randn(mm_rank, self.hidden_size, dtype=dtype, device=device))
            self.lora_B_img = nn.Parameter(torch.zeros(out_features, mm_rank, dtype=dtype, device=device))
            # Audio LoRA
            self.lora_A_aud = nn.Parameter(torch.randn(mm_rank, self.hidden_size, dtype=dtype, device=device))
            self.lora_B_aud = nn.Parameter(torch.zeros(out_features, mm_rank, dtype=dtype, device=device))
        
        self.scaling = lora_alpha / r
        self.x_img = None 
        self.x_aud = None

    def forward(self, x):
        module_device = self.lora_A_t.device
        # Avoid unnecessary copy when already correct
        if x.device != module_device or x.dtype != self.dtype:
            x_lora = x.to(
                device=module_device,
                dtype=self.dtype,
            )
        else:
            x_lora = x

        # Base model
        result = self.base_layer(x_lora)

        # Text LoRA path
        text_low = F.linear(
            x_lora,
            self.lora_A_t,
        )

        text_delta = F.linear(
            text_low,
            self.lora_B_t,
        )
        result = result + text_delta * self.scaling

        # Multimodal LoRAs
        if self.has_multimodal:

            if self.x_img is not None:
                xi = self.x_img

                if xi.device != module_device or xi.dtype != self.dtype:
                    xi = xi.to(
                        device=module_device,
                        dtype=self.dtype,
                    )
            
                img_low = F.linear(
                    xi,
                    self.lora_A_img,
                )

                # -> [B, out_features]
                img_delta = F.linear(
                    img_low,
                    self.lora_B_img,
                )
                img_delta = img_delta.unsqueeze(1)
                scaling_img = 3.0
                result = result + img_delta * self.scaling * scaling_img
        
            if self.x_aud is not None:
                xa = self.x_aud

                if xa.device != module_device or xa.dtype != self.dtype:
                    xa = xa.to(
                        device=module_device,
                        dtype=self.dtype,
                    )
                aud_low = F.linear(
                    xa,
                    self.lora_A_aud,
                )

                # -> [B, out_features]
                aud_delta = F.linear(
                    aud_low,
                    self.lora_B_aud,
                )
                aud_delta = aud_delta.unsqueeze(1)
                scaling_aud = 8.0
                result = result + aud_delta * self.scaling * scaling_aud
            
        return result

class msLoRA(nn.Module):
    def __init__(self, pretrained_path, r, lora_modules_count, image_embeddings, audio_embeddings=None, multimodal_rank_buff=4, load_in_8bit=True, dtype=torch.bfloat16):
        super().__init__()
        
        config = AutoConfig.from_pretrained(pretrained_path)
        if hasattr(config, "pretraining_tp") and config.pretraining_tp != 1:
            config.pretraining_tp = 1
            
        quantization_config = None
        if load_in_8bit: quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_path, 
            config=config,
            quantization_config=quantization_config,
            torch_dtype=dtype, 
            device_map="auto"
        )
        self.image_embeddings = image_embeddings
        self.audio_embeddings = audio_embeddings
        self.multimodal_rank_buff = multimodal_rank_buff
        self.dtype = dtype
        
        self.hidden_size = self.model.config.hidden_size
        img_dim = 512 # CLIP hidden_size
        aud_dim = 768
        
        self.f_img = nn.Linear(img_dim, self.hidden_size, dtype=dtype).to(self.model.device)
        self.f_aud = nn.Linear(aud_dim, self.hidden_size, dtype=dtype).to(self.model.device)
        
        all_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        target_modules = all_target_modules[:lora_modules_count]
        
        print(f"Targeting modules for LoRA: {target_modules}")

        # Traverse all layers and replace them with custom LoRA layers
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear) and any(m in name for m in target_modules):
                parts = name.rsplit('.', 1)
                if len(parts) > 1:
                    parent = self.model.get_submodule(parts[0])
                    child_name = parts[1]
                else:
                    parent = self.model
                    child_name = parts[0]
                
                new_layer = MyLoraLayer(module, self.hidden_size, r, multimodal_rank_buff=multimodal_rank_buff, dtype=dtype)
                setattr(parent, child_name, new_layer)
        
        # Freeze the original model parameters and only train LoRA and f_img
        for n, p in self.model.named_parameters():
            if "lora_" not in n.lower():
                p.requires_grad = False
        for p in list(self.f_img.parameters()) + list(self.f_aud.parameters()):
            p.requires_grad = True

        print_trainable_parameters(self, label_name=f"msLoRA Architecture (Ours, r={r})")

    def set_multimodal_features(self, img_ids, aud_ids=None):
        device = next(self.f_img.parameters()).device
        
        raw_embs = torch.stack([self.image_embeddings[id] for id in img_ids]).to(device, dtype=self.dtype)
        
        x_img = self.f_img(raw_embs)
        x_img = torch.nn.functional.normalize(x_img, p=2, dim=-1)

        x_aud = None
        if self.audio_embeddings is not None and aud_ids is not None:
            # MSR-VTT with audio ID
            aud_raw = torch.stack([self.audio_embeddings[i] for i in aud_ids]).to(device, dtype=self.dtype)
            x_aud = torch.nn.functional.normalize(self.f_aud(aud_raw), p=2, dim=-1)
        
        # Distribute to all msLoRA layers
        for m in self.model.modules():
            if isinstance(m, MyLoraLayer):
                m.x_img = x_img
                m.x_aud = x_aud

    def forward(self, input_ids, attention_mask, img_ids, aud_ids=None, target_lens=None):
        self.set_multimodal_features(img_ids, aud_ids)
        
        # Build labels for training (ignore the Prompt part, only calculate the Loss of the Label part)
        labels = torch.full_like(input_ids, -100, device=self.model.device)
        if target_lens is not None:
            for i, t_len in enumerate(target_lens):
                if t_len > 0:
                    # Labels on the right (usually right-aligned during training, or determined by padding logic)
                    labels[i, -t_len:] = input_ids[i, -t_len:]
        
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    

class LoraLayer(nn.Module):
    def __init__(self, base_layer, hidden_size, r=8, lora_alpha=16, dtype=torch.bfloat16, lora_name='LoRA'):
        super().__init__()
        self.base_layer = base_layer
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.lora_name = lora_name
        self.scaling = lora_alpha / r
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        device = base_layer.weight.device

        self.lora_A_t = nn.Parameter(torch.randn(r, in_features, dtype=dtype, device=device))
        self.lora_B_t = nn.Parameter(torch.zeros(out_features, r, dtype=dtype, device=device))

    def forward(self, x):
        module_device = self.lora_A_t.device
        x_on_device = x.to(device=module_device, dtype=self.dtype)

        result = self.base_layer(x_on_device)

        lora_t = (x_on_device @ self.lora_A_t.t() @ self.lora_B_t.t()) * self.scaling
        result += lora_t

        return result

class LoRA_Concat(nn.Module):
    def __init__(self, pretrained_path, r, lora_modules_count, image_embeddings, audio_embeddings=None, load_in_8bit=True, dtype=torch.bfloat16, lora_name='LoRA'):
        super().__init__()
        
        config = AutoConfig.from_pretrained(pretrained_path)
        if hasattr(config, "pretraining_tp") and config.pretraining_tp != 1:
            config.pretraining_tp = 1
            
        quantization_config = None
        if load_in_8bit: quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_path, 
            config=config,
            quantization_config=quantization_config,
            torch_dtype=dtype, 
            device_map="auto"
        )
        self.image_embeddings = image_embeddings
        self.audio_embeddings = audio_embeddings
        self.dtype = dtype
        self.hidden_size = self.model.config.hidden_size
        
        # 1. Image Projector
        img_dim = 512 # CLIP hidden_size
        #self.visual_projector = nn.Sequential(
        #    nn.Linear(img_dim, self.hidden_size, dtype=dtype),
        #    nn.GELU(),
        #    nn.Linear(self.hidden_size, self.hidden_size, dtype=dtype)
        #).to(self.model.device)
        self.visual_projector = nn.Linear(img_dim, self.hidden_size, dtype=dtype).to(self.model.device)
        # 2. Audio Projector (Wav2Vec2 768 -> LLM H)
        aud_dim = 768 # Wav2Vec2 hidden_size
        self.audio_projector = nn.Linear(aud_dim, self.hidden_size, dtype=dtype).to(self.model.device)

        # 3. Inject LoRA
        all_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        target_modules = all_target_modules[:lora_modules_count]

        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear) and any(m in name for m in target_modules):
                parts = name.rsplit('.', 1)
                parent = self.model.get_submodule(parts[0]) if len(parts) > 1 else self.model
                child_name = parts[-1]
                
                new_layer = LoraLayer(module, self.hidden_size, r, dtype=dtype, lora_name=lora_name)
                setattr(parent, child_name, new_layer)
        
        # 3. Set trainable parameters
        for n, p in self.named_parameters():
            p.requires_grad = ("lora_" in n.lower() or "projector" in n)

        print_trainable_parameters(self, label_name=f"LoRA")

    def _get_multimodal_embeds(self, img_ids, aud_ids=None):
        """Inouts: [Visual_Token, Audio_Token, Text_Tokens]"""
        device = self.model.device
        batch_size = len(img_ids)
        
        # Image embedding size [B, 1, H]
        raw_v = torch.stack([self.image_embeddings[i] for i in img_ids]).to(device, dtype=self.dtype)
        v_embeds = self.visual_projector(raw_v).unsqueeze(1)
        
        # Audio embedding size: [B, 1, H]
        a_embeds = None
        if self.audio_embeddings is not None and aud_ids is not None:
            raw_a = torch.stack([self.audio_embeddings[i] for i in aud_ids]).to(device, dtype=self.dtype)
            a_embeds = self.audio_projector(raw_a).unsqueeze(1)
            
        # concatenate non-text modalities: [B, N, H]
        if a_embeds is not None:
            mm_embeds = torch.cat([v_embeds, a_embeds], dim=1)
        else:
            mm_embeds = v_embeds
            
        return mm_embeds

    def forward(self, input_ids, attention_mask, img_ids, aud_ids=None, target_lens=None):
        device = self.model.device
        
        # 1. Obtain the multimodal parts
        mm_embeds = self._get_multimodal_embeds(img_ids, aud_ids)
        mm_len = mm_embeds.shape[1]
        
        # 2. text tokens: [B, S, H]
        inputs_embeds = self.model.get_input_embeddings()(input_ids.to(device))
        
        # 3. all input token: [B, mm_len + S, H]
        full_embeds = torch.cat([mm_embeds, inputs_embeds], dim=1)
        
        # 4. Mask
        mm_mask = torch.ones((attention_mask.shape[0], mm_len), device=device)
        full_mask = torch.cat([mm_mask, attention_mask.to(device)], dim=1)
        
        # 5. Labels (Label is always at the end of the sequence and is not affected by modal increase)
        labels = torch.full((full_embeds.shape[0], full_embeds.shape[1]), -100, device=device)
        if target_lens is not None:
            for i, t_len in enumerate(target_lens):
                if t_len > 0:
                    labels[i, -t_len:] = input_ids[i, -t_len:]
        
        return self.model(inputs_embeds=full_embeds, attention_mask=full_mask, labels=labels)

    def generate(self, input_ids, attention_mask, img_ids, aud_ids=None, **kwargs):
        device = self.model.device
        mm_embeds = self._get_multimodal_embeds(img_ids, aud_ids)
        inputs_embeds = self.model.get_input_embeddings()(input_ids.to(device))
        
        full_embeds = torch.cat([mm_embeds, inputs_embeds], dim=1)
        mm_mask = torch.ones((attention_mask.shape[0], mm_embeds.shape[1]), device=device)
        full_mask = torch.cat([mm_mask, attention_mask.to(device)], dim=1)
        
        return self.model.generate(inputs_embeds=full_embeds, attention_mask=full_mask, **kwargs)
