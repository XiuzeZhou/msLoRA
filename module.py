import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig

class MyLoraLayer(nn.Module):
    def __init__(self, base_layer, hidden_size, r=8, lora_alpha=16, multimodal_scaling=4, dtype=torch.bfloat16):
        super().__init__()
        self.base_layer = base_layer
        self.hidden_size = hidden_size
        self.multimodal_scaling = multimodal_scaling
        self.dtype = dtype
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features

        device = base_layer.weight.device

        self.lora_A_t = nn.Parameter(torch.randn(r, in_features, dtype=dtype, device=device))
        self.lora_B_t = nn.Parameter(torch.zeros(out_features, r, dtype=dtype, device=device))
        
        self.has_multimodal = (in_features == hidden_size) # The six modules except for down_proj
        self.has_multimodal = (in_features == hidden_size or out_features == hidden_size) # All 7 modules
        
        if self.has_multimodal:
            # Image LoRA
            self.lora_A_img = nn.Parameter(torch.randn(r*self.multimodal_scaling, self.hidden_size, dtype=dtype, device=device))
            self.lora_B_img = nn.Parameter(torch.zeros(out_features, r*self.multimodal_scaling, dtype=dtype, device=device))
            # Audio LoRA
            self.lora_A_aud = nn.Parameter(torch.randn(r*self.multimodal_scaling, self.hidden_size, dtype=dtype, device=device))
            self.lora_B_aud = nn.Parameter(torch.zeros(out_features, r*self.multimodal_scaling, dtype=dtype, device=device))
        
        self.scaling = lora_alpha / r
        self.x_img = None 
        self.x_aud = None

    def forward(self, x):
        module_device = self.lora_A_t.device
        x_on_device = x.to(device=module_device, dtype=self.dtype)
        result = self.base_layer(x_on_device)

        # Text LoRA path
        lora_t = (x_on_device @ self.lora_A_t.t() @ self.lora_B_t.t()) * self.scaling
        result += lora_t

        # Image LoRA path
        if self.has_multimodal:
            seq_len = x_on_device.size(1)

            if self.x_img is not None:
                xi = self.x_img.to(module_device, dtype=self.dtype)
                xi_exp = xi.unsqueeze(1).expand(-1, seq_len, -1)
            
                lora_img = (xi_exp @ self.lora_A_img.t() @ self.lora_B_img.t()) * self.scaling
                result += lora_img * 4.0
        
            if self.x_aud is not None:
                xa = self.x_aud.to(module_device, dtype=self.dtype)
                xa_exp = xa.unsqueeze(1).expand(-1, seq_len, -1)
                lora_aud= (xa_exp @ self.lora_A_aud.t() @ self.lora_B_aud.t()) * self.scaling
                result += lora_aud * 8.0
            
        return result

class msLoRA(nn.Module):
    def __init__(self, pretrained_path, r, lora_modules_count, image_embeddings, audio_embeddings=None, multimodal_scaling=4, dtype=torch.bfloat16):
        super().__init__()
        
        config = AutoConfig.from_pretrained(pretrained_path)
        if hasattr(config, "pretraining_tp") and config.pretraining_tp != 1:
            config.pretraining_tp = 1
            
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_path, 
            config=config,
            torch_dtype=dtype, 
            device_map="auto"
        )
        self.image_embeddings = image_embeddings
        self.audio_embeddings = audio_embeddings
        self.multimodal_scaling = multimodal_scaling
        self.r = r
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
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and any(m in name for m in target_modules):
                parts = name.rsplit('.', 1)
                if len(parts) > 1:
                    parent = self.model.get_submodule(parts[0])
                    child_name = parts[1]
                else:
                    parent = self.model
                    child_name = parts[0]
                
                new_layer = MyLoraLayer(module, self.hidden_size, r, multimodal_scaling=multimodal_scaling, dtype=dtype)
                setattr(parent, child_name, new_layer)
        
        # Freeze the original model parameters and only train LoRA and f_img
        for n, p in self.model.named_parameters():
            if "lora_" not in n:
                p.requires_grad = False
        for p in list(self.f_img.parameters()) + list(self.f_aud.parameters()):
            p.requires_grad = True

    def set_multimodal_features(self, img_ids, aud_ids=None):
        device = next(self.f_img.parameters()).device
        
        raw_embs = torch.stack([self.image_embeddings[id] for id in img_ids]).to(device, dtype=self.dtype)
        
        x_img = self.f_img(raw_embs)
        x_img = torch.nn.functional.normalize(x_img, p=2, dim=-1)

        x_aud = None
        if self.audio_embeddings is not None and aud_ids is not None:
            aud_raw = torch.stack([self.audio_embeddings[i] for i in aud_ids]).to(device, dtype=self.dtype)
            x_aud = torch.nn.functional.normalize(self.f_aud(aud_raw), p=2, dim=-1)
        
        # Distribute to all msLoRA layers
        for m in self.model.modules():
            if isinstance(m, MyLoraLayer):
                m.x_img = x_img
                m.x_aud = x_aud

    def forward(self, input_ids, attention_mask, img_ids, aud_ids=None, target_lens=None):
        self.set_multimodal_features(img_ids, aud_ids)
        
        labels = torch.full_like(input_ids, -100, device=self.model.device)
        if target_lens is not None:
            for i, t_len in enumerate(target_lens):
                if t_len > 0:
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
        
        if self.lora_name == 'MultiAysmLoRA':
            self.lora_A_t_2 = nn.Parameter(torch.randn(r*4, in_features, dtype=dtype, device=device))
            self.lora_B_t_2 = nn.Parameter(torch.zeros(out_features, r*4, dtype=dtype, device=device))
            self.lora_A_t_3 = nn.Parameter(torch.randn(r*4, in_features, dtype=dtype, device=device))
            self.lora_B_t_3 = nn.Parameter(torch.zeros(out_features, r*4, dtype=dtype, device=device))

    def forward(self, x):
        module_device = self.lora_A_t.device
        x_on_device = x.to(device=module_device, dtype=self.dtype)

        result = self.base_layer(x_on_device)

        lora_t = (x_on_device @ self.lora_A_t.t() @ self.lora_B_t.t()) * self.scaling
        result += lora_t

        if self.lora_name == 'MultiAysmLoRA':
            lora_t_2 = (x_on_device @ self.lora_A_t_2.t() @ self.lora_B_t_2.t()) * self.scaling
            lora_t_3 = (x_on_device @ self.lora_A_t_3.t() @ self.lora_B_t_3.t()) * self.scaling
            result += lora_t_2 + lora_t_3

        return result

class msLoRA_Concat(nn.Module):
    def __init__(self, pretrained_path, r, lora_modules_count, image_embeddings, audio_embeddings=None, dtype=torch.bfloat16, lora_name='LoRA'):
        super().__init__()
        
        config = AutoConfig.from_pretrained(pretrained_path)
        if hasattr(config, "pretraining_tp") and config.pretraining_tp != 1:
            config.pretraining_tp = 1
            
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_path, 
            config=config,
            torch_dtype=dtype, 
            device_map="auto"
        )
        self.image_embeddings = image_embeddings
        self.audio_embeddings = audio_embeddings
        self.dtype = dtype
        self.hidden_size = self.model.config.hidden_size
        
        # 1. Image Projector
        self.visual_projector = nn.Sequential(
            nn.Linear(512, self.hidden_size, dtype=dtype),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size, dtype=dtype)
        ).to(self.model.device)

        # 2. Audio Projector (Wav2Vec2 768 -> LLM H)
        self.audio_projector = nn.Sequential(
            nn.Linear(768, self.hidden_size, dtype=dtype),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size, dtype=dtype)
        ).to(self.model.device)

        # 3. Inject LoRA
        all_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        target_modules = all_target_modules[:lora_modules_count]

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and any(m in name for m in target_modules):
                parts = name.rsplit('.', 1)
                parent = self.model.get_submodule(parts[0]) if len(parts) > 1 else self.model
                child_name = parts[-1]
                
                new_layer = LoraLayer(module, self.hidden_size, r, dtype=dtype, lora_name=lora_name)
                setattr(parent, child_name, new_layer)
        
        # 3. Set trainable parameters
        for n, p in self.named_parameters():
            p.requires_grad = ("lora_" in n or "projector" in n)

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
        
        mm_embeds = self._get_multimodal_embeds(img_ids, aud_ids)
        mm_len = mm_embeds.shape[1]
        
        inputs_embeds = self.model.get_input_embeddings()(input_ids.to(device))
        
        full_embeds = torch.cat([mm_embeds, inputs_embeds], dim=1)
        
        mm_mask = torch.ones((attention_mask.shape[0], mm_len), device=device)
        full_mask = torch.cat([mm_mask, attention_mask.to(device)], dim=1)
        
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