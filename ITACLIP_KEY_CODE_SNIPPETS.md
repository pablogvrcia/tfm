# ITACLIP: Key Code Snippets for SAM2+CLIP Adaptation

## 1. MULTI-LAYER ATTENTION FUSION (clip/model.py, lines 250-267)

**The core architectural innovation - how to extract and fuse intermediate layer features:**

```python
# In VisionTransformer.forward(), after ln_pre and before final layer
layer_count = 0
attn_list = []
selected_intermediate_layers = [7, 8, 10]  # Out of 12 layers

for blk in self.transformer.resblocks[:-1]:  # All but last layer
    layer_count += 1
    if layer_count in selected_intermediate_layers:
        # Extract attention maps from strategic layers
        saved_attn = self.custom_attn(
            blk.attn, 
            blk.ln_1(x), 
            return_attn=True, 
            attn_self=attn_self
        )
        attn_list.append(saved_attn)
    x = blk(x)

# Average intermediate attention maps
avg_attn = torch.mean(torch.stack(attn_list), dim=0)

# Process final layer with fused attention
for blk in self.transformer.resblocks[-1:]:
    custom_attn = self.custom_attn(blk.attn, blk.ln_1(x), return_attn=True, attn_self=attn_self)
    # 50-50 fusion with intermediate layers
    avg_attn = 0.5 * custom_attn + 0.5 * avg_attn
    # Use fused attention in final layer
    x = x + self.use_saved_attn(blk.attn, blk.ln_1(x), avg_attn)
```

**For SAM2**: Extract features from multiple decoder levels and use similar weighted fusion.

---

## 2. CUSTOM ATTENTION WITH SELF-ATTENTION OPTION (clip/model.py, lines 313-343)

**Enables alternative attention computation for capturing token self-relationships:**

```python
def custom_attn(self, attn_layer, x, return_attn=False, with_attn=False, attn_self=False):
    num_heads = attn_layer.num_heads
    _, bsz, embed_dim = x.size()
    head_dim = embed_dim // num_heads
    scale = head_dim ** -0.5
    
    q, k, v = F.linear(x, attn_layer.in_proj_weight, attn_layer.in_proj_bias).chunk(3, dim=-1)
    q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    
    if attn_self:  # Alternative: self-self attention
        q_attn = torch.bmm(q, q.transpose(1, 2)) * scale
        k_attn = torch.bmm(k, k.transpose(1, 2)) * scale
        attn_weights = F.softmax(q_attn, dim=-1) + F.softmax(k_attn, dim=-1)
    else:  # Standard: cross attention
        attn_weights = torch.bmm(q * scale, k.transpose(1, 2))
        attn_weights = F.softmax(attn_weights, dim=-1)
    
    if return_attn:
        return attn_weights
    
    attn_output = torch.bmm(attn_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
    attn_output = attn_layer.out_proj(attn_output)
    
    if with_attn:
        return attn_output, attn_weights
    return attn_output
```

**Key insight**: Standard attention = Q @ K.T, but self-attention = Q @ Q.T + K @ K.T captures both query and key self-coherence.

---

## 3. IMAGE ENGINEERING: AUGMENTATION ENSEMBLE (itaclip_segmentor.py, lines 42-48, 116-145)

**Multiple augmentation branches combined with weighted averaging:**

```python
# Setup (lines 42-48)
self.transforms = [
    T.Grayscale(),                          # Remove color
    T.GaussianBlur(kernel_size=11, sigma=5) # Heavy blur
]
self.flip_transforms = [
    T.RandomVerticalFlip(p=1),    # Vertical flip
    T.RandomHorizontalFlip(p=1)   # Horizontal flip
]

# Usage (lines 116-145)
def forward_feature(self, img, text_features, logit_size=None):
    img_list = []
    flip_list = []
    
    if not self.img_engineering:
        # Simple path: single encoding
        image_features = self.net.encode_image(img, return_all=True, 
                                              attn_self=self.attn_self, 
                                              device=self.device)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        img_list.append(image_features)
    else:
        # Complex path: multi-branch ensemble
        torch.manual_seed(0)
        
        # Original
        image_features = self.net.encode_image(img, return_all=True, 
                                              attn_self=self.attn_self, 
                                              device=self.device)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        img_list.append(image_features)
        
        # First-category augmentations
        for transform in self.transforms:
            new_img = transform(img.squeeze())
            new_img = new_img.unsqueeze(0)
            if new_img.shape[1] == 1:  # Grayscale -> RGB
                new_img = new_img.expand(1, 3, -1, -1)
            image_features = self.net.encode_image(new_img, return_all=True, 
                                                  attn_self=self.attn_self, 
                                                  device=self.device)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            img_list.append(image_features)
        
        # Second-category augmentations
        for transform in self.flip_transforms:
            new_img = transform(img.squeeze())
            new_img = new_img.unsqueeze(0)
            if new_img.shape[1] == 1:
                new_img = new_img.expand(1, 3, -1, -1)
            flipped_image_features = self.net.encode_image(new_img, return_all=True, 
                                                          attn_self=self.attn_self, 
                                                          device=self.device)
            flipped_image_features /= flipped_image_features.norm(dim=-1, keepdim=True)
            flip_list.append(flipped_image_features)
        
        # Average all augmentations
        image_features = torch.mean(torch.stack(img_list), dim=0)
    
    # Rest of function...
    image_features = image_features[:, 1:]  # Remove CLS token
    logits = image_features @ text_features.T
    
    if self.img_engineering:
        flip_logits = self.get_flipped_logits(flip_list, self.flip_transforms, ...)
        # Weighted ensemble: 75% original + 25% flipped
        logits = (self.img_eng_coefficient) * logits + \
                (1 - self.img_eng_coefficient) * flip_logits
    
    return logits
```

**Key params**: `img_eng_coefficient=0.75` is biased toward original to preserve localization.

---

## 4. TEXT FEATURE GENERATION WITH TEMPLATES (itaclip_segmentor.py, lines 174-185)

**Creating robust class embeddings via template averaging:**

```python
def text_feature(self, query_words, templates=openai_imagenet_template):
    query_features = []
    with torch.no_grad():
        for qw in query_words:
            # Generate 80 different prompts for single class
            prompts = [temp(qw) for temp in templates]
            query = clip.tokenize(prompts).to(self.device)
            
            # Encode all templates
            feature = self.net.encode_text(query)  # [80, 512]
            
            # Normalize each embedding
            feature /= feature.norm(dim=-1, keepdim=True)
            
            # Average across all templates
            feature = feature.mean(dim=0)  # [512]
            
            # Final normalization
            feature /= feature.norm()
            
            query_features.append(feature.unsqueeze(0))
    
    return torch.cat(query_features, dim=0)  # [num_classes, 512]
```

**Example templates from prompts/imagenet_template.py:**
```python
lambda c: f'a bad photo of a {c}.',
lambda c: f'a photo of many {c}.',
lambda c: f'a sculpture of a {c}.',
lambda c: f'a photo of the hard to see {c}.',
lambda c: f'a low resolution photo of the {c}.',
lambda c: f'a rendering of a {c}.',
# ... 74 more covering quality, style, context variations
```

---

## 5. TEXT FEATURE FUSION: CLASS NAME + LLM DEFINITIONS (itaclip_segmentor.py, lines 51-65)

**Combining base class embeddings with enriched LLM-generated descriptions:**

```python
if auxiliary_text_path is None:
    # Only use class names with template augmentation
    self.query_features = self.text_feature(query_words)
else:
    # Load auxiliary texts and combine
    auxiliary_texts = self.get_aux_text(auxiliary_text_path)
    original_features = self.text_feature(query_words)  # [C, 512]
    aux_features = self.text_feature(auxiliary_texts)   # [C, 512]
    
    if self.bg:  # Handle background class separately
        self.query_features = torch.zeros_like(original_features)
        num_bg_words = (self.query_idx == 0).sum().item()
        
        # Map auxiliary features to class indices
        aux_features = aux_features[self.query_idx[num_bg_words:] - 1]
        
        # Weighted combination for non-background
        self.query_features[num_bg_words:] = \
            (1 - self.def_coefficient) * original_features[num_bg_words:] + \
            (self.def_coefficient) * aux_features
        
        # Keep background unchanged
        self.query_features[:num_bg_words] = original_features[:num_bg_words]
    else:
        # Simple weighted average
        aux_features = aux_features[self.query_idx]
        self.query_features = \
            (1 - self.def_coefficient) * original_features + \
            (self.def_coefficient) * aux_features
```

**Coefficients by dataset**:
- COCO-Stuff (171 classes): def_coefficient=0.2 (80% original, 20% aux)
- COCO-Object (81 classes): def_coefficient=0.1 (90% original, 10% aux)
- VOC (20 classes): def_coefficient=0.05 (95% original, 5% aux)

**Pattern**: Fewer classes = less definition boost needed (keep original dominant).

---

## 6. SLIDING WINDOW INFERENCE (itaclip_segmentor.py, lines 187-234)

**Handles large images via overlapping crops with proper averaging:**

```python
def forward_slide(self, img, img_metas, text_features, query_idx, 
                  pamr=None, stride=112, crop_size=224):
    if type(img) == list:
        img = img[0].unsqueeze(0)
    if type(stride) == int:
        stride = (stride, stride)
    if type(crop_size) == int:
        crop_size = (crop_size, crop_size)
    
    h_stride, w_stride = stride
    h_crop, w_crop = crop_size
    batch_size, _, h_img, w_img = img.shape
    out_channels = len(query_idx)
    
    # Calculate grid dimensions
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    
    # Output accumulators
    preds = img.new_zeros((batch_size, out_channels, h_img, w_img), device=self.device)
    count_mat = img.new_zeros((batch_size, 1, h_img, w_img), device=self.device)
    
    # Slide window across image
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            
            # Adjust to not go out of bounds
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            
            # Extract crop
            crop_img = img[:, :, y1:y2, x1:x2]
            
            # Infer on crop
            crop_seg_logit = self.forward_feature(crop_img, text_features=text_features)
            
            # Accumulate with proper padding
            preds += nn.functional.pad(
                crop_seg_logit,
                (int(x1), int(preds.shape[3] - x2), 
                 int(y1), int(preds.shape[2] - y2))
            )
            
            # Track how many times each pixel was predicted
            count_mat[:, :, y1:y2, x1:x2] += 1
    
    # Average by count to handle overlaps
    assert (count_mat == 0).sum() == 0, "Some pixels never covered!"
    preds = preds / count_mat
    
    # Resize to original size
    img_size = img_metas[0]['ori_shape'][:2]
    logits = nn.functional.interpolate(preds, size=img_size, mode='bilinear')
    
    # Optional PAMR refinement
    if pamr:
        img = nn.functional.interpolate(img, size=img_size, mode='bilinear')
        self.pamr = self.pamr.to(self.device)
        logits = self.pamr(img, logits.to(img.dtype)).to(img.dtype)
    
    return logits
```

**COCO-Stuff config**: stride=28, crop_size=224 -> 8x overlap coverage.

---

## 7. MASK POSTPROCESSING (itaclip_segmentor.py, lines 264-301)

**Applies threshold, morphological ops, and area filtering:**

```python
def postprocess_result(self, seg_logits, data_samples, query_idx):
    batch_size = seg_logits.shape[0]
    
    for i in range(batch_size):
        seg_logits_i = seg_logits[i] * self.logit_scale  # Scale logits
        seg_logits_i = seg_logits_i.softmax(0)  # [num_classes, H, W]
        
        num_cls, num_queries = max(query_idx) + 1, len(query_idx)
        
        # Handle multiple text queries per class
        if num_cls != num_queries:
            seg_logits_i = seg_logits_i.unsqueeze(0)  # [1, num_queries, H, W]
            cls_index = nn.functional.one_hot(query_idx)  # [num_queries, num_cls]
            cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
            
            # Max pool across query variants
            seg_logits_i = (seg_logits_i * cls_index).max(1)[0]  # [num_cls, H, W]
        
        # Area threshold: suppress small regions
        if self.area_thd is not None:
            predictions = nn.functional.one_hot(
                seg_logits_i.argmax(0), num_cls
            ).to(seg_logits_i.dtype)
            
            # Count pixels per non-background class
            area_pred = predictions[:, :, 1:].sum((0, 1), keepdim=True)
            
            # Keep only if area > threshold
            area_pred = (area_pred > self.area_thd * area_pred.sum()).to(seg_logits_i.dtype)
            seg_logits_i[1:] *= area_pred.transpose(0, -1)
        
        # Get hard predictions
        seg_pred = seg_logits_i.argmax(0, keepdim=True)
        
        # Probability threshold: set low-confidence to background
        seg_pred[seg_logits_i.max(0, keepdim=True)[0] < self.prob_thd] = 0
        
        # Morphological closing: fill small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        seg_pred = torch.from_numpy(
            cv2.morphologyEx(
                seg_pred.squeeze().cpu().numpy().astype(np.uint8),
                cv2.MORPH_CLOSE,
                kernel
            )
        ).unsqueeze(0)
        
        # Store results
        data_samples[i].set_data({
            'seg_logits': PixelData(**{'data': seg_logits_i}),
            'pred_sem_seg': PixelData(**{'data': seg_pred})
        })
    
    return data_samples
```

**Parameters**:
- `logit_scale`: COCO-Stuff=40, COCO-Object=50, VOC=60 (sharper at higher values)
- `prob_thd`: COCO-Object=0.1, VOC=0.1 (suppress uncertain predictions)
- `area_thd`: VOC=0.1 (suppress small false positives)

---

## 8. CLASS NAME PARSING (itaclip_segmentor.py, lines 323-334)

**Handles multiple text variants per visual class:**

```python
def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)
    
    class_names, class_indices = [], []
    for idx in range(num_cls):
        # Split by comma to get variants
        names_i = name_sets[idx].split(', ')
        class_names += names_i
        # Track which visual class each variant belongs to
        class_indices += [idx for _ in range(len(names_i))]
    
    # Clean newlines
    class_names = [item.replace('\n', '') for item in class_names]
    
    return class_names, class_indices
```

**Example from cls_coco_stuff.txt**:
```
person, person in shirt, person in jeans, person in dress, ...
bicycle
car
...
```

Results in:
- class_names = ['person', 'person in shirt', 'person in jeans', ...]
- class_indices = [0, 0, 0, ..., 1, 2, ...]

This allows multiple text embeddings per visual class.

---

## 9. LLM DEFINITION GENERATION (llama3_definition_generation.py, lines 39-72)

**Generating auxiliary texts using LLaMa 3-8B-Instruct:**

```python
pipeline = transformers.pipeline(
    "text-generation",
    model_id="meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    token=access_token
)

for class_name in name_sets:
    messages = [
        {
            "role": "system",
            "content": (
                "Give a brief definition of prompted word like given example "
                "definitions: house >= a building that people, usually one "
                "family, live in; car >= a road vehicle with an engine, four "
                "wheels, and seats for a small number of people; (no more than "
                "50 words, do not use extra words other than the definition of "
                "given word)"
            )
        },
        {"role": "user", "content": f"{class_name} >="}
    ]
    
    prompt = pipeline.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=[
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ],
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    
    # Extract and save generated definition
    definition = outputs[0]["generated_text"][len(prompt):]
    with open(txt_path, 'a') as file:
        file.write(f'{class_name} >={definition}\n')
```

**Expected format** (coco_stuff_definitions.txt):
```
person >= a human being, a living individual.
bicycle >= a vehicle with two wheels, powered by pedaling...
car >= a road vehicle with an engine, four wheels...
```

---

## 10. DENSE PREDICTION PIPELINE (itaclip_segmentor.py, lines 112-172)

**Complete flow from image to per-pixel class scores:**

```python
def forward_feature(self, img, text_features, logit_size=None):
    # 1. Encode image to get all patch embeddings
    image_features = self.net.encode_image(
        img,
        return_all=True,              # Get all patches, not just CLS
        attn_self=self.attn_self,     # Use self-attention variant
        device=self.device
    )
    
    # 2. Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    
    # 3. Remove CLS token: [B, 1+HW, D] -> [B, HW, D]
    image_features = image_features[:, 1:]
    
    # 4. Compute class logits via dot product
    # [B, HW, D] @ [D, C] -> [B, HW, C] -> [B, C, HW]
    logits = image_features @ text_features.T
    
    # 5. Reshape to spatial grid
    patch_size = self.net.visual.patch_size  # 16 for ViT-B/16
    w = img.shape[-2] // patch_size
    h = img.shape[-1] // patch_size
    out_dim = logits.shape[-1]
    logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h)
    # Now: [B, C, patch_h, patch_w] e.g., [B, C, 14, 14]
    
    # 6. Upsample to original resolution via bilinear interpolation
    if logit_size is None:
        logits = nn.functional.interpolate(
            logits,
            size=img.shape[-2:],  # [H, W]
            mode='bilinear'
        )
    else:
        logits = nn.functional.interpolate(
            logits,
            size=logit_size,
            mode='bilinear'
        )
    
    # 7. Optional image engineering ensemble
    if self.img_engineering:
        flip_logits = self.get_flipped_logits(
            flip_list,
            self.flip_transforms,
            size=img.shape[-2:],
            w=w, h=h,
            out_dim=out_dim
        )
        # Weighted average with original
        logits = (self.img_eng_coefficient) * logits + \
                (1 - self.img_eng_coefficient) * flip_logits
    
    return logits  # [B, C, H, W]
```

**Data flow**:
```
Image [B, 3, H, W]
  ↓
CLIP encoder → patches [B, 14×14, 512] + CLS [B, 512]
  ↓
Remove CLS → [B, 196, 512]
  ↓
Dot with text_features [512, C] → [B, 196, C]
  ↓
Reshape → [B, C, 14, 14]
  ↓
Bilinear upsample → [B, C, H, W] ← Per-pixel class logits
```

---

## Summary: Implementing for SAM2+CLIP

1. **Architecture**: Fuse SAM2 decoder features at multiple levels (like intermediate ViT attention)
2. **Image Engineering**: Apply similar augmentations (grayscale, blur, flips) at mask feature level
3. **Text Enhancement**: Generate definitions/synonyms for all object classes, not just embeddings
4. **Mask Scoring**: Similar bilinear upsampling from patch-level to full resolution
5. **Postprocessing**: Apply morphological ops + area thresholds + probability thresholds
6. **Inference**: Use sliding window if needed, with proper overlap averaging

