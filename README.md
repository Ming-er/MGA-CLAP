# MGA-CLAP

The official implementation of "Advancing Multi-grained Alignment for Contrastive Language-Audio Pre-training" (accepted by **ACM MM 2024, oral, top 3.97% among total submissions**). 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



## Environment

Our code implementation is based on the retrieval part of [WavCaps](https://github.com/XinhaoMei/WavCaps/tree/master). Similar to WavCaps, MGA-CLAP is lightweight and can be reproduced on a single RTX 3090 with 24 GB RAM. As for the environment, one can build it following WavCaps. 



## Example

We provide a well-trained model checkpoint, which can be accessed through Google drive (https://drive.google.com/file/d/1RWTuVMEPy-L0uK6WYIX2wwxHjD1YSQFz/view?usp=drive_link).  One can download it, and put it in the <u>pretrained_models/models</u>.

We provide an example to show how to extract frame features and frame-caption correspondence in <u>example.py</u>, remember to modify the checkpoint path in <u>settings/inference_example.yaml</u>. The key code is listed as follow,

```python
# get fine-grained word_level embeddings
_, word_embeds, attn_mask = model.encode_text(classes) 
# aggregate word_level embeddings to sentence_level by shared codebook
text_embeds = model.msc(word_embeds, model.codebook, attn_mask) 
# get fine-grained frame_level embeddings
_, frame_embeds = model.encode_audio(audio_time_series.unsqueeze(0)) 
# aggregate frame_level embeddings to clip_level by shared codebook
audio_embeds = model.msc(frame_embeds, model.codebook) 
```



## Evaluation

We provide a well-trained model checkpoint model.pt, which can be accessed through Google drive.  One can download it, and put it in the <u>pretrained_models/models</u>.

We show several evaluation examples of fine-grained and coarse-grained tasks

- Sound event detection

  see <u>zero_shot_sed.py</u>, it provides the inference code for [audioset_strong_eval](https://research.google.com/audioset/download_strong.html) dataset, remember to modify the data and checkpoint path in <u>settings/inference_sed.yaml</u>

- Text-to-audio grounding

  see <u>zero_shot_grounding.py</u>, it provides the inference code for [audio_grounding](https://github.com/wsntxxn/TextToAudioGrounding) dataset, remember to modify the data and checkpoint path in <u>settings/inference_grounding.yaml</u>

- Audio classification

  see <u>zero_shot_clas.py</u>, it provides the inference code for esc-50, urbansound8k and vggsound dataset, remember to modify the data and checkpoint path in <u>settings/inference_cls.yaml</u>



## Training

- Prepare the WavCaps dataset as [WavCaps](https://github.com/XinhaoMei/WavCaps/tree/master). 

- Run the code

  The training settings are given in <u>settings/pretrain.yaml</u>. Simply run the code by

  ```
  python pretrain.py
  ```

- Highlights of our novel designs

  - Modality-shared codebook
  
    can be found in <u>models/ase_model.py</u>
  
  - Locality-aware block
  
    can be found in <u>models/hts_at.py</u>
  
  - Hard-negative guided contrastive loss

​	      can be found in <u>tools/losses.py</u>




## Citation

If you want to cite this paper:

```
Li Y, Guo Z, Wang X, et al. Advancing Multi-grained Alignment for Contrastive Language-Audio Pre-training[J]. arXiv preprint arXiv:2408.07919, 2024.
```
