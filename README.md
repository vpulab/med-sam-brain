# SAM Adaptation for mp-MRI Brain Tumor Segmentation

This is the repository of our accepted CVPR-2024 paper for [DEF-AI-MIA Workshop](https://ai-medical-image-analysis.github.io/4th/). 

We address in our study the primary challenge of adapting SAM for mp-MRI brain scans, which typically encompass multiple MRI modalities not fully utilized by standard three-channel vision models. We demonstrate that leveraging all available MRI modalities achieves superior performance compared to the standard mechanism of repeating a MRI scan to fit the input embedding. Furthermore, we incorporate Parameter Efficient Fine-Tuning (PEFT) through LoRA blocks to solve the lack of SAM's medical specific knowledge.

### *Pipeline Overview*

![Captura de pantalla 2024-04-11 a las 18 23 38](https://github.com/vpulab/med-sam-brain/assets/96308828/4b82d250-e471-4052-89e4-e428e2b49048)

We propose to adapt the encoder by: 1) accounting for all the mp-MRI volumetric image modalities; and 2) specifically tuning of the encoder to retain the open-world segmentation capabilities of SAM.


### *Proposed Encoder*

![Captura de pantalla 2024-04-11 a las 18 25 17](https://github.com/vpulab/med-sam-brain/assets/96308828/13217e7d-71ad-4398-8ff8-218aece39365)

We propose to modify the patch embedding layer, so that it accounts for the all the MRI modalities, allowing for a seamless integration of the information. Then, we employ LoRAs to tune Multi Layer Perceptron blocks (MLP) and Attention (Q,K,V embedding) layers of the 
transformer blocks.


### *Cite:*

```
@inproceedings{cdiana2024med-sam-brain,
  title={How SAM Perceives Different mp-MRI Brain Tumor Domains?},
  author={Diana-Albelda, Cecilia and Alcover-Couso, Roberto and García-Martín, Álvaro and Bescos, Jesus},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4959--4970},
  year={2024}
}
```
