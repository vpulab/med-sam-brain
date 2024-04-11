# SAM Adaptation for mp-MRI Brain Tumor Segmentation

We address in our study the primary challenge of adapting SAM for mp-MRI brain scans, which typically encompass multiple MRI modalities not fully utilized by standard three-channel vision models. We demonstrate that leveraging all available MRI modalities achieves superior performance compared to the standard mechanism of repeating a MRI scan to fit the input embedding. Furthermore, we incorporate Parameter Efficient Fine-Tuning (PEFT) through LoRA blocks to solve the lack of SAM's medical specific knowledge.
