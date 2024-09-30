### *Requirements*
CUDA version: 11.8.

torch==1.13.0 ; torchaudio==0.13.0 ; torchvision==0.14.0.

You can check the propper torch version for your CUDA at: https://pytorch.org/get-started/previous-versions/


### *Data acquisition*
Download any BraTS dataset from Synapse following their instructions.

Link: https://www.synapse.org/Synapse:syn51156910/wiki/627000

The dataset folder structure should be as follows:

```
-data
--brats #Adult Glioma Segmentation
--brats_ssa #Subsaharan Glioma Segmentation
--brats_ped #Pediatric Glioma Segmentation
--brats_men #Meningioma Segmentation
```

You can download as much BraTS datasets as you want to use, but all of them should be placed inside the `data` folder following the above specified naming instructions.

### *Download code & Set the environment*
Open a terminal and execute the following commands:


```
git clone https://github.com/vpulab/med-sam-brain/;
cd med-sam-brain;
conda env create -f environment.yml;
conda activate sam_adapt_brain;
```

### *Training & Testing*

**Training**

```
python train.py -net sam -mod sam_lora -exp_name ... -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -b 1 -dataset brats -thd True  -data_path ../data -w 8 -four_chan True 
```

Parameter `mod` can be defined as: `sam_lora` to train LoRA blocks making SAM adapt to the medical domain; or `sam` in case you want to maintain the original SAM architecture. Parameter `four_chan` should be defined as `True` if you want to use all 4 MRI modalities; or `False` if just taking e of them to not train the Patch Embedding Layer. Parameter `dataset` must be defined as any of the names indicated in the 'Data acquisition' section.

*NOTE*: After running the training command, 'sam_vit_b_01ec64.pth' will be downloaded. If pretrained weights are not downloaded propperly, you cand do it manually through [this link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and store it in 'checkpoint/sam/'. The saved model parameters will be placed in the 'logs/' directory.

**Validation**

```
python valid.py -net sam -mod sam_lora -thd True  -dataset brats -weights logs/.../Model/best_dice -sam_ckpt logs/.../Model/best_dice -mode Validation -four_chan True 
```

Parameters `weights` and `sam_ckpt` should be replaced by the directory of the saved model file in 'logs/'.

**NOTE:** In case you don't have enough GPU to execute the training process, you can uncomment the following code lines on `function.py`, which reduces computational cost by taking 4 random slices per volume (the selected slices change each iteration).

```
# If not enough GPU, uncomment the following 3 lines (lines 73-76 and 226-230)
# i_slices = SelectEquiSlices(4, masks)
# imgs = imgs[:,:,:,:,i_slices] 
# masks = masks[:,:,:,:,i_slices]
```
