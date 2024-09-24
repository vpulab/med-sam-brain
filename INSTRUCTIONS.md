**Requirements**
CUDA 11.8
torch==1.13.0 ; torchaudio==0.13.0 ; torchvision==0.14.0

**Data acquisition**
Download any BraTS dataset from Synapse following their instructions.
Link: https://www.synapse.org/Synapse:syn51156910/wiki/627000

**Code & Environment**
Open a terminal and execute the following commands:
```
git clone https://github.com/vpulab/med-sam-brain/;
cd med-sam-brain;
conda env create -f environment.yml;
conda activate sam_adapt_brain;
```
