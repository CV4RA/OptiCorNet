# OptiCorNet: Optimizing Sequence-Based Context Correlation for Visual Place Recognition



## Setup
### Conda
```bash
conda create -n seqnet numpy pytorch=1.8.0 torchvision tqdm scikit-learn faiss tensorboardx h5py -c pytorch -c conda-forge
```

## Run

### Train
To train sequential descriptors through SeqNet on the Nordland dataset:
```python
python main.py --mode train --pooling dsdnet --dataset nordland-sw --seqL 10 --w 5 --outDims 4096 --expName "w5"
```
or the Oxford dataset (set `--dataset oxford-pnv` for pointnetvlad-like data split as described in the [CVPR 2021 Workshop paper](https://arxiv.org/abs/2106.11481)):
```python
python main.py --mode train --pooling dsdnet --dataset oxford-v1.0 --seqL 5 --w 3 --outDims 4096 --expName "w3"
```
or the MSLS dataset (specifying `--msls_trainCity` and `--msls_valCity` as default values):
```python
python main.py --mode train --pooling dsdnet --dataset msls --msls_trainCity melbourne --msls_valCity austin --seqL 5 --w 3 --outDims 4096 --expName "msls_w3"
```

To train transformed single descriptors through dsdnet:
```python
python main.py --mode train --pooling dsdnet --dataset nordland-sw --seqL 1 --w 1 --outDims 4096 --expName "w1"
```

### Test
On the Nordland dataset:
```python
python main.py --mode test --pooling dsdnet --dataset nordland-sf --seqL 5 --split test --resume ./data/runs/Jun03_15-22-44_l10_w5/ 
```
On the MSLS dataset (can change `--msls_valCity` to `melbourne` or `austin` too):
```python
python main.py --mode test --pooling dsdnet --dataset msls --msls_valCity amman --seqL 5 --split test --resume ./data/runs/<modelName>/
```
  
## Acknowledgement
The code in this repository is based on [seqNet](https://github.com/oravus/seqNet). Thanks to for his contributions to this code during the development of our project [OptiCorNet](https://github.com/CV4RA/OptiCorNet.git).
