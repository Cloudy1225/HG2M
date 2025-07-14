# Teaching MLPs to Master Heterogeneous Graph-Structured Knowledge for Efficient and Accurate Inference

Official Implementation of Teaching MLPs to Master Heterogeneous Graph-Structured Knowledge for Efficient and Accurate Inference.

- Dataset Loader (TMDB, CroVal, ArXiv, and IGBs)
- Two evaluation settings: transductive and inductive
- Various teacher HGNN architectures (RSAGE, RGCN, RGAT, SimpleHGN, ieHGCN) and student MLPs
- Training paradigm for teacher HGNNs and student MLPs



## Getting Started

### Setup Environment

To run the code, please install the following libraries: dgl==2.4.0+cu124, torch==2.4.0, numpy==2.1.3, scipy==1.14.1



### Preparing Datasets

All datasets are available under `data/`.

- `TMDB`, `CroVal`, and `ArXiv` have already been well-organized.
- `IGB` datasets are provided by [IllinoisGraphBenchmark/IGB-Datasets](https://github.com/IllinoisGraphBenchmark/IGB-Datasets).
  1. Download [IGB-549K-19/2K](https://igb-public-awsopen.s3.amazonaws.com/igb-heterogeneous/igb_heterogeneous_tiny.tar.gz) and [IGB-3M-19](https://igb-public-awsopen.s3.amazonaws.com/igb-heterogeneous/igb_heterogeneous_small.tar.gz).
  2. Extract them to `data/igb/tiny/` and `data/igb/small/`, respectively.
  3. Move `data/igb/{tiny|small}/paper_year.npy` to `data/igb/{tiny|small}/processed/paper/paper_year.npy`. (More details about the paper year data are available at this [link](https://github.com/IllinoisGraphBenchmark/IGB-Datasets/issues/41). Here we provide our processed paper year data for convenience.)
- Your favourite datasets: download and add to the `load_data` function in `dataloader.py`.



### Training and Evaluation

To quickly train a teacher model you can run `train_teacher_{tran|ind}.py` by specifying the experiment setting, i.e. teacher model, e.g. `RSAGE`, and dataset, e.g. `TMDB`, as per the example below.

```
python train_teacher_tran.py --model_type RSAGE --dataset TMDB 

python train_teacher_ind.py --model_type RSAGE --dataset TMDB
```

To quickly train a student model with a pretrained teacher you can run `train_student_{tran|ind}.py` or `train_stud_{tran|ind}_plus.py` by specifying the teacher model, and dataset like the example below. Make sure you train the teacher first and have its result stored in the correct path.

```
python train_student_tran.py --teacher RSAGE --dataset TMDB
python train_student_tran_plus.py --teacher RSAGE --dataset TMDB

python train_student_ind.py --teacher RSAGE --dataset TMDB
python train_student_ind_plus.py --teacher RSAGE --dataset TMDB
```



## Acknowledgements

The code is implemented based on [snap-research/graphless-neural-networks](https://github.com/snap-research/graphless-neural-networks).
