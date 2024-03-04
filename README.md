# kNN-BioEL

The code and dataset of paper [***Improving Biomedical Entity Linking with Retrieval-enhanced Learning***](https://arxiv.org/abs/2312.09806) in Proceedings of ICASSSP 2024.

### Env

```shell
conda create -n bioel python==3.10.10
conda activate bioel
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117
```

### Base Model

Please [download](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext) the baseline model SapBERT in advance and rename it as `SapBERT-from-PubMedBERT-fulltext`, then place it in the `models` folder.

### Training 

You can execute the following command to train each dataset, where `0` represents the ID of the GPU device.

```shell
bash scripts/train_cometa.sh 0
bash scripts/train_aap.sh 0
bash scripts/train_ncbi.sh 0
bash scripts/train_bc5cdr.sh 0
```

You can also skip the training and directly download our pre-trained model from [here](https://pan.baidu.com/share/init?surl=pObrESVxskpjQgVZytAozQ&pwd=793f) (password `793f`) or [here](https://drive.google.com/drive/folders/1Z6i-Qfpw_8gYotJlW35Cp69Z5YsFkYUK?usp=sharing). Please place the downloaded weights for the four datasets in the `save` directory, and the directory is organized as follows:

```shell
save
|--cometa
|--aap
|--ncbi
|--bc5cdr
```

### Evaluation

After training the model or downloading the pre-trained weights, execute the following command to evaluate kNN-BioEL.

```shell
bash scripts/eval_knn_cometa.sh 0
bash scripts/eval_knn_aap.sh 0
bash scripts/eval_knn_ncbi.sh 0
bash scripts/eval_knn_bc5cdr.sh 0
```

### Citation

If you use this model or code, please cite it as follows:

```shell
@article{lin2023improving,
  title={Improving Biomedical Entity Linking with Retrieval-enhanced Learning},
  author={Lin, Zhenxi and Zhang, Ziheng and Wu, Xian and Zheng, Yefeng},
  journal={arXiv preprint arXiv:2312.09806},
  year={2023}
}
```

