# ac4C Introduction

> N4-acetylcytidine (ac4C) is an essential component of the epitranscriptome, which plays a crucial role in regulating mRNA expression by enhancing stability and translation efficiency. A growing body of evidence suggests that ac4C can help ensure correct codon reads during translation, thereby improving translation efficiency,enhancing mRNA stability, and regulating gene expression. For example, ac4C has recently been discovered in the mRNA of humans and archaea,and it plays an important role in regulating RNA stability, RNA translation, and heat stress adaptation. In addition,recent studies have confirmed that ac4c is associated with a variety of cancers, leukemia, and myeloma in mRNA in humans and mammals.The RNA level of ac4C modification changes in plants under low temperature stress, and this study suggests that ac4C modification may be involved in the adaptation response of plants to low temperature. When plants are subjected to salinity stress or drought stress, the acetylation level of RNA changes set.
# Use tutorial
##  Training the model

```shell 
python train_iac4C-GRU.py --device cuda:1 --train_file ./datalist/true-train.csv --test_file ./datalist/true-test.csv --output_pth ./mix_module.pth --output_csv ./tra.csv 

```
##  Predictive model score

```shell
python Gru-iac4C_predict --input_fa ./datalist/seq.fa --result_csv results.csv --result_json results.json --module ./mix_module.pth


## In order to make it easy for you to use, we have built a website that can be used directly

```shell
https://newbreeding.ctgu.edu.cn/
```
