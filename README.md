# C2F-MetaEmbed
### [Paper]() | [BibTex]()

This repository is the official code for the paper "Coarse-to-Fine Lightweight Meta-Embedding for ID-Based Recommendation" by Yang Wang (<a href="mailto:yangwang@hfut.edu.cn">Email</a>), Haipeng Liu (<a href="mailto:hpliu_hfut@hotmail.com">Email</a>), Zeqian Yi, Biao Qian, and Meng Wang (corresponding author).

## 📚 Table of Contents

- [📖 Introduction]()
- [🌟 C2F-MetaEmbed]()
- [📊 Experimental Results]()
- [🔖 Citation]()


<details open>
<summary><h1>📖 Introduction</h1></summary>

</details>


<details open>
<summary><h1>🌟 C2F-MetaEmbed </h1></summary>

## 1. Dependencies
* OS: Ubuntu 20.04.6
* nvidia :
	- cuda: 12.3
	- cudnn: 8.5.0
* python3
* pytorch >= 1.13.0
* Python packages:
  ```bash
  pip install -r requirements.txt
  ```

## 2. Dataset 
Train and test sets of Gowalla, Yelp2020 and Amazon-book are located in [here](https://pan.baidu.com/s/1TUeNaT6_wioDBWwhIswgfg?pwd=f3vp).

## 3. The Coarse-Grained Training Stage

### Implementation Details
* We followed the [LEGCF](https://github.com/xurong-liang/LEGCF) framework to set most of the hyperparameters during the coarse training stage, with **bold font** indicating the parameters with different values.
* More details are provided in the later part of the quantitative analysis.

| **Hyperparameter**               | **Gowalla**  | **Yelp2020** | **Amazon-book** |
|----------------------------------|--------------|--------------|-----------------|
| **Sign_ft**                      | **0**       | **0**        | **0**           |
| **Clusters (c)**                 | **300**     | **300**      | **300**         |
| **Evaluate Frequency**           | **5**       | **5**        | **5**           |
| Assignment Update Frequency (m)  | every epoch  | every epoch  | every epoch      |
| GCN Layers                       | 3            | 4            | 4               |
| L2 Penalty Factor                | 5            | 5            | 5               |
| Learning Rate (lr)               | 1e-3         | 1e-3         | 1e-3            |
| Composition Coarse Embeddings per Entity (t) | 2       | 2            | 2               |
| Initial Anchor Weight (w\*)      | 0.5          | 0.6          | 0.9             |
| ...      |...          | ...          | ...             |

### Run the following command

```python
Python3 engine.py --dataset_name gowalla --num_clusters 300 --num_composition_centroid 2 --device_id 0
```

## 4. The Fine-Grained Training Stage

### Implementation Details
* **Bold font** indicating the parameters that differ from those used in the coarse stage.
* More details are provided in the later part of the quantitative analysis.
  
| **Hyperparameter**               | **Gowalla**  | **Yelp2020** | **Amazon-book** |
|----------------------------------|--------------|--------------|-----------------|
| **Sign_ft**                                        | **1**        | **1**        | **1**           |
| **Clusters (c)**                                   | **100**      | **100**      | **100**         |
| **Evaluate Frequency**                             | **5**        | **5**        | **5**           |
| **Learning Rate (lr)**                             | **3e-3**     | **3e-3**     | **3e-3**        |
| **Composition Fine Embeddings per Entity (ft)**     | **4**        | **4**        | **4**           |
| **Initial Anchor Weight (w\*)**                    | **0.5**      | **0.5**      | **0.5**         |
| **Components of SparsePCA**                         | **80**       | **80**       | **80**          |
| **Soft Thresholding**                               | **1**        | **1**        | **1**           |
| **Weight parameter**                                | **0.2**      | **0.2**      | **0.2**         |
| Assignment Update Frequency (m)  | every epoch  | every epoch  | every epoch      |
| GCN Layers                       | 3            | 4            | 4               |
| L2 Penalty Factor                | 5            | 5            | 5               |
| ...      |...          | ...          | ...             |


### Run the following command

```python
Python3 engine.py --dataset_name gowalla --num_clusters 300 --num_composition_centroid 2 --device_id 0
```

## 5. Acknowledgments
This implementation is based on / inspired by:
* [https://github.com/xurong-liang/LEGCF](https://github.com/xurong-liang/LEGCF) (LEGCF)
  
</details>

<details open>
<summary><h1>📊 Experimental Results</h1></summary>

## Quantitative analysis
| **Log**               | **Gowalla**  | **Yelp2020** | **Amazon-book** |
|----------------------------------|--------------|--------------|-----------------|
| **Coarse-Grained Training Stage**        |[Hyperparameter&Results](https://github.com/htyjers/C2F-MetaEmbed/tree/main/result/gowalla/Coarse-grained%20Training%20Stage)|[Hyperparameter&Results](https://github.com/htyjers/C2F-MetaEmbed/tree/main/result/yelp2020/Coarse-grained%20Training%20Stage)|[Hyperparameter&Results]()
| **Fine-Grained Training Stage**        |[Hyperparameter&Results](https://github.com/htyjers/C2F-MetaEmbed/tree/main/result/gowalla/Fine-grained%20Training%20Stage)|[Hyperparameter&Results](https://github.com/htyjers/C2F-MetaEmbed/tree/main/result/yelp2020/Fine-grained%20Training%20Stage)|[Hyperparameter&Results]()

## Ablation study about space complexity
![](result/as.png)
</details>


<details open>
<summary><h1>🔖 Citation</h1></summary>

</details>
