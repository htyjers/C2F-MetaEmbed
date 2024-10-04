# C2F-MetaEmbed
### [Paper]() | [BibTex]()

This repository is the official code for the paper "Coarse-to-Fine Lightweight Meta-Embedding for ID-Based Recommendation" by Yang Wang <a href="mailto:yangwang@hfut.edu.cn">
    <img src="https://img.shields.io/badge/-Email-red?style=flat-square&logo=gmail&logoColor=white"> , Haipeng Liu <a href="mailto:hpliu_hfut@hotmail.com">
    <img src="https://img.shields.io/badge/-Email-red?style=flat-square&logo=gmail&logoColor=white"> , Zeqian Yi, Biao Qian, Meng Wang (corresponding author). 


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

## 2. Dataset

## 3. The Coarse Training Stage

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
Python3 
```

## 4. The Fine Training Stage

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
Python3 
```

## 5. Acknowledgments
This implementation is based on / inspired by:
* [https://github.com/xurong-liang/LEGCF](https://github.com/xurong-liang/LEGCF) (LEGCF)
  
</details>

<details open>
<summary><h1>📊 Experimental Results</h1></summary>

## 1. Quantitative analysis
| **Log**               | **Gowalla**  | **Yelp2020** | **Amazon-book** |
|----------------------------------|--------------|--------------|-----------------|
| **The Coarse Training Stage**        |[Hyperparameter]() & [Results]()|[Hyperparameter]() & [Results]()|[Hyperparameter]() & [Results]()|
| **The Fine Training Stage**        |[Hyperparameter]() & [Results]()|[Hyperparameter]() & [Results]()|[Hyperparameter]() & [Results]()|


## 2. Ablation study

</details>


<details open>
<summary><h1>🔖 Citation</h1></summary>

</details>
