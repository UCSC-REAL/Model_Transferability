# Model Transferability with Responsive Decision Subjects
This repository accompanies the paper Model Transferability with Responsive Decision Subjects accepted by ICML 2023 -- [Yatong Chen](https://github.com/YatongChen/), [Zeyu Tang](https://zeyu.one/), [Kun Zhang](https://www.andrew.cmu.edu/user/kunz1/), [Yang Liu](https://yliuu.com/).

# Abstract: 

Given an algorithmic predictor that is accurate on some source population consisting of strategic human decision subjects, will it remain accurate if the population _respond_ to it?
In our setting, an agent or a user corresponds to a sample $(X,Y)$ drawn from a distribution $\mathcal{D}$ and will face a model $h$ and its classification result $h(X)$. Agents can modify $X$ to adapt to $h$, which will incur a distribution shift on $(X,Y)$.  
    Our formulation is motivated by applications where the deployed machine learning models are subjected to human agents, and will ultimately face _responsive_ and _interactive_ data distributions. 
    We formalize the discussions of the transferability of a model by studying how the performance of the model trained on the available source distribution (data) would translate to the performance on its _induced_ domain. 
    We provide both upper bounds for the performance gap due to the induced domain shift, as well as lower bounds for the trade-offs that a classifier has to suffer on either the source training distribution or the induced target distribution. We provide further instantiated analysis for two popular domain adaptation settings, including _covariate shift_ and _target shift_.


# Citation

If you want to cite our paper, please cite the following format:

```
@article{chen2023model,
  title={Model Transferability With Responsive Decision Subjects},
  author={Chen, Yatong and Tang, Zeyu and Zhang, Kun and Liu, Yang},
  booktitle={International Conference on Machine Learning},
  organization={PMLR}
  year={2023}
}
```
