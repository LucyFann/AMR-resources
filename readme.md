# AMR Resources

A  list of Abstract meaning representation (AMR) resources: research papers, code, data, applications, etc. The list is not limited to Abstract meaning representation exclusively. It also includes work highly related to AMR, such as translation and information extraction.

## Table of contents

[toc]

## Introduction to AMR

抽象意义表示（Abstract Meaning Representation，AMR）是一种将句子的意义（时间地点谁对谁怎样地做了什么）表示为以概念为节点的单源有向无环图的语言学框架。AMR正在引起学术界越来越广泛的关注，已经涌现了许多利用AMR进行机器翻译、QA、关系提取等应用的工作。

Suppose we have the following input sentence:

    The boy wants the girl to believe him.

AMR aims to make the following format:

    (w / want-01
       :ARG0 (b / boy)
       :ARG1 (b2 / believe-01
            :ARG0 (g / girl)
            :ARG1 b))

## Papers sorted in chronological order

### 2013

* *["Abstract Meaning Representation for Sembanking"](https://aclanthology.org/W13-2322.pdf)* - ACL 2013

  Laura Banarescu, Claire Bonial, Shu Cai

### 2014

* 


### 2015

* 


### 2016

* 

### 2017

* 


### 2018

* 


### 2019

* 


### 2020

* 


### 2021

* 



## Papers grouped by category

### Surveys



### Evaluation

smatch metric is developed for comparing one AMR against another. It computes an f-score over the semantic triples inside AMRs.

* *["Smatch: an Evaluation Metric for Semantic Feature Structures,"](https://amr.isi.edu/smatch-13.pdf)* ACL 2013 ([smatch code](https://github.com/snowblink14/smatch) ,[demo](https://amr.isi.edu/eval/smatch/compare.html))

  S. Cai and K. Knight

### AMR parsing approaches

#### Transition based

* [*"A Transition-based Algorithm for AMR Parsing",*](https://aclanthology.org/N15-1040) ACL 2015

* *["Transition-Based Dependency Parsing with Stack Long Short-Term Memory,"](http://arxiv.org/abs/1505.08075)* ACL 2015 [code]()

  Chris Dyer, Miguel Ballesteros, Wang Ling, Austin Matthews, Noah A. Smith

* [*"CAMR at SemEval-2016 Task 8: An Extended Transition-based AMR Parser,"*](https://aclanthology.org/S16-1181) ACL 2016

  Chuan Wang, Sameer Pradhan, Xiaoman Pan, Heng Ji, Nianwen Xue

* [*"Transition-based Parsing with Stack-Transformers,"*](https://aclanthology.org/2020.findings-emnlp.89) ACL 2020

  Ramón Fernandez Astudillo, Miguel Ballesteros, Tahira Naseem, Austin Blodgett, Radu Florian

#### Seq2graph

* [*"Deep biaffine attention for neural dependency parsing",*](http://arxiv.org/abs/1611.01734) ACL 2017 [code](https://github.com/tdozat/Parser-v1)[code](https://github.com/yzhangcs/parser)

  Timothy Dozat, Christopher D. Manning

* [*"AMR Parsing via Graph-Sequence Iterative Inference,"*](http://arxiv.org/abs/2004.05572)  ACL 2020 [code](https://github.com/jcyk/AMR-gs)

  Deng Cai, Wai Lam

* [*"Levi Graph AMR Parser using Heterogeneous Attention",*](http://arxiv.org/abs/2107.04152) ACL 2021 [code](https://github.com/emorynlp/levi-graph-amr-parser)

  Han He, Jinho D. Choi

### AMR for downstream applications

AMR graph output has been shown to be a useful input for many downstream tasks. In this section, several downstream tasks that benefited from AMR output are listed. 

#### Machine translation

#### QA

#### NLG


## Data

* AMR 2.0 ([LDC2017T10](https://catalog.ldc.upenn.edu/LDC2017T10))

* AMR 3.0([LDC2020T02](https://catalog.ldc.upenn.edu/LDC2020T02))

* CAMR 2.0 由布兰迪斯大学和南京师范大学联合标注的[中文抽象意义表示语料库2.0](https://catalog.ldc.upenn.edu/LDC2021T13)

  
