
# AMR Resources

A  list of Abstract meaning representation (AMR) resources: research papers, code, data, applications, etc. The list is not limited to Abstract meaning representation exclusively. It also includes work highly related to AMR, such as translation and information extraction.

## Table of contents
<!--ts-->
* [AMR Resources](#amr-resources)
   * [Table of contents](#table-of-contents)
   * [Introduction to AMR](#introduction-to-amr)
   * [Papers sorted in chronological order](#papers-sorted-in-chronological-order)
      * [2013](#2013)
      * [2014](#2014)
   * [Papers grouped by category](#papers-grouped-by-category)
      * [Surveys](#surveys)
      * [Evaluation](#evaluation)
      * [AMR parsing approaches](#amr-parsing-approaches)
         * [Transition based](#transition-based)
         * [Seq2graph](#seq2graph)
      * [AMR for downstream applications](#amr-for-downstream-applications)
         * [Machine translation](#machine-translation)
         * [Question answering](#question-answering)
         * [Natural language generation](#natural-language-generation)
         * [Information extraction](#information-extraction)
         * [Human–robot interaction](#humanrobot-interaction)
         * [Multimodal](#multimodal)
         * [Semantic Role Labeling （SRL）](#semantic-role-labeling-srl)
         * [Cross-lingual](#cross-lingual)
         * [Math problems](#math-problems)
         * [Entity Linking](#entity-linking)
         * [Data Augmentation](#data-augmentation)
         * [Code generation](#code-generation)
   * [Data](#data)

<!-- Added by: fansiqi, at: 2021年 9月16日 星期四 10时28分48秒 CST -->

<!--te-->


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
* [*"Renewing and revising SemLink",*](https://aclanthology.org/W13-5503.pdf) -ACL 2013
  Claire Bonial, Kevin Stowe & Martha Palmer
* [*"Unsupervised Induction of Cross-lingual Semantic Relations",*](https://aclanthology.org/D13-1064.pdf) -ACL 2013
  M Lewis, M Steedman

* [*"Textual inference and meaning representation in human robot interaction",*](https://aclanthology.org/W13-3820.pdf) -ACL 2013
  Emanuele Bastianelli Giuseppe Castellucci Danilo Croce Roberto Basili
### 2014

* [*" A discriminative graph-based parser for the abstract meaning representation",*](https://aclanthology.org/P14-1134.pdf) -ACL 2014
  Jeffrey Flanigan Sam Thomson Jaime Carbonell Chris Dyer Noah A. Smith
  
* [*"Large-scale Semantic Parsing without Question-Answer Pairs",*](https://aclanthology.org/Q14-1030.pdf) ACL 2014
  Siva Reddy, Mirella Lapata, Mark Steedman
## Papers grouped by category

### Surveys

* [*"Learning Executable Semantic Parsers for Natural Language Understanding",*](https://cs.stanford.edu/~pliang/papers/executable-cacm2016.pdf)-2016
  Percy Liang

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

AMR体现在知识库的构建和使用上，好的知识结构更有利于更新和应用。

#### Machine translation
* [*"Neural Semantic Parsing by Character-based Translation: Experiments with Abstract Meaning Representations,"*](https://arxiv.org/pdf/1705.09980.pdf) -CLIN 2017
  Rik van Noord, Johan Bos
* [*"Semantic Neural Machine Translation Using AMR",*](https://arxiv.org/pdf/1902.07282.pdf) -ACL 2019
  Linfeng Song, Daniel Gildea, Yue Zhang, Zhiguo Wang and Jinsong Su
* [*"Self-Attention with Structural Position Representations",*](https://arxiv.org/pdf/1909.00383.pdf) - EMNLP 2019
  Xing Wang, Zhaopeng Tu, Longyue Wang, Shuming Shi

#### Question answering
* [*"Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task",*](https://arxiv.org/pdf/1809.08887.pdf) -	EMNLP 2018, Long Paper
  Tao Yu Rui Zhang Kai Yang Michihiro Yasunaga
  
* [*"TypeSQL: Knowledge-based Type-Aware Neural Text-to-SQL Generation",*](https://arxiv.org/pdf/1804.09769.pdf)-	NAACL 2018
  Tao Yu, Zifan Li, Zilin Zhang, Rui Zhang, Dragomir Radev
  
* [*"Leveraging Abstract Meaning Representation for Knowledge Base Question Answering",*](https://aclanthology.org/2021.findings-acl.339.pdf) -ACL2021
  
  Pavan Kapanipathi, Ibrahim Abdelaziz, Srinivas Ravishankar
  
* [*"A Semantic Parsing and Reasoning-Based Approach to Knowledge Base Question Answering",*](https://ojs.aaai.org/index.php/AAAI/article/view/17988) -AAAI 2021

  Ibrahim Abdelaziz


#### Natural language generation
* [*"Toward Abstractive Summarization Using Semantic Representations",*](https://arxiv.org/pdf/1805.10399.pdf) ACL 2018
  Fei Liu, Jeffrey Flanigan, Sam Thomson, Norman Sadeh, Noah A. Smith
  
* [*"A Graph-to-Sequence Model for AMR-to-Text Generation",*](https://arxiv.org/pdf/1805.02473.pdf) ACL 2018
  Linfeng Song, Yue Zhang, Zhiguo Wang, Daniel Gildea
  
* [*"Deep Graph Convolutional Encoders for tructured Data to Text Generation",*](https://arxiv.org/pdf/1810.09995.pdf) [code](https://github.com/diegma/graph-2-text)
  Diego Marcheggiani,Laura Perez-Beltrachini
  
* [*"Structural Neural Encoders for AMR-to-text Generation",*](Marco Damonte, Shay B. Cohen) -NAACL 2019 [code](https://github.com/mdtux89/OpenNMT-py-AMR-to-text)
  Marco Damonte, Shay B. Cohen
  
* [*"Modeling Graph Structure in Transformer for Better AMR-to-Text Generation",*](https://arxiv.org/pdf/1909.00136.pdf) -EMNLP 2019
  Jie Zhu, Junhui Li, Muhua Zhu, Longhua Qian, Min Zhang, Guodong Zhou
  
* [*"Generation from Abstract Meaning Representation using Tree Transducers",*](https://www.cs.cmu.edu/~jgc/publication/flanigantree.pdf) -NAACL 2016  [code](https://github.com/jflanigan/jamr)

  Jeffrey Flanigan,Chris Dyer,Noah A. Smith,Jaime Carbonell
  
* [*"Enhancing AMR-to-Text Generation with Dual Graph Representations",*](https://arxiv.org/pdf/1909.00352.pdf) -EMNLP 2019 [code](https://github.com/UKPLab/emnlp2019-dualgraph)

  Leonardo F. R. Ribeiro, Claire Gardent, Iryna Gurevych

* [*"GPT-too: A language-model-first approach for AMR-to-text generation",*](https://aclanthology.org/2020.acl-main.167.pdf) -ACL 2020

  Manuel Mager, Ramon Fernandez Astudillo, Tahira Naseem, Md Arafat Sultan, Young-Suk Lee, Radu Florian, Salim Roukos

* [*"Investigating Pretrained Language Models for Graph-to-Text Generation"*](https://arxiv.org/pdf/2007.08426.pdf)  -2020 [code](https://github.com/UKPLab/plms-graph2text)

  Leonardo F. R. Ribeiro, Martin Schmitt, Hinrich Schütze, Iryna Gurevych

* [*"Structural Information Preserving for Graph-to-Text Generation",*](https://aclanthology.org/2020.acl-main.712.pdf)", -ACL 2020

  Linfeng Song, Ante Wang, Jinsong Su, Yue Zhang, Kun Xu, Yubin Ge, Dong Yu

#### Information extraction
* [*"Unsupervised Induction of Cross-lingual Semantic Relations",*](https://aclanthology.org/D13-1064.pdf) -ACL 2013 [code](https://github.com/Amazing-J/structural-transformer)
  M Lewis, M Steedman

* [*"Unified Visual-Semantic Embeddings: Bridging Vision and Language With Structured Meaning Representations",*](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_Unified_Visual-Semantic_Embeddings_Bridging_Vision_and_Language_With_Structured_Meaning_CVPR_2019_paper.pdf) -CVPR 2019
  H Wu, J Mao, Y Zhang, Y Jiang
#### Human–robot interaction
* [*"Textual inference and meaning representation in human robot interaction",*](https://aclanthology.org/W13-3820.pdf) -ACL
  Emanuele Bastianelli Giuseppe Castellucci Danilo Croce Roberto Basili
  
* [*"Fine-grained Information Extraction from Biomedical Literature based on Knowledge-enriched Abstract Meaning Representation",*](https://aclanthology.org/2021.acl-long.489.pdf) -ACL 2021

  Zixuan Zhang, Nikolaus Parulian, Heng Ji

#### Multimodal
* [*"Generating semantically precise scene graphs from textual descriptions for improved image retrieval",*](https://aclanthology.org/W15-2812.pdf) -ACL 2015
  Sebastian Schuster, Ranjay Krishna, Angel Chang, Li Fei-Fei, and Christopher D. Manning
  
* [*"Genie: A Generator of Natural Language Semantic Parsers for Virtual Assistant Commands",*](https://arxiv.org/pdf/1904.09020.pdf) -PLDI [code](https://github.com/stanford-oval/genie-toolkit)

  Giovanni Campagna, Silei Xu, Mehrad Moradshahi, Richard Socher, Monica S. Lam

* [*"Cross-media Structured Common Space for Multimedia Event Extraction",*](https://arxiv.org/pdf/2005.02472.pdf) -ACL 2020

  Manling Li, Alireza Zareian, Qi Zeng, Spencer Whitehead, Di Lu, Heng Ji, Shih-Fu Chang

* [*"Bootstrapping Multilingual AMR with Contextual Word Alignments",*](https://arxiv.org/pdf/2102.02189.pdf) -ACL 2021

  Janaki Sheth, Young-Suk Lee, Ramon Fernandez Astudillo

#### Semantic Role Labeling （SRL）
* [*"Question-Answer Driven Semantic Role Labeling:Using Natural Language to Annotate Natural Language",*](https://aclanthology.org/D15-1076.pdf) -EMNLP 2015
  Luheng He Mike Lewis Luke Zettlemoyer
#### Cross-lingual
* [*"Cross-lingual Name Tagging and Linking for 282 Languages",*](https://aclanthology.org/P17-1178.pdf) -ACL 2017
  Xiaoman Pan Boliang Zhang Jonathan May
* [*"The Parallel Meaning Bank: Towards a Multilingual Corpus of Translations Annotated with Compositional Meaning Representations",*](https://arxiv.org/pdf/1702.03964.pdf) -EACL 2017
  Lasha Abzianidze Johannes Bjerva Kilian Evang Hessel Haagsma
#### Math problems
* [*"Automatically Solving Number Word Problems by Semantic Parsing and Reasoning",*](https://aclanthology.org/D15-1135.pdf) -ACL 2015
  Shuming Shi Yuehui Wangn Chin-Yew Lin Xiaojiang Liu1 and Yong Rui
#### Entity Linking
* [*"Unsupervised Entity Linking with Abstract Meaning Representation",*](https://aclanthology.org/N15-1119.pdf) -ACL 2015 [code](https://github.com/panx27/amr-reader)
  Xiaoman Pan Taylor Cassidy Ulf Hermjakob Heng Ji Kevin Knight
#### Data Augmentation
* [*"Richer event description: Integrating event coreference with temporal, causal and bridging annotation",*](https://aclanthology.org/W16-5706.pdf) -ACL 2016
  Tim O’Gorman and Kristin Wright-Bettner and Martha Palmer
#### Code generation
* [*"A Syntactic Neural Model for General-Purpose Code Generation",*](https://arxiv.org/pdf/1704.01696.pdf) -ACL 2017
  Pengcheng Yin, Graham Neubig
* [*"TRANX: A Transition-based Neural Abstract Syntax Parser for Semantic Parsing and Code Generation",*](https://arxiv.org/pdf/1810.02720.pdf) -EMNLP 2018
  Pengcheng Yin, Graham Neubig


## Data

* AMR 2.0 ([LDC2017T10](https://catalog.ldc.upenn.edu/LDC2017T10))

* AMR 3.0([LDC2020T02](https://catalog.ldc.upenn.edu/LDC2020T02))

* CAMR 2.0 由布兰迪斯大学和南京师范大学联合标注的[中文抽象意义表示语料库2.0](https://catalog.ldc.upenn.edu/LDC2021T13)

  
