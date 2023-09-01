# QABot

## **一、项目简介**

本项目为基于Ascend: 1*Ascend910与mindspore-ascend 1.10.0的使用Bert进行全模型微调的问答机器人，其可以实现输入一句英文问题以及一个句子，从而获得该句子是否为问题答案的分类。

- **代码目录结构说明**
  
    ```
    |─QABot
        ├─ours              # 训练数据集
            ├─WikiQA-dev.tsv
            ├─WikiQA-test.tsv
            ├─WikiQA-train.tsv
        ├─QABot.ipynb       # 实验报告  
        ├─train.py          # 训练模型
        ├─predict.py        # 推理QA
        ├─README.md         # 使用指南
        └─requirements.txt  # 依赖文件
    ```
    
- 自验结果：在训练集上达到90.88%，在验证集上达到90.84%
- 自验环境：
    - Ascend: 1*Ascend910|ARM: 24核 96GB
    - python 3.7

## **二、运行代码**

1. 下载测试数据集:
   
    本项目采用的数据集为开放式问答数据集WikiQA。WikiQA使用Bing查询日志作为问题源，每个问题都链接到一个可能有答案的维基百科页面，页面的摘要部分提供了关于这个问题的重要信息，WikiQA使用其中的句子作为问题的候选答案。数据集中共包括3047个问题和29258个句子。数据可以从[这里](https://work.datafountain.cn/forum?id=121&type=2&source=1)进行下载。原始数据存在一些问题没有答案，已进行初步清洗，数据存放在ours/文件夹下
    
2. 使用以下指令进行第三方库的安装：
   
    ```
    pip install requirements.txt
    ```
    
3. 调整训练超参数，本实验中用的参数为
   
    ```
    batch size = 32
    epoch = 5
    learning rate = 2e-5
    loss function = CrossEntropyLoss
    optimizer = Adam
    ```
    
4. 训练
   
    进入项目文件夹下，使用`python train.py`进行训练
    
5. 推理
    - 进入项目文件夹下，使用`python predict.py`进行推理
    - 有两个模式：单句模式”single“与批量处理模式”file“进行判断

## 三、实验报告

在`qabot.ipynb`中有较为详细的实验报告，可供参考 

## **四、参考资料**

- 基于Bert实现知识库问答：[https://work.datafountain.cn/forum?id=121&type=2&source=1](https://work.datafountain.cn/forum?id=121&type=2&source=1)
- MindNLP开源地址：h[ttps://openi.pcl.ac.cn/lvyufeng/mindnlp](https://openi.pcl.ac.cn/lvyufeng/mindnlp)
- MindNLP文档：[https://mindnlp.cqu.ai/en/latest/](https://mindnlp.cqu.ai/en/latest/)
- 基于GPT2与mindspore的总结项目：[https://github.com/mindspore-lab/mindnlp/blob/master/examples/summarization/gpt2_summarization.ipynb](https://github.com/mindspore-lab/mindnlp/blob/master/examples/summarization/gpt2_summarization.ipynb)
- 基于Bert与mindnlp的情绪分类任务：[https://developer.huaweicloud.com/develop/aigallery/notebook/detail?id=e486c037-76ae-415b-90a7-7766ea189982](https://developer.huaweicloud.com/develop/aigallery/notebook/detail?id=e486c037-76ae-415b-90a7-7766ea189982)