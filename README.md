# StackPropagation-SLU-TF

This repo is a tensorflow implementation for the paper `A Stack-Propagation Framework with Token-Level Intent Detection for Spoken Language Understanding`

This is the cite information of the paper. And you can find this paper in [ACLweb](https://www.aclweb.org/anthology/D19-1214/) or [arXiv](https://arxiv.org/abs/1909.02188)

> Qin, L., Che, W., Li, Y., Wen, H., & Liu, T. (2019, November). 
> A Stack-Propagation Framework with Token-Level Intent Detection for Spoken Language Understanding. 
> In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP) (pp. 2078-2087).

The code that published by the author of the paper is implemented by PyTorch and you can find the code in the[StackPropagation-SLU](https://github.com/LeePleased/StackPropagation-SLU).

# Use:

- This code just support the snips dataset now, if you want to use other dataset, you need to convert the data like format in the folder data/snips.
- The alphabet is use for the snips dataset now, if you want to use other dataset, you need to add or change the information into the floder alphabet.
- The size of the alphabet determine some parameters of the model, so if you change the alphabet, you need to change the value of arguments `num_words`, `num_intents`, `num_slots`

# Environment
I use TF1 to implement this model and you can run it with tf1.13-1.15, the tf with other versions haven't been verified the usability and you can try it if it is necessary for you.

The code don't need any other package without TensorFlow and Numpy. It is a **procedure-oriented** style now, maybe I will update it to object-oriented style in future. 
