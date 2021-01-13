#   StackPropagation-SLU
This repo is a tensorflow implementation for the paper `A Stack-Propagation Framework with Token-Level Intent Detection for Spoken Language Understanding`  

    [1] QIN L, CHE W, LI Y等. A Stack-Propagation Framework with Token-Level Intent Detection for Spoken Language Understanding[J/OL]. 2019(2): 2078–2087. . https://arxiv.org/pdf/1909.02188.pdf. DOI:10.18653/v1/d19-1214.

It's just a messy version so far, which is the initial one. I will refactor later on.

#   Use:
*   This code just support the snips dataset now, if you want to use other dataset, you need to convert the data like format in the folder data/snips.
*   The alphabet is use for the snips dataset now, if you want to use other dataset, you need to add or change the information into the floder alphabet.
*   The size of the alphabet determine some parameters of the model, so if you change the alphabet, you need yo change the parameters of the function `create_model()` in `model.py` which determine the number of the different words, the number of intents and the number of slots, it will be called twice in the python script.
