# Ghost

### Lil BERT will be your Ghost Writer: 

I'm planning on fine tuning [BERT](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270) on rap lyrics and calling it Lil BERT. The idea is to make an interactive, mad-lib style poetry generator where the user "asks" for words with [MASK] tokens.

### Dataset: 
Using [1362 Hip Hop artists metioned on Wikipedia](https://en.wikipedia.org/wiki/List_of_hip_hop_musicians) (accessed on 2019-05-22), the lyrics were scraped from [Genius](https://genius.com). Checkout the [Dataset Datasheet](https://github.com/Ljferrer/Ghost/tree/master/data) for more detailed information. 

### Extra Links:
- Inspiration: [Kevin Knight's Poetry Generator](https://aclweb.org/anthology/P17-4008.pdf)
- BERT: [Original Paper](https://arxiv.org/abs/1810.04805), [Explained](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270), [PyTorch Implementation](https://github.com/huggingface/pytorch-pretrained-BERT)
- Genius API: [Client](https://genius.com/api-clients)

### Compute Env:
CPU Env:
```bash
conda env create .
. activate GG
ipython kernel install --user --name=GG
python -m spacy download "en"
```

