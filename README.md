# Sentiment Analysis of Amazon Reviews

* Created baseline model - Logistic Regression with Tf-Idf
* Fine-tuned DistilBert
* Applied several techniques to increase inference speed and decrease size on GPU and CPU

| Models       | Accuracy |
| -------------|:--------:|
| Log Reg      | 90.29    |
| DistilBert   | 96.22    |

[Detailed notebook](https://www.kaggle.com/alexalex02/sentiment-analysis-distilbert-amazon-reviews)

## Inference Optimization

* TorchScript
* Dynamic Quantization
* ONNX Runtime

### CPU

![cpu](https://i.ibb.co/jLPy0fm/128-64.png)

### GPU

![gpu](https://i.ibb.co/3v3grqS/gpu-256-64.png)

[Detailed notebook on optimization](https://www.kaggle.com/alexalex02/nlp-transformers-inference-optimization)
