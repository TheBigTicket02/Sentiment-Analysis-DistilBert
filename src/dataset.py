import config
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Mapping, List

class SentimentDataset(Dataset):
    """
    Wrapper around Torch Dataset to perform text classification
    """
    def __init__(self,
                 texts: List[str],
                 labels: List[str],
                 max_seq_length: int = config.MAX_SEQ_LENGTH,
                 model_name: str = config.MODEL_NAME):
        """
        Args:
            texts (List[str]): a list with texts to classify or to train the
                classifier on
            labels List[str]: a list with classification labels (optional)
            label_dict (dict): a dictionary mapping class names to class ids,
                to be passed to the validation data (optional)
            max_seq_length (int): maximal sequence length in tokens,
                texts will be stripped to this length
            model_name (str): transformer model name, needed to perform
                appropriate tokenization

        """

        self.texts = texts
        self.labels = labels
        self.max_seq_length = max_seq_length


        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # special tokens for transformers
        # in the simplest case a [CLS] token is added in the beginning
        # and [SEP] token is added in the end of a piece of text
        # [CLS] <indexes text tokens> [SEP] .. <[PAD]>
        self.sep_vid = self.tokenizer.vocab["[SEP]"]
        self.cls_vid = self.tokenizer.vocab["[CLS]"]
        self.pad_vid = self.tokenizer.vocab["[PAD]"]

    def __len__(self):
        """
        Returns:
            int: length of the dataset
        """
        return len(self.texts)

    def __getitem__(self, index) -> Mapping[str, torch.Tensor]:
        """Gets element of the dataset

        Args:
            index (int): index of the element in the dataset
        Returns:
            Single element by index
        """

        # encoding the text
        x = self.texts[index]
        x_encoded = self.tokenizer.encode(
            x,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        ).squeeze(0)

        # padding short texts
        true_seq_length = x_encoded.size(0)
        pad_size = self.max_seq_length - true_seq_length
        pad_ids = torch.Tensor([self.pad_vid] * pad_size).long()
        x_tensor = torch.cat((x_encoded, pad_ids))

        # dealing with attention masks - there's a 1 for each input token and
        # if the sequence is shorter that `max_seq_length` then the rest is
        # padded with zeroes. Attention mask will be passed to the model in
        # order to compute attention scores only with input data
        # ignoring padding
        mask = torch.ones_like(x_encoded, dtype=torch.int8)
        mask_pad = torch.zeros_like(pad_ids, dtype=torch.int8)
        mask = torch.cat((mask, mask_pad))

        output_dict = {
            "features": x_tensor,
            'attention_mask': mask
        }

        # encoding target
        output_dict["targets"] = torch.tensor(self.labels[index], dtype=torch.long)

        return output_dict