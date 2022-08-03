import os
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from model.topic.utils import load_tokenizer


class IAI_TOPIC:
    def __init__(self, model_dir, no_cuda):
        self.args = torch.load(os.path.join(model_dir, 'training_args.bin'))
        self.tokenizer = load_tokenizer(self.args)
        self.device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)  # Config will be automatically loaded from model_dir
        self.model.to(self.device)
        self.model.eval()

    def convert_input_file_to_tensor_dataset(self,
                                             text,
                                             args,
                                             cls_token_segment_id=0,
                                             pad_token_segment_id=0,
                                             sequence_a_segment_id=0,
                                             mask_padding_with_zero=True):
        # Setting based on the current model type
        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        pad_token_id = self.tokenizer.pad_token_id

        all_input_ids = []
        all_attention_mask = []
        all_token_type_ids = []

        line = text.strip()
        tokens = self.tokenizer.tokenize(line)
        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > args.max_seq_len - special_tokens_count:
            tokens = tokens[:(args.max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = args.max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)

        # Change to Tensor
        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
        all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)

        dataset = (all_input_ids, all_attention_mask, all_token_type_ids)

        return dataset


    def predict(self, text):
        # Convert input file to TensorDataset
        dataset = self.convert_input_file_to_tensor_dataset(text, self.args)
        dataset = tuple(t.to(self.device) for t in dataset)

        # Predict
        with torch.no_grad():
            inputs = {"input_ids": dataset[0],
                      "attention_mask": dataset[1],
                      "labels": None}
            if self.args.model_type != "distilkobert":
                inputs["token_type_ids"] = dataset[2]
            outputs = self.model(**inputs)
            logits = outputs[0]

            softmax_layer = nn.Softmax(-1)
            softmax_result = softmax_layer(logits)
            max_emotion = max(t[0] for t in softmax_result)
            y_pred = logits.max(dim=1)[1]

            topic_pred = int(y_pred.detach().cpu().numpy()[0])
            topic_prod = softmax_result.detach().cpu().numpy()
            #max_topic_prob = float(max_emotion.detach().cpu().numpy())
            max_topic_prob = topic_prod[0].max()

        return topic_pred, topic_prod, max_topic_prob
