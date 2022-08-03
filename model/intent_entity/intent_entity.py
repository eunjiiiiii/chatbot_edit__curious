import os
import numpy as np

import torch

from model.intent_entity.utils import MODEL_CLASSES, get_intent_labels, get_slot_labels, load_tokenizer


class JointIntEnt:
    def __init__(self, model_dir, no_cuda=False):
        self.args = torch.load(os.path.join(model_dir, 'training_args.bin'), map_location=torch.device("cuda:0"))
        self.device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"

        self.intent_label_lst = get_intent_labels(self.args)
        self.slot_label_lst = get_slot_labels(self.args)
        self.model = MODEL_CLASSES[self.args.model_type][1].from_pretrained(
            model_dir,
            args=self.args,
            intent_label_lst=self.intent_label_lst,
            slot_label_lst=self.slot_label_lst
        )
        self.model.to(self.device)
        self.tokenizer = load_tokenizer(self.args)

    def _tokenize(self, text: str):
        words = text.strip().split()

        tokens = []
        slot_label_mask = []
        for w in words:
            w_tokens = self.tokenizer.tokenize(w)
            if not w_tokens:
                w_tokens = [self.tokenizer.unk_token]
            tokens.extend(w_tokens)
            slot_label_mask.extend([self.args.ignore_index + 1] + [self.args.ignore_index] * (len(w_tokens) - 1))

        special_tokens_count = 2
        if len(tokens) > self.args.max_seq_len - special_tokens_count:
            tokens = tokens[: (self.args.max_seq_len - special_tokens_count)]
            slot_label_mask = slot_label_mask[: (self.args.max.seq_len - special_tokens_count)]

        # add [SEP] token
        tokens += [self.tokenizer.sep_token]
        token_type_ids = [0] * len(tokens)
        slot_label_mask += [self.args.ignore_index]

        # add [CLS] token
        tokens = [self.tokenizer.cls_token] + tokens
        token_type_ids = [0] + token_type_ids
        slot_label_mask = [self.args.ignore_index] + slot_label_mask

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        # zero-pad up to the sequence length
        padding_length = self.args.max_seq_len - len(input_ids)
        input_ids = input_ids + ([self.tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        slot_label_mask = slot_label_mask + ([self.args.ignore_index] * padding_length)

        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)
        slot_label_mask = torch.tensor(slot_label_mask, dtype=torch.long).unsqueeze(0)

        ret = {"input_ids": input_ids.to(self.device),
               "attention_mask": attention_mask.to(self.device),
               "token_type_ids": token_type_ids.to(self.device),
               "slot_label_mask": slot_label_mask.to(self.device)}

        return ret

    def __call__(self, inp):
        self.model.eval()

        slot_preds = None
        all_slot_label_mask = None
        with torch.no_grad():
            inputs = self._tokenize(inp)
            slot_label_mask = inputs.pop("slot_label_mask")
            inputs.update({"intent_label_ids": None, "slot_labels_ids": None})
            outputs = self.model(**inputs)
            _, (intent_logits, slot_logits) = outputs[:2]

            # Intent Prediction
            intent_preds = intent_logits.detach().cpu().numpy()
            intent_preds = np.argmax(intent_preds, axis=1)[0]

            # Slot Prediction
            if slot_preds is None:
                if self.args.use_crf:
                    # decode() in `torchcrf` returns list with best index directly
                    slot_preds = np.array(self.model.crf.decode(slot_logits))
                else:
                    slot_preds = slot_logits.detach().cpu().numpy()
                all_slot_label_mask = slot_label_mask.detach().cpu().numpy()
            else:
                if self.args.use_crf:
                    slot_preds = np.append(slot_preds, np.array(self.model.crf.decode(slot_logits)), axis=0)
                else:
                    slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)
                all_slot_label_mask = np.append(all_slot_label_mask, slot_label_mask.detach().cpu().numpy(), axis=0)

        if not self.args.use_crf:
            slot_preds = np.argmax(slot_preds, axis=2)

        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}

        slot_preds_list = []
        for i in range(slot_preds.shape[0]):
            for j in range(slot_preds.shape[1]):
                if all_slot_label_mask[i, j] != self.args.ignore_index:
                    slot_preds_list.append(slot_label_map[slot_preds[i][j]])

        #return self.intent_label_lst[intent_preds], slot_preds_list
        return self.intent_label_lst[intent_preds], slot_preds_list


if __name__ == '__main__':
    int_ent = JointIntEnt("./jointbert_demo_model", no_cuda=True)
    a, b = int_ent("오늘 내 기분이 너때문에 안좋아")
