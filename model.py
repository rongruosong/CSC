# coding=utf-8
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertConfig

class BertForCSC(BertPreTrainedModel):
    def __init__(self, config, num_labels, ignore_index) -> None:
        super(BertForCSC, self).__init__(config)
        self.num_labels = num_labels
        self.ignore_index = ignore_index

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            # Only keep active parts of the loss
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)