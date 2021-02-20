import logging
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import (RobertaConfig, RobertaModel,
                          RobertaForSequenceClassification, RobertaTokenizer,
                          ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,
                          BertPreTrainedModel)

logger = logging.getLogger(__name__)


class WideAndDeepLayer(nn.Module):

    def __init__(self, config, **model_kwargs):
        super().__init__()
        self.use_layers = [
            int(x) for x in model_kwargs["use_layers"] if 11 >= int(x) >= 1
        ]
        self.use_avg = model_kwargs["use_avg"]
        logger.info(f"WideAndDeepLayer uses layers: {self.use_layers}")
        logger.info(f"WideAndDeepLayer use average states: {self.use_avg}")

        self.deep_dense = nn.Linear(
            config.hidden_size * (1 + len(self.use_layers)), config.hidden_size)
        self.deep_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(
            config.hidden_size + model_kwargs['wide_dimension'],
            config.num_labels)

        self.wide_weight = model_kwargs["wide_weight"]

    def forward(self, deep_features, wide_features, hidden_states,
                attention_mask, **kwargs):
        if self.use_avg:
            d = torch.sum(
                deep_features * attention_mask[:, :, None].float(), dim=1)
            instance_lengths = torch.sum(attention_mask, dim=1)[:, None].float()
            d = d / instance_lengths
        else:
            d = deep_features[:, 0, :]  # take <s> token (equiv. to [CLS])

        if len(self.use_layers) > 0 and len(hidden_states) > 0:
            # add other hidden layers
            for layer_id in self.use_layers:
                # hidden_states: (embedding, layer 0, layer 1, ... layer 11)
                if self.use_avg:
                    layer_state = torch.sum(
                        hidden_states[layer_id] *
                        attention_mask[:, :, None].float(),
                        dim=1)
                    layer_state = layer_state / instance_lengths
                else:
                    layer_state = hidden_states[layer_id][:, 0, :]
                d = torch.cat((d, layer_state), -1)

        d = self.deep_dropout(d)
        d = self.deep_dense(d)
        d = torch.tanh(d)
        d = self.deep_dropout(d)

        w = wide_features * self.wide_weight
        wd = torch.cat((d, w), -1)

        wd = self.out_proj(wd)
        return wd


class RobertaWideAndDeep(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, *model_args, **model_kwargs):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.classifier = WideAndDeepLayer(config, **model_kwargs)
        self.output_hidden_states = config.output_hidden_states

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        wide_ids=None,
        labels=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # sequence_output, _, (hidden_states), (attentions) = outputs
        #                     emb, layer 0-11   layer 0-11
        sequence_output = outputs[0]
        hidden_states = outputs[2] if self.output_hidden_states else ()
        logits = self.classifier(sequence_output, wide_ids, hidden_states,
                                 attention_mask)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs
