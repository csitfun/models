import logging
import sys
import torch
import torch.nn as nn
from torch.nn import functional as fn
from torch.nn import CrossEntropyLoss, MSELoss, CosineSimilarity, CosineEmbeddingLoss
from transformers import (RobertaConfig, RobertaModel,
                          RobertaForSequenceClassification, RobertaTokenizer,
                          ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,
                          BertPreTrainedModel)
from xdec_config import get_logger

logger = get_logger(__name__)


class PairSimMLP(nn.Module):

    def __init__(self, config, **model_kwargs):
        super().__init__()
        self.hidden_layer = nn.Linear(config.hidden_size * 4,
                                      config.hidden_size)
        self.hidden_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_1, input_2, **kwargs):
        input_ = torch.cat(
            (input_1, input_2, torch.abs(input_1 - input_2), input_1 * input_2),
            dim=-1)
        input_ = self.hidden_dropout(input_)
        h = self.hidden_layer(input_)
        h = torch.tanh(h)
        d = self.hidden_dropout(h)
        out = self.output_layer(d)
        return out


class PairMLP(nn.Module):

    def __init__(self, config, **model_kwargs):
        super().__init__()
        self.hidden_layer = nn.Linear(config.hidden_size * 2,
                                      config.hidden_size)
        self.hidden_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_1, input_2, **kwargs):
        input_ = torch.cat((input_1, input_2), dim=-1)
        input_ = self.hidden_dropout(input_)
        h = self.hidden_layer(input_)
        h = torch.tanh(h)
        d = self.hidden_dropout(h)
        out = self.output_layer(d)
        return out


class RobertaPairSimAtt(BertPreTrainedModel):
    """
    Representation similarity based MLP model.

    cf. https://www.aclweb.org/anthology/W18-6456.pdf
    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, *model_args, **model_kwargs):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.output_hidden_states = config.output_hidden_states
        self.pair_sim_mlp = PairSimMLP(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        query_input_ids=None,
        query_attention_mask=None,
        query_token_type_ids=None,
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
        query_outputs = self.roberta(
            query_input_ids,
            attention_mask=query_attention_mask,
            token_type_ids=query_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # sequence_output, _, (hidden_states), (attentions) = outputs
        #                     emb, layer 0-11   layer 0-11
        logger.debug("input_ids: {}".format(input_ids))
        # B x len x dim
        sequence_output = outputs[0]
        query_sequence_output = query_outputs[0]
        # take <s> token (equiv. to [CLS])
        # B x dim
        query_summary = query_sequence_output[:, 0, :]
        history_summary = sequence_output[:, 0, :]

        logger.debug("history_summary size: {}".format(history_summary.size()))
        logger.debug("query_summary size: {}".format(query_summary.size()))

        logits = self.pair_sim_mlp(history_summary, query_summary)

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


class RobertaPairAtt(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, *model_args, **model_kwargs):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.output_hidden_states = config.output_hidden_states
        self.pair_mlp = PairMLP(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        query_input_ids=None,
        query_attention_mask=None,
        query_token_type_ids=None,
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
        query_outputs = self.roberta(
            query_input_ids,
            attention_mask=query_attention_mask,
            token_type_ids=query_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # sequence_output, _, (hidden_states), (attentions) = outputs
        #                     emb, layer 0-11   layer 0-11
        logger.debug("input_ids: {}".format(input_ids))
        # B x len x dim
        sequence_output = outputs[0]
        logger.debug("so size: {}".format(sequence_output.size()))
        query_sequence_output = query_outputs[0]
        logger.debug("qso size: {}".format(query_sequence_output.size()))
        # take <s> token (equiv. to [CLS])
        # B x dim
        query_summary = query_sequence_output[:, 0, :]
        logger.debug("qs size: {}".format(query_summary.size()))
        # B x len
        att = torch.matmul(sequence_output,
                           query_summary.unsqueeze(-1)).squeeze(-1)
        attention_mask = ~attention_mask.bool()
        logger.debug("1 attention_mask: {} type: {}".format(
            attention_mask, attention_mask.type()))
        logger.debug("1 attention_mask.size(): {}".format(
            attention_mask.size()))
        att[attention_mask] = float("-inf")
        # B x len -> B x 1 x len
        att = nn.functional.softmax(att, dim=1).unsqueeze(1)
        # sequence_output: B x len x dim
        # B x 1 x dim -> B x dim
        sequence_summary = torch.matmul(att, sequence_output).squeeze(1)
        logits = self.pair_mlp(sequence_summary, query_summary)
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
