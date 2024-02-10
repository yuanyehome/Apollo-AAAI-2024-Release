import torch
import torch.nn as nn
import math
import json
from copy import deepcopy
from typing import Dict, Union, Any, Optional, Tuple, List
from transformers import (
    BertModel,
    BertForMaskedLM,
    Trainer,
)
from transformers.modeling_utils import unwrap_model
from transformers.models.bert.modeling_bert import BertEncoder
from transformers.utils import is_apex_available
from transformers.modeling_outputs import (
    MaskedLMOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions,
)
from .apollo_utils import (
    get_curr_apollo_info,
    ApolloInfo,
    safe_get_rank,
    extend_meta_layer,
    stack_meta_layer,
)


if is_apex_available():
    from apex import amp


class ApolloBertEncoder(BertEncoder):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        apollo_info: Optional[ApolloInfo] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                print(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        curr_layers = [self.layer[idx] for idx in apollo_info.extend_layer_list]
        for i, layer_module in enumerate(curr_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class ApolloBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.encoder = ApolloBertEncoder(config)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        apollo_info: Optional[ApolloInfo] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        # Modified from BertModel.forward()
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device
            )

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=device
                )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            apollo_info=apollo_info,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class ApolloBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.bert = ApolloBertModel(config, add_pooling_layer=False)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        apollo_info: Optional[ApolloInfo] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            apollo_info=apollo_info,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ApolloBertTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.apollo_args = kwargs.pop("apollo_args")
        super().__init__(*args, **kwargs)
        total_steps = int(
            self.args.num_train_epochs
            * max(
                math.floor(
                    len(self.get_train_dataloader())
                    / self.args.gradient_accumulation_steps
                ),
                1,
            )
        )
        print("Total steps: {}".format(total_steps))
        self.apollo_info_output_file = (
            (
                self.apollo_args.apollo_info_output_file
                + ".rank_{}".format(safe_get_rank())
            )
            if self.apollo_args.apollo_info_output_file
            else None
        )
        self.apollo_info_generator = self._build_apollo_info()

    def _build_apollo_info(self):
        if self.apollo_info_output_file:
            info_writer = open(self.apollo_info_output_file, "w")
        else:
            info_writer = None

        curr_global_step_with_accu = 0
        low = self.apollo_args.apollo_layers[0]
        all_stack_points = [
            max(
                math.floor(
                    len(self.get_train_dataloader())
                    / self.args.gradient_accumulation_steps
                ),
                1,
            )
            * epoch_idx
            for epoch_idx in self.apollo_args.apollo_epoch_list
        ]
        print("all_stack_points:", all_stack_points)
        stack_func = (
            extend_meta_layer
            if self.apollo_args.grow_method == "extend"
            else stack_meta_layer
        )
        layer_num_idx = 0
        while True:
            do_stack = False
            new_step = False
            curr_global_step = (
                curr_global_step_with_accu // self.args.gradient_accumulation_steps
            )
            if curr_global_step_with_accu % self.args.gradient_accumulation_steps == 0:
                new_step = True
                if curr_global_step in all_stack_points:
                    do_stack = True
                    layer_num_idx += 1
            curr_unique_layer = self.apollo_args.apollo_layers[layer_num_idx]
            curr_apollo_info = get_curr_apollo_info(
                curr_unique_layer,
                self.model.config.num_hidden_layers,
                self.apollo_args.dist_function,
                self.apollo_args.grow_method,
            )
            curr_apollo_info.do_stack = do_stack
            curr_apollo_info.stack_info = stack_func(
                self.apollo_args.apollo_layers[layer_num_idx - 1], curr_unique_layer
            )
            if info_writer:
                info_writer.write(
                    json.dumps(
                        {
                            "curr_global_step_with_accu": curr_global_step_with_accu,
                            "curr_global_step": curr_global_step,
                            "curr_unique_layer": curr_unique_layer,
                            "curr_meta_layer": curr_apollo_info.curr_meta_layer,
                            "extend_layer_list": curr_apollo_info.extend_layer_list,
                            "do_stack": curr_apollo_info.do_stack,
                            "stack_info": curr_apollo_info.stack_info,
                        }
                    )
                    + "\n"
                )
                info_writer.flush()
            yield curr_apollo_info
            curr_global_step_with_accu += 1

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        # Modified from Trainer.training_step()
        model.train()
        inputs = self._prepare_inputs(inputs)
        curr_apollo_info = next(self.apollo_info_generator)
        inputs["apollo_info"] = curr_apollo_info
        if curr_apollo_info.do_stack:
            self.stack_model(curr_apollo_info.stack_info)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps

    def stack_model(self, stack_info: List[int]):
        # Modified from Trainer.training_step()
        encoder_layers: nn.ModuleList = unwrap_model(self.model).bert.encoder.layer
        tgt_layer_idx = len(stack_info) - 1
        for src_layer_idx in stack_info[::-1]:
            for (src_n, src_p), (tgt_n, tgt_p) in zip(
                encoder_layers[src_layer_idx].named_parameters(),
                encoder_layers[tgt_layer_idx].named_parameters(),
            ):
                assert src_n == tgt_n
                tgt_p.data = deepcopy(src_p.data).detach()
            if safe_get_rank() == 0:
                print(
                    "stack_model: Copy layer {} to layer {}".format(
                        src_layer_idx, tgt_layer_idx
                    )
                )
            tgt_layer_idx -= 1
