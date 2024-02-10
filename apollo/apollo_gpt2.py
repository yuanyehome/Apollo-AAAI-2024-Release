import torch
import torch.nn as nn
import math
import json
from copy import deepcopy
from typing import Dict, Union, Any, Optional, Tuple, List
from transformers import (
    GPT2Model,
    GPT2LMHeadModel,
    Trainer,
)
from transformers.modeling_utils import unwrap_model
from transformers.utils import is_apex_available
from transformers.modeling_outputs import (
    CausalLMOutputWithCrossAttentions,
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


class ApolloGPT2Model(GPT2Model):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        apollo_info: Optional[ApolloInfo] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        curr_layers = nn.ModuleList(
            [self.h[idx] for idx in apollo_info.extend_layer_list]
        )

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
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(curr_layers))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                input_shape[-1] + past_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                print(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(curr_layers, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(
                        past_state.to(hidden_states.device) for past_state in layer_past
                    )
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    outputs[2 if use_cache else 1],
                )
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (
                        outputs[3 if use_cache else 2],
                    )

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class ApolloGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = ApolloGPT2Model(config)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        apollo_info: Optional[ApolloInfo] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            apollo_info=apollo_info,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )


class ApolloGPT2Trainer(Trainer):
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
        encoder_layers: nn.ModuleList = unwrap_model(self.model).transformer.h
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
