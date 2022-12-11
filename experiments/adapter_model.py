import os
import torch
import logging
import torch.nn as nn

from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, List
from adapter_generation_utils import GenerationMixin
from transformers import BartForConditionalGeneration
from transformers.file_utils import WEIGHTS_NAME, ModelOutput
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.bart.modeling_bart import BartAttention, BartEncoder, BartDecoder

logger = logging.getLogger(__name__)

ADAPTER_PARTS = {
    'bart': (BartForConditionalGeneration, BartEncoder, BartDecoder),
}

DOMAINS = ['restaurant', 'attraction', 'hotel', 'train', 'mixed']


@dataclass
class AdapterBaseModelOutput(ModelOutput):
    """ Custom output class for the adapter models """
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    lm_encoder_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class AdapterSeq2SeqLMOutput(ModelOutput):
    """ Custom output class for the adapter models """

    loss: Optional[torch.FloatTensor] = None
    logits: List[torch.FloatTensor] = None
    past_key_values: Optional[List[Tuple[Tuple[torch.FloatTensor]]]] = None
    decoder_hidden_states: Optional[List[Tuple[torch.FloatTensor]]] = None
    decoder_attentions: Optional[List[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[List[Tuple[torch.FloatTensor]]] = None
    encoder_last_hidden_state: Optional[List[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[List[Tuple[torch.FloatTensor]]] = None
    encoder_attentions: Optional[List[Tuple[torch.FloatTensor]]] = None


class AdapterConfig(PretrainedConfig):
    """ Adapter configuration; derived from the configuration of the pre-trained BART-Large model """

    def __init__(self, args):
        super(AdapterConfig, self).__init__()
        self.args = args

        self.activation_dropout = 0.1
        self.activation_function = 'gelu'
        self.add_bias_logits = False
        self.add_final_layer_norm = False
        self.architectures = ['BartModel']
        self.attention_dropout = 0.1
        self.bos_token_id = 0
        self.classif_dropout = 0.1
        self.classifier_dropout = 0.0
        self.d_model = self.args.adapter_size
        self.decoder_attention_heads = self.args.adapter_num_heads
        self.decoder_ffn_dim = self.args.adapter_ffn_size
        self.decoder_layerdrop = 0.0
        self.decoder_layers = self.args.adapter_num_layers
        self.decoder_start_token_id = 2
        self.dropout = 0.1
        self.early_stopping = True
        self.encoder_attention_heads = self.args.adapter_num_heads
        self.encoder_ffn_dim = self.args.adapter_ffn_size
        self.encoder_layerdrop = 0.0
        self.encoder_layers = self.args.adapter_num_layers
        self.eos_token_id = 2
        self.forced_eos_token_id = 2
        self.gradient_checkpointing = False
        self.id2label = {
            '0': 'LABEL_0',
            '1': 'LABEL_1',
            '2': 'LABEL_2'
        }
        self.init_std = 0.02
        self.is_encoder_decoder = True
        self.label2id = {
                       'LABEL_0': 0,
                       'LABEL_1': 1,
                       'LABEL_2': 2
                   },
        self.max_position_embeddings = 1024
        self.model_type = 'bart'
        self.no_repeat_ngram_size = 3
        self.normalize_before = False
        self.num_beams = 4
        self.num_hidden_layers = self.args.adapter_num_layers
        self.output_hidden_states = True
        self.output_attentions = True
        self.pad_token_id = 1
        self.scale_embedding = False
        self.task_specific_params = {
                                   'summarization': {
                                       'length_penalty': 1.0,
                                       'max_length': 128,
                                       'min_length': 12,
                                       'num_beams': 4
                                   },
                                   'summarization_cnn': {
                                       'length_penalty': 2.0,
                                       'max_length': 142,
                                       'min_length': 56,
                                       'num_beams': 4
                                   },
                                   'summarization_xsum': {
                                       'length_penalty': 1.0,
                                       'max_length': 62,
                                       'min_length': 11,
                                       'num_beams': 6
                                   }
                               },
        self.transformers_version = "4.8.1"
        self.use_cache = True
        self.vocab_size = 50265


class AdapterLayer(nn.Module):
    """ Adapter layer class for encoder-decoder models;
    can be inserted between encoder or decoder layers of a pretrained LM """

    def __init__(self, args, adapter_config, pretrained_model, in_encoder=False, use_weights_from_layer=None):
        super(AdapterLayer, self).__init__()
        self.adapter_config = adapter_config
        self.args = args
        self.pretrained_model = pretrained_model
        self.in_encoder = in_encoder
        self.use_weights_from_layer = use_weights_from_layer

        # Bottleneck adapters consist of a downward projection layer, encoder layers, and an upward projection layer
        # Adapters in the encoder utilize encoder layers as internal components, while decoder adapters use
        # decoder layers; d_model denotes the state dimensionality of the pretrained model / adapter
        self.down_project = nn.Linear(self.pretrained_model.model.config.d_model, self.adapter_config.d_model)
        self.internal_model = \
            ADAPTER_PARTS[args.model_type.lower()][1](self.adapter_config) if self.in_encoder \
            else ADAPTER_PARTS[args.model_type.lower()][2](self.adapter_config)
        self.up_project = nn.Linear(self.adapter_config.d_model, self.pretrained_model.model.config.d_model)

        self.down_project.to(self.args.device)
        self.internal_model.to(self.args.device)
        self.up_project.to(self.args.device)

        self.init_weights()

    def forward(self,
                hidden_states=None,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                head_mask=None,
                cross_attn_head_mask=None,
                past_key_values=None,
                output_attentions=True,
                output_hidden_states=True,
                use_cache=None):
        """ Defines the forward pass through the adapter network """

        # Project-down the decoder input states
        down_projected = self.down_project(hidden_states)
        # Reuse LM attention mask to ignore padded positions and obtain internal adapter representations
        internal_hidden_state, internal_self_attention, internal_cross_attention, internal_present_key_value = \
            None, None, None, None
        if self.in_encoder:
            internal_outputs = self.internal_model(input_ids=None,
                                                   attention_mask=attention_mask,
                                                   head_mask=head_mask,
                                                   inputs_embeds=down_projected,
                                                   output_attentions=output_attentions,
                                                   output_hidden_states=output_hidden_states,
                                                   return_dict=True)
            if output_attentions:
                internal_self_attention = internal_outputs.attentions
        else:
            internal_outputs = self.internal_model(input_ids=None,
                                                   attention_mask=attention_mask,
                                                   encoder_hidden_states=encoder_hidden_states,
                                                   encoder_attention_mask=encoder_attention_mask,
                                                   head_mask=head_mask,
                                                   cross_attn_head_mask=cross_attn_head_mask,
                                                   past_key_values=past_key_values,
                                                   inputs_embeds=down_projected,
                                                   use_cache=use_cache,
                                                   output_attentions=output_attentions,
                                                   output_hidden_states=output_hidden_states,
                                                   return_dict=True)

            if output_attentions:
                internal_self_attention = internal_outputs.attentions
                internal_cross_attention = internal_outputs.cross_attentions
            if use_cache:
                internal_present_key_value = internal_outputs.past_key_values

        # Project up and establish residual connection
        internal_hidden_state = internal_outputs.last_hidden_state
        up_projected = self.up_project(internal_hidden_state)

        return hidden_states + up_projected, \
            (internal_hidden_state, internal_self_attention, internal_cross_attention, internal_present_key_value)

    def _init_outer_weights(self, module):
        """ BART specific random weight initialization,
         see: https://github.com/huggingface/transformers/blob/master/src/transformers/models/bart/modeling_bart.py """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.adapter_config.init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.adapter_config.init_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _load_internal_weights(self):
        """ Initializes the adapter internal_model parameters with those of the subsequent LM layer;
         the resulting adapter does no longer include a bottleneck; all adapter layers are initialized with the
         parameters of the same LM layer """
        for layer_id in range(len(self.internal_model.layers)):
            if self.in_encoder:
                self.internal_model.layers[layer_id].load_state_dict(
                    self.pretrained_model.model.model.encoder.layers[self.use_weights_from_layer].state_dict())
            else:
                self.internal_model.layers[layer_id].load_state_dict(
                    self.pretrained_model.model.model.decoder.layers[self.use_weights_from_layer].state_dict())
            # Ensure that adapters are trainable
            for p in self.internal_model.layers[layer_id].parameters():
                p.requires_grad = True

    def init_weights(self):
        """ Initializes weights of the adapter"""
        # Initialize weights
        self._init_outer_weights(self.down_project)
        self._init_outer_weights(self.up_project)
        if self.use_weights_from_layer is not None:
            self._load_internal_weights()  # parameters are initialized randomly, otherwise


class PretrainedModel(nn.Module):
    """ Pretrained model class """

    def __init__(self, args):
        super(PretrainedModel, self).__init__()
        self.args = args
        self.model = ADAPTER_PARTS[self.args.model_type.lower()][0].from_pretrained(
            args.model_name_or_path, output_hidden_states=True)
        self.config = self.model.config

        # Freeze pre-trained model
        if not args.plm_only:
            self.freeze_plm_params()

    def freeze_plm_params(self):
        """ Unfreezes the PLM for downstream fine-tuning """
        # Freeze pre-trained model
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_plm_params(self):
        """ Unfreezes the PLM for downstream fine-tuning """
        # Freeze pre-trained model
        for p in self.parameters():
            p.requires_grad = True

    def forward(self,
                input_ids,
                decoder_input_ids,
                attention_mask=None,
                decoder_attention_mask=None,
                labels=None,
                return_dict=False):

        """ Defines the forward pass through the pre-trained model """
        # Obtain model representations
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             labels=labels,
                             decoder_input_ids=decoder_input_ids,
                             decoder_attention_mask=decoder_attention_mask,
                             return_dict=return_dict)
        return outputs


class AdapterEncoder(nn.Module):
    """ Adapter-augmented BART encoder """

    def __init__(self, args, pretrained_model):
        super(AdapterEncoder, self).__init__()
        self.args = args
        self.pretrained_model = pretrained_model
        self.config = pretrained_model.model.config
        self.adapter_config = AdapterConfig(args)
        self.adapter_list = args.adapter_list  # contains adapter insertion positions for the encoder and decoder
        # If no insertion sites are specified, insert an adapter layer after each LM layer
        if self.adapter_list is None:
            self.adapter_list = [layer_id for layer_id in range(0, len(pretrained_model.model.model.encoder.layers))]
        else:
            # Only retain layer IDs for the encoder
            self.adapter_list = [int(adapter_id.split('-')[1]) for adapter_id in args.adapter_list.split(',') if
                                 adapter_id.split('-')[0] == 'enc']

        # Initialize encoder-specific adapter networks
        encoder_adapters_list = list()
        for adapter_id in self.adapter_list:
            # for adapters inserted after the final LM layer, initialize with parameters of the final layer
            # if adapter_id == 0, the adapter would be initialized with the parameters of the 1st LM layer and
            # receive LM embeddings as input
            init_from_layer = int(adapter_id) if self.args.initialize_adapters_from_layers else None
            init_from_layer = (len(pretrained_model.model.model.encoder.layers) - 1) if \
                init_from_layer == len(pretrained_model.model.model.encoder.layers) else init_from_layer
            encoder_adapters_list.append(AdapterLayer(args,
                                                      self.adapter_config,
                                                      self.pretrained_model,
                                                      in_encoder=True,
                                                      use_weights_from_layer=init_from_layer))

        self.encoder_adapters = nn.ModuleList(encoder_adapters_list)
        # Initialize layers used to fuse final LM and adapter encoder states
        self.concat_dense = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size).to(self.args.device)

    def forward(self,
                pretrained_model_outputs,
                input_ids=None,
                attention_mask=None,
                head_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=False):

        """ Defines the forward pass through the adapter-augmented encoder """

        # Check input correctness
        if input_ids is None and attention_mask is None:
            raise ValueError('Either input_ids or attention_mask must be specified!')

        # Track layer outputs
        all_self_attentions = () if self.adapter_config.output_attentions else None

        # Unpack outputs of the pre-trained model; LM outputs a dictionary
        outputs = pretrained_model_outputs
        if 'encoder_hidden_states' in outputs.keys():
            lm_encoder_states = outputs['encoder_hidden_states']  # len == 13 fo BART-Large (embeddings + 12 layers)
        else:
            lm_encoder_states = outputs['hidden_states']
        final_encoder_states = lm_encoder_states[-1]
        # Initialize post-adapter representation
        hidden_adapter_out = torch.zeros(final_encoder_states.size()).to(self.args.device)

        # Create attention mask if none is provided
        if attention_mask is None:
            attention_mask = torch.where(torch.tensor(input_ids == self.config.pad_token_id), 0, 1)

        # Obtain adapter representations
        adapter_hidden_states = ()
        adapter_hidden_states_count = 0
        for i, adapter_layer in enumerate(self.encoder_adapters):
            # Select relevant hidden state of the pre-trained model and (if available) add state of previous adapter
            # if self.adapter_list[i] == 0, lm_encoder_states[self.adapter_list[i]] equals LM embeddings
            fusion_state = lm_encoder_states[self.adapter_list[i]] + hidden_adapter_out  # add previous adapter output
            hidden_adapter_out, internal_out = \
                adapter_layer(hidden_states=fusion_state,
                              attention_mask=attention_mask,
                              head_mask=head_mask,
                              output_attentions=output_attentions,
                              output_hidden_states=output_hidden_states)

            # Store adapter states
            adapter_hidden_states = adapter_hidden_states + (hidden_adapter_out,)
            adapter_hidden_states_count += 1
            # Optionally, add skip connections; self.args.adapter_skip_layers == 0 by default
            if self.args.adapter_skip_layers >= 1:
                if adapter_hidden_states_count % self.args.adapter_skip_layers == 0:
                    hidden_adapter_out = hidden_adapter_out + adapter_hidden_states[
                        int(adapter_hidden_states_count / self.args.adapter_skip_layers)]
            # Store information from internal adapter layers (i.e. the BART encoder layer within the adapter)
            if self.adapter_config.output_attentions:
                all_self_attentions = all_self_attentions + internal_out[1]  # internal_out contains tuples

        # Concatenate LM and adapter states
        concatenated_features = self.concat_dense(torch.cat([final_encoder_states, hidden_adapter_out], dim=-1))

        if return_dict:
            return tuple(v for v in
                         [concatenated_features, adapter_hidden_states, all_self_attentions, lm_encoder_states])
        else:
            return AdapterBaseModelOutput(last_hidden_state=concatenated_features,
                                          hidden_states=adapter_hidden_states,
                                          attentions=all_self_attentions,
                                          lm_encoder_states=lm_encoder_states)


class AdapterDecoder(nn.Module):
    """ Adapter-augmented BART decoder """

    def __init__(self, args, pretrained_model):
        super(AdapterDecoder, self).__init__()
        self.args = args
        self.pretrained_model = pretrained_model
        self.config = pretrained_model.model.config
        self.adapter_config = AdapterConfig(args)
        self.adapter_list = args.adapter_list  # contains adapter insertion positions for the encoder and decoder
        # If no insertion sites are specified, insert an adapter layer after each LM layer
        if self.adapter_list is None:
            self.adapter_list = [layer_id for layer_id in range(0, len(pretrained_model.model.model.decoder.layers))]
        else:
            # Only retain layer IDs for the decoder
            self.adapter_list = [int(adapter_id.split('-')[1]) for adapter_id in args.adapter_list.split(',') if
                                 adapter_id.split('-')[0] == 'dec']

        # Initialize encoder-specific adapter networks
        decoder_adapters_list = list()
        for adapter_id in self.adapter_list:
            # for adapters inserted after the final LM layer, initialize with parameters of the final layer
            init_from_layer = int(adapter_id) if self.args.initialize_adapters_from_layers else None
            init_from_layer = (len(pretrained_model.model.model.encoder.layers) - 1) if \
                init_from_layer == len(pretrained_model.model.model.decoder.layers) else init_from_layer  # specific to BART-Large
            decoder_adapters_list.append(AdapterLayer(args,
                                                      self.adapter_config,
                                                      self.pretrained_model,
                                                      in_encoder=False,
                                                      use_weights_from_layer=init_from_layer))

        self.decoder_adapters = nn.ModuleList(decoder_adapters_list)

        # Initialize down-projection for the encoder outputs
        self.down_project_encoder = \
            nn.Linear(self.pretrained_model.model.config.d_model, self.adapter_config.d_model).to(self.args.device)

        # Initialize layers used to fuse final LM and adapter decoders states
        self.concat_dense = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size).to(self.args.device)

    def forward(self,
                encoder_hidden_states,
                pretrained_model_outputs=None,
                decoder_input_ids=None,
                attention_mask=None,
                encoder_input_ids=None,
                encoder_attention_mask=None,
                head_mask=None,
                cross_attn_head_mask=None,
                past_key_values=None,
                output_attentions=None,
                output_hidden_states=None,
                use_cache=None,
                return_dict=False):

        """ Defines the forward pass through the adapter-augmented decoder """

        # Track layer outputs
        all_self_attns = () if output_attentions else None
        all_cross_attns = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = [] if use_cache else None

        # Unpack outputs of the pre-trained model; LM outputs a dictionary
        outputs = pretrained_model_outputs
        if 'decoder_hidden_states' in outputs.keys():
            lm_decoder_states = outputs['decoder_hidden_states']  # len == 13 fo BART-Large (embeddings + 12 layers)
        else:
            lm_decoder_states = outputs['hidden_states']
        final_decoder_states = lm_decoder_states[-1]
        # Initialize post-adapter representation
        hidden_adapter_out = torch.zeros(lm_decoder_states[-1].size()).to(self.args.device)
        # Down-project encoder outputs to be the same size as the adapter representations
        encoder_hidden_states = self.down_project_encoder(encoder_hidden_states)

        # Obtain adapter representations
        adapter_hidden_states = ()
        adapter_hidden_states_count = 0
        for i, adapter_layer in enumerate(self.decoder_adapters):
            # Select relevant hidden state of the pre-trained model and (if available) add state of previous adapter
            fusion_state = lm_decoder_states[self.adapter_list[i]] + hidden_adapter_out  # add previous adapter output
            # Adapter-decoder layers cross-attend to the adapter-encoder representations
            hidden_adapter_out, internal_out = \
                adapter_layer(hidden_states=fusion_state,
                              attention_mask=attention_mask,
                              encoder_hidden_states=encoder_hidden_states,
                              encoder_attention_mask=encoder_attention_mask,
                              head_mask=head_mask,
                              cross_attn_head_mask=cross_attn_head_mask,
                              past_key_values=past_key_values[i] if past_key_values is not None else past_key_values,
                              output_attentions=output_attentions,
                              output_hidden_states=output_hidden_states,
                              use_cache=use_cache)  # feed through the current adapter

            # Store adapter states
            adapter_hidden_states = adapter_hidden_states + (hidden_adapter_out,)
            adapter_hidden_states_count += 1
            # Optionally, add skip connections; self.args.adapter_skip_layers == 0 by default
            if self.args.adapter_skip_layers >= 1:
                if adapter_hidden_states_count % self.args.adapter_skip_layers == 0:
                    hidden_adapter_out = hidden_adapter_out + adapter_hidden_states[
                        int(adapter_hidden_states_count / self.args.dapter_skip_layers)]

            # Store information from internal adapter layers
            if output_attentions:
                all_self_attns = all_self_attns + internal_out[1]
                all_cross_attns = all_cross_attns + internal_out[2]
            if use_cache:
                next_decoder_cache.append(internal_out[3])  # collects decoder outputs [(sk, sv, ck, cv)] * num_layers

        # Concatenate LM and adapter states
        concatenated_features = self.concat_dense(torch.cat([final_decoder_states, hidden_adapter_out], dim=-1))

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in
                         [concatenated_features, next_cache, adapter_hidden_states, all_self_attns, all_cross_attns])
        else:
            return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=concatenated_features,
                                                             past_key_values=next_cache,
                                                             hidden_states=adapter_hidden_states,
                                                             attentions=all_self_attns,
                                                             cross_attentions=all_cross_attns)


class MultiDomainAdapterEncoder(nn.Module):
    """ Class combing the encoders of multiple domains into a single module """

    def __init__(self, args, config, pretrained_model, adapter_config, all_encoders, encoder_domains):
        super(MultiDomainAdapterEncoder, self).__init__()
        self.args = args
        self.config = config
        self.pretrained_model = pretrained_model
        self.adapter_config = adapter_config
        self.all_adapter_encoders = nn.ModuleList(all_encoders)
        self.encoder_domains = encoder_domains

        # Initialize mechanisms for combining the outputs of different adapters and the PLM
        if self.args.adapter_combo_method == 'concatenate':
            # Single projection layer (includes bias)
            self.concat_hidden_projection = nn.Linear(self.config.hidden_size * (len(self.args.active_domains) + 1),
                                                      self.config.hidden_size).to(self.args.device)

        if self.args.adapter_combo_method in ['gate', 'gate_hidden']:
            self.gating_weights_projection = \
                nn.Linear(self.config.hidden_size, (len(self.args.active_domains) + 1)).to(self.args.device)
            self.gating_softmax = nn.Softmax(dim=-1)

        if self.args.adapter_combo_method == 'expert':
            if len(self.args.active_domains) > 1:
                # Merge adapter representations
                self.adapter_weights_projection = \
                    nn.Linear(self.config.hidden_size, len(self.args.active_domains)).to(self.args.device)
            # Merge adapter and LM representations
            self.diff_weights_projection = nn.Linear(self.config.hidden_size, 1).to(self.args.device)

        if self.args.adapter_combo_method == 'attention':
            self.adapter_attention = BartAttention(self.config.hidden_size,
                                                   self.config.encoder_attention_heads,
                                                   self.config.attention_dropout,
                                                   is_decoder=False,
                                                   bias=True)

            self.activation_fn = nn.GELU()
            self.fc1 = nn.Linear(self.config.d_model, self.config.encoder_ffn_dim)
            self.fc2 = nn.Linear(self.config.encoder_ffn_dim, self.config.d_model)
            self.final_layer_norm = nn.LayerNorm(self.config.d_model)

        if self.args.adapter_combo_method == 'gru':
            self.fusion_cell = nn.GRUCell(self.config.d_model, self.config.d_model, bias=True, device=self.args.device)

    def forward(self,
                pretrained_model_outputs=None,
                input_ids=None,
                attention_mask=None,
                head_mask=None,
                current_domain=None):

        """ Combined forward pass through all domain-specific adapter encoders """

        # Obtain representations from the pretrained encoder if none are provided
        if pretrained_model_outputs is None:
            pretrained_encoder = self.pretrained_model.model.model.get_encoder()
            pretrained_encoder_outputs = pretrained_encoder(input_ids=input_ids,
                                                            attention_mask=attention_mask,
                                                            return_dict=True)
        else:
            pretrained_encoder_outputs = pretrained_model_outputs

        # Iterate over all domain-specific encoders
        all_encoder_outputs = []
        for encoder_id in range(len(self.all_adapter_encoders)):

            # Perform the encoder forward pass; this is needed for the 'generate' function
            encoder_outputs = \
                self.all_adapter_encoders[encoder_id](pretrained_encoder_outputs,
                                                      input_ids=input_ids,
                                                      attention_mask=attention_mask,
                                                      output_attentions=self.adapter_config.output_attentions,
                                                      output_hidden_states=self.adapter_config.output_hidden_states,
                                                      head_mask=head_mask,
                                                      return_dict=False)
            all_encoder_outputs.append(encoder_outputs)

        # Add pretrained encoder output
        all_encoder_outputs = tuple([pretrained_encoder_outputs] + all_encoder_outputs)
        final_encoder_outputs = []  # only keep the states
        for encoder_outputs in all_encoder_outputs:
            final_encoder_outputs.append(encoder_outputs[1][-1])

        mixed_encoder_out = final_encoder_outputs[0]
        if self.args.adapter_combo_method == 'mean':
            # kind-of like softmax ensembling (facilitated by the re-used LM head)
            mixed_encoder_out = torch.mean(torch.stack(final_encoder_outputs, dim=0), dim=0)

        if self.args.adapter_combo_method == 'concatenate':
            # Pre-trained model parameters will be fine-tuned on downstream task for better adapter integration
            mixed_encoder_out = self.concat_hidden_projection(torch.cat(final_encoder_outputs, dim=-1))

        if self.args.adapter_combo_method in ['gate', 'gate_hidden']:
            # a gated extension of concatenate
            hidden_scaling_weights = \
                torch.unsqueeze(self.gating_weights_projection(final_encoder_outputs[0]), dim=-2)
            hidden_weights = self.gating_softmax(hidden_scaling_weights)
            stacked_hidden = torch.stack(final_encoder_outputs, dim=-1)
            mixed_encoder_out = torch.mean(hidden_weights * stacked_hidden, dim=-1)

        if self.args.adapter_combo_method == 'expert':
            # a modification of the 'gate' method
            if len(self.args.active_domains) > 1:
                adapter_scaling_weights = \
                    torch.unsqueeze(self.adapter_weights_projection(final_encoder_outputs[0]), dim=-2)
                adapter_weights = self.expert_softmax(adapter_scaling_weights)
                stacked_adapters = torch.stack(final_encoder_outputs[1:], dim=-1)
                adapters_out = torch.mean(adapter_weights * stacked_adapters, dim=-1)
            else:
                adapters_out = final_encoder_outputs[1]

            diff_weights = self.diff_weights_projection(final_encoder_outputs[0])
            mixed_encoder_out = final_encoder_outputs[0] + (diff_weights * (adapters_out - final_encoder_outputs[0]))

        if self.args.adapter_combo_method == 'attention':
            key_value_states = torch.cat(final_encoder_outputs, dim=1)  # concatenate at time-step
            attended_adapter_hidden = self.adapter_attention(hidden_states=final_encoder_outputs[0],
                                                             key_value_states=key_value_states)[0]
            hidden_added = final_encoder_outputs[0] + attended_adapter_hidden
            residual = hidden_added

            # Feed-forward projection
            hidden_added = self.activation_fn(self.fc1(hidden_added))
            hidden_added = nn.functional.dropout(hidden_added, p=self.config.activation_dropout, training=self.training)
            hidden_added = self.fc2(hidden_added)
            hidden_added = nn.functional.dropout(hidden_added, p=self.config.dropout, training=self.training)
            hidden_added = residual + hidden_added
            mixed_encoder_out = self.final_layer_norm(hidden_added)

        if self.args.adapter_combo_method == 'gru':
            # Flatten cell inputs
            mixed_encoder_out = final_encoder_outputs[0].view(-1, self.config.d_model)
            for adapter_hidden in final_encoder_outputs[1:]:
                cell_input = adapter_hidden.view(-1, self.config.d_model)
                # Fuse states
                mixed_encoder_out = self.fusion_cell(cell_input, mixed_encoder_out)
            # Reshape
            mixed_encoder_out = mixed_encoder_out.view(final_encoder_outputs[0].size())

        if not self.args.no_encoder_integration:
            final_encoder_outputs[0] = mixed_encoder_out  # inject adapter information into PLM encoder output

        return all_encoder_outputs, final_encoder_outputs


class MultiDomainAdapterDecoder(nn.Module):
    """ Class combing the decoders of multiple domains into a single module """

    def __init__(self, args, pretrained_model, adapter_config, all_decoders, decoder_domains):
        super(MultiDomainAdapterDecoder, self).__init__()
        self.args = args
        self.pretrained_model = pretrained_model
        self.adapter_config = adapter_config
        self.all_adapter_decoders = nn.ModuleList(all_decoders)
        self.decoder_domains = decoder_domains

    def forward(self,
                pretrained_model_outputs=None,
                all_encoder_hidden_states=None,
                input_ids=None,
                encoder_attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                head_mask=None,
                decoder_head_mask=None,
                cross_attn_head_mask=None,
                past_key_values=None,
                use_cache=None,
                current_domain=None):
        """ Combined forward pass through all domain-specific adapter decoders """

        # Obtain representations from the pretrained decoder if none are provided
        if pretrained_model_outputs is None:
            assert decoder_input_ids is not None, 'decoder_input_ids must be provided during inference!'
            pretrained_decoder = self.pretrained_model.model.model.get_decoder()
            pretrained_model_outputs = pretrained_decoder(input_ids=decoder_input_ids,
                                                          attention_mask=decoder_attention_mask,
                                                          encoder_hidden_states=all_encoder_hidden_states[0][1],
                                                          encoder_attention_mask=encoder_attention_mask,
                                                          head_mask=head_mask,
                                                          cross_attn_head_mask=cross_attn_head_mask,
                                                          past_key_values=past_key_values[0],
                                                          use_cache=use_cache,
                                                          output_attentions=self.adapter_config.output_attentions,
                                                          output_hidden_states=self.adapter_config.output_hidden_states,
                                                          return_dict=True)

            pretrained_hidden_state = pretrained_model_outputs.last_hidden_state
            pretrained_model_cache = pretrained_model_outputs.past_key_values

        else:
            pretrained_hidden_state = pretrained_model_outputs.decoder_hidden_states[-1]
            pretrained_model_cache = pretrained_model_outputs.past_key_values

        past_key_values = past_key_values[1:]  # ignore pretrained model cache
        # Iterate over all domain-specific decoders
        all_decoder_outputs = []

        for decoder_id in range(len(self.all_adapter_decoders)):

            # Perform the decoder forward pass
            decoder_outputs = \
                self.all_adapter_decoders[decoder_id](all_encoder_hidden_states[decoder_id][0],
                                                      pretrained_model_outputs=pretrained_model_outputs,
                                                      decoder_input_ids=decoder_input_ids,
                                                      attention_mask=decoder_attention_mask,
                                                      encoder_input_ids=input_ids,
                                                      encoder_attention_mask=encoder_attention_mask,
                                                      head_mask=decoder_head_mask,
                                                      cross_attn_head_mask=cross_attn_head_mask,
                                                      past_key_values=past_key_values[decoder_id],
                                                      output_attentions=self.adapter_config.output_attentions,
                                                      output_hidden_states=self.adapter_config.output_hidden_states,
                                                      use_cache=use_cache,
                                                      return_dict=False)
            all_decoder_outputs.append(decoder_outputs)

        # Add pre-trained model cache
        all_decoder_outputs = [(pretrained_hidden_state, pretrained_model_cache)] + all_decoder_outputs

        return all_decoder_outputs


class AdapterModel(nn.Module, GenerationMixin, ABC):
    """ Adapter-equipped model class """

    def __init__(self, args, pretrained_model):

        super(AdapterModel, self).__init__()
        self.args = args
        self.pretrained_model = pretrained_model
        self.config = pretrained_model.model.config
        self.adapter_config = AdapterConfig(args)

        # Initialize sub-networks
        all_encoders_list = list()
        all_decoders_list = list()
        for domain in DOMAINS:  # preserve canonical domain order
            if domain in self.args.active_domains:
                all_encoders_list.append(AdapterEncoder(args, pretrained_model))
                all_decoders_list.append(AdapterDecoder(args, pretrained_model))

        ordered_active_domains = [domain for domain in DOMAINS if domain in self.args.active_domains]
        self.multi_encoder = MultiDomainAdapterEncoder(args,
                                                       self.config,
                                                       pretrained_model,
                                                       self.adapter_config,
                                                       all_encoders_list,
                                                       ordered_active_domains)
        self.multi_decoder = MultiDomainAdapterDecoder(args,
                                                       pretrained_model,
                                                       self.adapter_config,
                                                       all_decoders_list,
                                                       ordered_active_domains)

        if self.args.task is not None:
            for p in self.parameters():
                p.requires_grad = False
            self.pretrained_model.unfreeze_plm_params()
            if self.args.clone_lm_head:
                # Copy head compatible with adapters
                self.task_head = deepcopy(self.pretrained_model.model.lm_head)
                # Freeze adapter head
                self.pretrained_model.mode.lm_head.weight.requires_grad = False
                self.task_head.weight.requires_grad = True
            else:
                self.task_head = self.pretrained_model.model.lm_head

        # Initialize mechanisms for combining the outputs of different adapters and the PLM
        if self.args.adapter_combo_method == 'concatenate':
            # Single projection layer (includes bias)
            self.concat_logits_projection = nn.Linear(self.config.hidden_size * (len(self.args.active_domains) + 1),
                                                      self.config.hidden_size).to(self.args.device)

        if self.args.adapter_combo_method in ['gate', 'gate_hidden']:
            self.gating_weights_projection = \
                nn.Linear(self.config.hidden_size, (len(self.args.active_domains) + 1)).to(self.args.device)
            self.gating_softmax = nn.Softmax(dim=-1)

        if self.args.adapter_combo_method == 'attention':
            self.adapter_attention = BartAttention(self.config.hidden_size,
                                                   self.config.decoder_attention_heads,
                                                   self.config.attention_dropout,
                                                   is_decoder=False,
                                                   bias=True)

            self.activation_fn = nn.GELU()
            self.fc1 = nn.Linear(self.config.d_model, self.config.decoder_ffn_dim)
            self.fc2 = nn.Linear(self.config.decoder_ffn_dim, self.config.d_model)
            self.final_layer_norm = nn.LayerNorm(self.config.d_model)

        if self.args.adapter_combo_method == 'gru':
            self.fusion_cell = nn.GRUCell(self.config.d_model * len(self.args.active_domains),
                                          self.config.d_model, bias=True, device=self.args.device)

        if self.args.adapter_combo_method == 'expert':
            if len(self.args.active_domains) > 1:
                # Merge adapter representations
                self.adapter_weights_projection = \
                    nn.Linear(self.config.hidden_size, len(self.args.active_domains)).to(self.args.device)
            # Merge adapter and LM representations
            self.diff_weights_projection = nn.Linear(self.config.hidden_size, 1).to(self.args.device)

        if self.args.adapter_combo_method in ['gate_multi', 'gate_hidden_multi']:
            self.gating_weights_projection_multi = \
                nn.Linear(self.config.hidden_size, (len(self.args.active_domains) + 1)).to(self.args.device)
            self.gating_softmax_multi = nn.Softmax(dim=-1)

    def forward(self,
                pretrained_model_outputs=None,
                current_domain=None,
                encoder_outputs=None,
                final_encoder_outputs=None,
                input_ids=None,
                labels=None,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                head_mask=None,
                decoder_head_mask=None,
                cross_attn_head_mask=None,
                past_key_values=None,
                use_cache=None,
                return_dict=False):
        """ Defines the forward pass through the full model, using the MLM training objective """

        if past_key_values is None:
            past_key_values = [None] * (len(self.args.active_domains) + 1)

        if encoder_outputs is None:
            all_encoder_outputs, final_encoder_outputs = self.multi_encoder(
                pretrained_model_outputs=pretrained_model_outputs,
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                current_domain=current_domain)
        else:
            all_encoder_outputs = encoder_outputs
            final_encoder_outputs = final_encoder_outputs

        # pretrained_encoder_outputs = all_encoder_outputs[0]
        all_encoder_outputs = all_encoder_outputs[1:]

        # Construct encoder hidden states to be passed to the decoder
        if pretrained_model_outputs is None:
            pretrained_encoder_hidden_state = final_encoder_outputs[0]
        else:
            pretrained_encoder_hidden_state = pretrained_model_outputs.encoder_last_hidden_state

        all_encoder_hidden_states = []
        for encoder_outputs in all_encoder_outputs:
            adapter_encoder_hidden_state = \
                encoder_outputs.last_hidden_state if type(encoder_outputs) != tuple else encoder_outputs[0]
            all_encoder_hidden_states.append([adapter_encoder_hidden_state, pretrained_encoder_hidden_state])

        all_decoder_outputs = self.multi_decoder(pretrained_model_outputs=pretrained_model_outputs,
                                                 all_encoder_hidden_states=all_encoder_hidden_states,
                                                 input_ids=input_ids,
                                                 encoder_attention_mask=attention_mask,
                                                 decoder_input_ids=decoder_input_ids,
                                                 decoder_attention_mask=decoder_attention_mask,
                                                 head_mask=head_mask,
                                                 decoder_head_mask=decoder_head_mask,
                                                 cross_attn_head_mask=cross_attn_head_mask,
                                                 past_key_values=past_key_values,
                                                 use_cache=use_cache,
                                                 current_domain=current_domain)

        pretrained_decoder_final_state = all_decoder_outputs[0][0]
        pretrained_decoder_cache = all_decoder_outputs[0][1]
        all_decoder_outputs = all_decoder_outputs[1:]

        if len(all_decoder_outputs) > 1:
            decoder_final_hidden = [decoder_outputs[0] for decoder_outputs in all_decoder_outputs]
            assert '_multi' in self.args.adapter_combo_method, \
                'Multiple adapters are used but the specified combination function only supports a single adapter'
        else:
            decoder_final_hidden = all_decoder_outputs[0][0]

        lm_logits, logits, masked_lm_loss = None, None, None
        if self.args.adapter_combo_method is None:
            # Adapter outputs are not combined in any way, this is usually the case during adapter training
            logits = self.pretrained_model.model.lm_head(decoder_final_hidden)  # decoder_final_hidden is the adapter hidden state
            masked_lm_loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                masked_lm_loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        else:
            # To be used for downstream tasks once adapters are learned
            if pretrained_model_outputs is not None:
                lm_logits = pretrained_model_outputs.logits
            else:
                lm_logits = self.task_head(pretrained_decoder_final_state) + \
                            self.pretrained_model.model.final_logits_bias

            if self.args.adapter_combo_method == 'mean':
                # kind-of like softmax ensembling (facilitated by the re-used LM head)
                adapter_logits = self.pretrained_model.model.lm_head(decoder_final_hidden)
                logits = torch.mean(torch.stack([adapter_logits, lm_logits], dim=0), dim=0)

            if self.args.adapter_combo_method == 'concatenate':
                # Pre-trained model parameters will be fine-tuned on downstream task for better adapter integration
                concat_hidden = self.concat_logits_projection(torch.cat((decoder_final_hidden,
                                                                         pretrained_decoder_final_state), dim=-1))
                logits = self.task_head(concat_hidden) + self.pretrained_model.model.final_logits_bias

            if self.args.adapter_combo_method == 'gate':
                # a gated extension of concatenate
                adapter_logits = self.pretrained_model.model.lm_head(decoder_final_hidden)
                hidden_scaling_weights = \
                    torch.unsqueeze(self.gating_weights_projection(pretrained_decoder_final_state), dim=-2)
                stacked_logits = torch.stack([adapter_logits, lm_logits], dim=-1)
                logits = torch.mean(self.gating_softmax(hidden_scaling_weights) * stacked_logits, dim=-1)

            if self.args.adapter_combo_method == 'gate_hidden':
                # a gated extension of concatenate
                hidden_scaling_weights = \
                    torch.unsqueeze(self.gating_weights_projection(pretrained_decoder_final_state), dim=-2)
                stacked_hidden_states = torch.stack([decoder_final_hidden, pretrained_decoder_final_state], dim=-1)
                hidden_gated = torch.mean(self.gating_softmax(hidden_scaling_weights) * stacked_hidden_states, dim=-1)
                logits = self.task_head(hidden_gated) + self.pretrained_model.model.final_logits_bias

            if self.args.adapter_combo_method == 'attention':
                # Preliminary, don't think this formulation is very promising
                attended_adapter_hidden = self.adapter_attention(hidden_states=pretrained_decoder_final_state,
                                                                 key_value_states=decoder_final_hidden)[0]
                hidden_added = pretrained_decoder_final_state + attended_adapter_hidden
                residual = hidden_added

                # Feed-forward projection
                hidden_added = self.activation_fn(self.fc1(hidden_added))
                hidden_added = \
                    nn.functional.dropout(hidden_added, p=self.config.activation_dropout, training=self.training)
                hidden_added = self.fc2(hidden_added)
                hidden_added = nn.functional.dropout(hidden_added, p=self.config.dropout, training=self.training)
                hidden_added = residual + hidden_added
                hidden_added = self.final_layer_norm(hidden_added)

                # Logit projection
                logits = self.task_head(hidden_added) + self.pretrained_model.model.final_logits_bias

            if self.args.adapter_combo_method == 'gru':
                # Flatten cell inputs
                hidden_updated = pretrained_decoder_final_state.view(-1, self.config.d_model)
                for adapter_hidden in [decoder_outputs[0] for decoder_outputs in all_decoder_outputs]:
                    cell_input = adapter_hidden.view(-1, self.config.d_model)
                    # Fuse states
                    hidden_updated = self.fusion_cell(cell_input, hidden_updated)
                # Reshape
                hidden_updated = hidden_updated.view(pretrained_decoder_final_state.size())

                logits = self.task_head(hidden_updated) + self.pretrained_model.model.final_logits_bias

            if self.args.adapter_combo_method == 'expert':
                if len(self.args.active_domains) > 1:
                    adapter_scaling_weights = \
                        torch.unsqueeze(self.adapter_weights_projection(pretrained_decoder_final_state), dim=-2)
                    adapter_weights = self.expert_softmax(adapter_scaling_weights)
                    stacked_adapters = \
                        torch.stack([decoder_outputs[0] for decoder_outputs in all_decoder_outputs], dim=-1)
                    adapters_out = torch.mean(adapter_weights * stacked_adapters, dim=-1)
                else:
                    adapters_out = all_decoder_outputs[0][0]

                # Compute logits
                diff_weights = self.diff_weights_projection(pretrained_decoder_final_state)
                adapter_logits = self.pretrained_model.model.lm_head(adapters_out)
                lm_logits = \
                    self.task_head(pretrained_decoder_final_state) + self.pretrained_model.model.final_logits_bias
                logits = lm_logits + (diff_weights * (adapter_logits - lm_logits))

            if self.args.adapter_combo_method == 'mean_multi':
                adapter_logits = [self.pretrained_model.model.lm_head(hidden) for hidden in decoder_final_hidden]
                logits = torch.mean(torch.stack(adapter_logits + [lm_logits], dim=0), dim=0)

            if self.args.adapter_combo_method == 'gate_multi':
                # a gated extension of concatenate
                adapter_logits = [self.pretrained_model.model.lm_head(hidden) for hidden in decoder_final_hidden]
                hidden_scaling_weights_multi = \
                    torch.unsqueeze(self.gating_weights_projection_multi(pretrained_decoder_final_state), dim=-2)
                stacked_logits = torch.stack(adapter_logits + [lm_logits], dim=-1)
                logits = torch.mean(self.gating_softmax_multi(hidden_scaling_weights_multi) * stacked_logits, dim=-1)

            if self.args.adapter_combo_method == 'gate_hidden_multi':
                # a gated extension of concatenate
                hidden_scaling_weights_multi = \
                    torch.unsqueeze(self.gating_weights_projection_multi(pretrained_decoder_final_state), dim=-2)
                stacked_hidden_states = torch.stack(decoder_final_hidden + [pretrained_decoder_final_state], dim=-1)
                hidden_gated = \
                    torch.mean(self.gating_softmax_multi(hidden_scaling_weights_multi) * stacked_hidden_states, dim=-1)
                logits = self.task_head(hidden_gated) + self.pretrained_model.model.final_logits_bias

            masked_lm_loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                masked_lm_loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        # Restructure outputs
        enc_all_concatenated_features = [encoder_outputs[0] for encoder_outputs in all_encoder_outputs]
        enc_all_adapter_hidden_states = [encoder_outputs[1] for encoder_outputs in all_encoder_outputs]
        enc_all_self_attentions = [encoder_outputs[2] for encoder_outputs in all_encoder_outputs]
        enc_all_lm_encoder_states = [encoder_outputs[3] for encoder_outputs in all_encoder_outputs]
        encoder_outputs = [enc_all_concatenated_features, 
                           enc_all_adapter_hidden_states, 
                           enc_all_self_attentions, 
                           enc_all_lm_encoder_states]

        dec_all_concatenated_features = [decoder_outputs[0] for decoder_outputs in all_decoder_outputs]
        dec_all_next_cache = \
            [pretrained_decoder_cache] + [decoder_outputs[1] for decoder_outputs in all_decoder_outputs]
        dec_all_adapter_hidden_states = [decoder_outputs[2] for decoder_outputs in all_decoder_outputs]
        dec_all_self_attn = [decoder_outputs[3] for decoder_outputs in all_decoder_outputs]
        dec_all_cross_attn = [decoder_outputs[4] for decoder_outputs in all_decoder_outputs]
        decoder_outputs = [dec_all_concatenated_features,
                           dec_all_next_cache,
                           dec_all_adapter_hidden_states,
                           dec_all_self_attn,
                           dec_all_cross_attn]

        if not return_dict:
            outputs = decoder_outputs + encoder_outputs
            output = (logits,) + tuple(outputs[1:])
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        else:
            return AdapterSeq2SeqLMOutput(
                loss=masked_lm_loss,
                logits=logits,
                past_key_values=decoder_outputs[1],
                decoder_hidden_states=decoder_outputs[2],
                decoder_attentions=decoder_outputs[3],
                cross_attentions=decoder_outputs[4],
                encoder_last_hidden_state=encoder_outputs[0],
                encoder_hidden_states=encoder_outputs[1],
                encoder_attentions=encoder_outputs[2],
            )

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            final_encoder_outputs=None,
            **kwargs
    ):
        """ Adopted from BART implementation """
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "final_encoder_outputs": final_encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def get_encoder(self):
        return self.multi_encoder

    def get_decoder(self):
        return self.multi_decoder

    def get_output_embeddings(self):
        return self.pretrained_model.model.lm_head

    def save_pretrained(self,
                        save_directory,
                        save_config=True,
                        save_function=torch.save):
        """ Save trained model """

        if os.path.isfile(save_directory):
            logger.error('Provided path ({}) should be a directory, not a file'.format(save_directory))
            return
        os.makedirs(save_directory, exist_ok=True)

        # Only save the model itself if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self
        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        # Save the config
        if save_config:
            model_to_save.config.save_pretrained(save_directory)
        # Save the model
        state_dict = model_to_save.state_dict()

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        save_function(state_dict, output_model_file)
        logger.info('Model weights saved in {}'.format(output_model_file))



