import torch
import torch.nn as nn
import torch.nn.functional as F

from bert import BertLayer, BertModel, BertSelfAttention
from utils import *

DEBUG_OUTPUT = False
NUM_TASKS_SUPPORTED = 3

class BertLayerWithParallelAdaption(BertLayer):

    # Shared across all layers NOT tasks
    instance_counter = 0

    def __init__(self, config, args, shared_adaption_layers_first = None, shared_adaption_layers_second = None, shared_attn_layers = None):
        super(BertLayerWithParallelAdaption, self).__init__(config)

        self.device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        self.config = config
        self.args = args

        print(f"BertLayerWithParallelAdaption config {self.args.parallel_adaption_layers} late-attach:{self.args.adaption_layer_late_attach} shared-attention:{self.args.adaption_layer_shared_attention}")

        # This size was borrowed directly from Bert and PALS
        self.adaption_layer_size_pals = 204
        self.adaption_layer_size_low_rank = 100

        # Set by the adaption layer type
        self.adaption_layer_size = None
        self.layer_index = BertLayerWithParallelAdaption.instance_counter
        self.number_of_hidden_layers = config.num_hidden_layers

        # These lists hold layers and parameters for each task
        self.adaption_layer_list_first = nn.ModuleList()
        self.adaption_layer_list_second = nn.ModuleList()
        self.attn_list = nn.ModuleList()
        pal_attn_config = copy.deepcopy(config)

        # Shared adaption layers
        self.shared_adaption_layers_first = shared_adaption_layers_first
        self.shared_adaption_layers_second = shared_adaption_layers_second
        self.shared_attn_layers = shared_attn_layers

        if self.args.parallel_adaption_layers == 'low-rank':
            self.adaption_layer_size = self.adaption_layer_size_low_rank

            for _ in range(NUM_TASKS_SUPPORTED):
                self.adaption_layer_list_first.append(nn.Linear(config.hidden_size, self.adaption_layer_size).to(self.device))
                self.adaption_layer_list_second.append(nn.Linear(self.adaption_layer_size, config.hidden_size).to(self.device))

        elif self.args.parallel_adaption_layers == 'pals':
            self.adaption_layer_size = self.adaption_layer_size_pals
            pal_attn_config.num_attention_heads = 6
            pal_attn_config.hidden_size = self.adaption_layer_size

            for _ in range(NUM_TASKS_SUPPORTED):
                self.adaption_layer_list_first.append(nn.Linear(config.hidden_size, self.adaption_layer_size).to(self.device))
                self.attn_list.append(BertSelfAttention(pal_attn_config).to(self.device))
                self.adaption_layer_list_second.append(nn.Linear(self.adaption_layer_size, config.hidden_size).to(self.device))

        elif self.args.parallel_adaption_layers == 'pals-shared':
            self.adaption_layer_size = self.adaption_layer_size_pals
            pal_attn_config.num_attention_heads = 6
            pal_attn_config.hidden_size = self.adaption_layer_size

            if BertLayerWithParallelAdaption.instance_counter == 0:
                print("Creating shared adaption layers")
                for _ in range(NUM_TASKS_SUPPORTED):
                    self.shared_adaption_layers_first.append(nn.Linear(config.hidden_size, self.adaption_layer_size).to(self.device))
                    self.shared_adaption_layers_second.append(nn.Linear(self.adaption_layer_size, config.hidden_size).to(self.device))
                    if self.args.adaption_layer_shared_attention:
                        self.shared_attn_layers.append(BertSelfAttention(pal_attn_config).to(self.device))
            else:
                print("Skipping creating shared adaption layers, already created!")

            if not self.args.adaption_layer_shared_attention:
                for _ in range(NUM_TASKS_SUPPORTED):
                    self.attn_list.append(BertSelfAttention(pal_attn_config).to(self.device))

        elif self.args.parallel_adaption_layers == 'mixed':
            # This is a mixed mode where the first half of the layers are low-rank and the second half are PALS
            self.adaption_layer_size = self.adaption_layer_size_pals
            pal_attn_config.num_attention_heads = 6
            pal_attn_config.hidden_size = self.adaption_layer_size

            for _ in range(NUM_TASKS_SUPPORTED):
                if self.layer_index < self.number_of_hidden_layers // 2:
                    print("Mixed adaptation layer: pals-shared at layer: ", self.layer_index)
                    if BertLayerWithParallelAdaption.instance_counter == 0:
                        print("Creating shared adaption layers")
                        for _ in range(NUM_TASKS_SUPPORTED):
                            self.shared_adaption_layers_first.append(nn.Linear(config.hidden_size, self.adaption_layer_size).to(self.device))
                            self.shared_adaption_layers_second.append(nn.Linear(self.adaption_layer_size, config.hidden_size).to(self.device))
                            if self.args.adaption_layer_shared_attention:
                                self.shared_attn_layers.append(BertSelfAttention(pal_attn_config).to(self.device))
                    else:
                        print("Skipping creating shared adaption layers, already created!")

                    if not self.args.adaption_layer_shared_attention:
                        for _ in range(NUM_TASKS_SUPPORTED):
                            self.attn_list.append(BertSelfAttention(pal_attn_config).to(self.device))
                else:
                    print("Mixed adaptation layer: low-rank at layer: ", self.layer_index)
                    self.adaption_layer_list_first.append(nn.Linear(config.hidden_size, self.adaption_layer_size).to(self.device))
                    self.adaption_layer_list_second.append(nn.Linear(self.adaption_layer_size, config.hidden_size).to(self.device))

        else:
            raise ValueError(f'Invalid adaption_layer_type: {self.adaption_mode}')

        # Count the number of instances of this class
        BertLayerWithParallelAdaption.instance_counter += 1

    def forward_parallel_adaption(self, hidden_states, attention_mask, task):
        # 1. Adaption layer. TS(h) = Vd * g(Ve * h)
        # g is a low-rank linear transformation for 'low-rank' adaption layers
        # g is multi-head attention for 'pals' adaption layers

        batch_size, seq_length, hidden_size = hidden_states.size()
        flattened_hidden_states = hidden_states.view(-1, hidden_size)

        if DEBUG_OUTPUT:
            print("TASK", task)
            print("SIZEOF hidden_states", hidden_states.size())
            print("SIZEOF Ve", self.Ve.size())
            print("SIZEOF Ve transpose", self.Ve.transpose(0, 1).size())
            print("SIZEOF Vd", self.Vd.size())
            print("SIZEOF flattened_hidden_states", flattened_hidden_states.size())

        adaption_output = None
        if self.args.parallel_adaption_layers == 'low-rank':
            adaption_output = self.adaption_layer_list_first[task](flattened_hidden_states)
            adaption_output = F.gelu(adaption_output)
            adaption_output = self.adaption_layer_list_second[task](adaption_output)

        elif self.args.parallel_adaption_layers == 'pals':
            adaption_output = self.adaption_layer_list_first[task](flattened_hidden_states)
            adaption_output = adaption_output.view(batch_size, seq_length,  self.adaption_layer_size)
            adaption_output = self.attn_list[task].forward(adaption_output, attention_mask)
            adaption_output = self.adaption_layer_list_second[task](adaption_output)
            adaption_output = F.gelu(adaption_output)

        elif self.args.parallel_adaption_layers == 'pals-shared':
            adaption_output = self.shared_adaption_layers_first[task](flattened_hidden_states)
            adaption_output = adaption_output.view(batch_size, seq_length,  self.adaption_layer_size)
            if self.args.adaption_layer_shared_attention:
                adaption_output = self.shared_attn_layers[task].forward(adaption_output, attention_mask)
            else:
                adaption_output = self.attn_list[task].forward(adaption_output, attention_mask)
            adaption_output = self.shared_adaption_layers_second[task](adaption_output)
            adaption_output = F.gelu(adaption_output)

        elif self.args.parallel_adaption_layers == 'mixed':
            if self.layer_index < self.number_of_hidden_layers // 2:
                adaption_output = self.shared_adaption_layers_first[task](flattened_hidden_states)
                adaption_output = adaption_output.view(batch_size, seq_length,  self.adaption_layer_size)
                if self.args.adaption_layer_shared_attention:
                    adaption_output = self.shared_attn_layers[task].forward(adaption_output, attention_mask)
                else:
                    adaption_output = self.attn_list[task].forward(adaption_output, attention_mask)
                adaption_output = self.shared_adaption_layers_second[task](adaption_output)
                adaption_output = F.gelu(adaption_output)
            else:
                adaption_output = self.adaption_layer_list_first[task](flattened_hidden_states)
                adaption_output = F.gelu(adaption_output)
                adaption_output = self.adaption_layer_list_second[task](adaption_output)

        else:
            raise ValueError(f'Invalid adaption_layer_type: {self.adaption_mode}')

        adaption_output = adaption_output.view(batch_size, seq_length, hidden_size)
        if DEBUG_OUTPUT:
            print("SIZEOF adaption_output", adaption_output.size())

        return adaption_output

    #OVERRIDE
    def forward(self, hidden_states, attention_mask, task, perturb=False):
        # 1. Multi-head attention layer. MH(h)
        attn_output = self.self_attention(hidden_states, attention_mask)

        # 1a. Parallel adaption layer. TS(h)
        adaption_output = None
        if(task == 0):
            if DEBUG_OUTPUT: print("TASK 0")
            adaption_output = self.forward_parallel_adaption(hidden_states, attention_mask, task)

        # 2. Add-norm for multi-head attention. LN(h + MH(h))
        if not self.args.adaption_layer_late_attach and task == 0:
            if DEBUG_OUTPUT: print("Adding adaption output to attention output")
            attn_output = attn_output + adaption_output
        attn_output_normalized = self.add_norm(hidden_states, attn_output, self.attention_dense, self.attention_dropout, self.attention_layer_norm)

        # 3. Feed forward layer. SA(h) = FFN(LN(h + MH(h)))
        interm_output = self.interm_af(self.interm_dense(attn_output_normalized))

        if DEBUG_OUTPUT:
            print("SIZEOF interm_output", interm_output.size())
            print("SIZEOF attn_output_normalized", attn_output_normalized.size())

        # 4. Add-norm for feed forward. Original: LN(h + SA(h)) Now: LN(h + SA(h) + TS(h))
        if self.args.adaption_layer_late_attach and task == 0:
            if DEBUG_OUTPUT: print("Adding adaption output to interm output")
            attn_output_normalized = attn_output_normalized + adaption_output
        output = self.add_norm(attn_output_normalized, interm_output, self.out_dense, self.out_dropout, self.out_layer_norm)

        return output

class BertModelWithParallelAdaption(BertModel):
    def __init__(self, config, args):
        super(BertModelWithParallelAdaption, self).__init__(config)

        # Shared adation layers
        self.shared_adaption_layers_first = nn.ModuleList()
        self.shared_adaption_layers_second = nn.ModuleList()
        self.shared_attn_layers = nn.ModuleList()

        # Replace the original BertLayer with the new BertLayerWithAdaption
        if args.parallel_adaption_layers == 'pals-shared' or args.parallel_adaption_layers == 'mixed':
            self.bert_layers = nn.ModuleList([BertLayerWithParallelAdaption(config, args, self.shared_adaption_layers_first, self.shared_adaption_layers_second, self.shared_attn_layers) for _ in range(config.num_hidden_layers)])
        else:
            self.bert_layers = nn.ModuleList([BertLayerWithParallelAdaption(config, args) for _ in range(config.num_hidden_layers)])

    def encode(self, hidden_states, attention_mask, task):
        """
        hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len]
        """
        # Get the extended attention mask for self-attention.
        # Returns extended_attention_mask of size [batch_size, 1, 1, seq_len].
        # Distinguishes between non-padding tokens (with a value of 0) and padding tokens
        # (with a value of a large negative number).
        extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)

        # Pass the hidden states through the encoder layers.
        for i, layer_module in enumerate(self.bert_layers):
            # Feed the encoding from the last bert_layer to the next.
            hidden_states = layer_module(hidden_states, extended_attention_mask, task)

        return hidden_states

    def forward(self, input_ids, attention_mask, task, perturb=False):
        """
        input_ids: [batch_size, seq_len], seq_len is the max length of the batch
        attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
        """
        # Get the embedding for each input token.
        embedding_output = self.embed(input_ids=input_ids)

        # # SMART: Apply perturbation
        if perturb:
            noise = torch.randn_like(embedding_output, requires_grad=True) * self.noise_var
            embedding_output += noise

        # Feed to a transformer (a stack of BertLayers).
        sequence_output = self.encode(embedding_output, attention_mask, task)

        # Get cls token hidden state.
        first_tk = sequence_output[:, 0]
        first_tk = self.pooler_dense(first_tk)
        first_tk = self.pooler_af(first_tk)

        return {'last_hidden_state': sequence_output, 'pooler_output': first_tk}

