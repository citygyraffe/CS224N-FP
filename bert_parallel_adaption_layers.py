import torch
import torch.nn as nn
import torch.nn.functional as F

from bert import BertLayer, BertModel
from utils import *

DEBUG_OUTPUT = False
NUM_TASKS_SUPPORTED = 3

class BertLayerWithParallelAdaption(BertLayer):
    def __init__(self, config, args):
        super(BertLayerWithParallelAdaption, self).__init__(config)
        
        self.device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

        # This size was borrowed directly from Bert and PALS
        self.adaption_layer_size_pals = 204
        self.adaption_layer_size_low_rank = 100
        
        # Set by the adaption layer type
        self.adaption_layer_size = None
        self.Vd = None
        self.Ve = None

        # These lists hold layers and parameters for each task
        #self.Vd_list = []
        #self.Ve_list = []
        self.adaption_layer_list_first = []
        self.adaption_layer_list_second = []

        if args.parallel_adaption_layers == 'low-rank':
            self.adaption_layer_size = self.adaption_layer_size_low_rank

            for _ in range(NUM_TASKS_SUPPORTED):
                #self.adaption_layer_list.append(nn.Linear(self.adaption_layer_size, self.adaption_layer_size).to(self.device))
                #self.Vd_list.append(nn.Parameter(torch.randn(self.adaption_layer_size, config.hidden_size, requires_grad=True)).to(self.device))
                #self.Ve_list.append(nn.Parameter(torch.randn(config.hidden_size, self.adaption_layer_size, requires_grad=True)).to(self.device))
                self.adaption_layer_list_first.append(nn.Linear(config.hidden_size, self.adaption_layer_size).to(self.device))
                self.adaption_layer_list_second.append(nn.Linear(self.adaption_layer_size, config.hidden_size).to(self.device))
        
        elif args.parallel_adaption_layers == 'pals':
            self.adaption_layer_size = self.adaption_layer_size_pals
            raise NotImplementedError('PALS adaption layer not yet implemented')
        else:
            raise ValueError(f'Invalid adaption_layer_type: {args.parallel_adaption_layers}')
    
    def forward_parallel_adaption(self, hidden_states, task):
        # 1. Adaption layer. TS(h) = Vd * g(Ve * h)
        # g is a low-rank linear transformation for 'low-rank' adaption layers
        # g is multi-head attention for 'pals' adaption layers

        #adaption_layer = self.adaption_layer_list[task]
        #self.Vd = self.Vd_list[task]
        #self.Ve = self.Ve_list[task]

        batch_size, seq_length, hidden_size = hidden_states.size()
        flattened_hidden_states = hidden_states.view(-1, hidden_size)
        
        if DEBUG_OUTPUT:
            print("TASK", task)
            print("SIZEOF hidden_states", hidden_states.size())
            print("SIZEOF Ve", self.Ve.size())
            print("SIZEOF Ve transpose", self.Ve.transpose(0, 1).size())
            print("SIZEOF Vd", self.Vd.size())
            print("SIZEOF flattened_hidden_states", flattened_hidden_states.size())

        #intermediate_output = torch.matmul(flattened_hidden_states, self.Ve)
        #intermediate_output = adaption_layer(intermediate_output)
        #adaption_output = torch.matmul(intermediate_output, self.Vd)
            
        adaption_output = self.adaption_layer_list_first[task](flattened_hidden_states)
        adaption_output = F.gelu(adaption_output)
        adaption_output = self.adaption_layer_list_second[task](adaption_output)
        adaption_output = adaption_output.view(batch_size, seq_length, hidden_size)

        if DEBUG_OUTPUT:
            print("SIZEOF adaption_output", adaption_output.size())

        return adaption_output

    #OVERRIDE
    def forward(self, hidden_states, attention_mask, task):
        # 1. Multi-head attention layer. MH(h)
        attn_output = self.self_attention(hidden_states, attention_mask)

        # 1a. Parallel adaption layer. TS(h)
        adaption_output = self.forward_parallel_adaption(hidden_states, task)

        # 2. Add-norm for multi-head attention. LN(h + MH(h))
        attn_output_normalized = self.add_norm(hidden_states, attn_output + adaption_output, self.attention_dense, self.attention_dropout, self.attention_layer_norm)

        # 3. Feed forward layer. SA(h) = FFN(LN(h + MH(h)))
        interm_output = self.interm_af(self.interm_dense(attn_output_normalized))

        if DEBUG_OUTPUT:
            print("SIZEOF interm_output", interm_output.size()) 
            print("SIZEOF attn_output_normalized", attn_output_normalized.size())

        # 4. Add-norm for feed forward. Original: LN(h + SA(h)) Now: LN(h + SA(h) + TS(h))
        output = self.add_norm(attn_output_normalized, interm_output, self.out_dense, self.out_dropout, self.out_layer_norm)

        return output

class BertModelWithParallelAdaption(BertModel):
    def __init__(self, config, args):
        super(BertModelWithParallelAdaption, self).__init__(config)

        # Replace the original BertLayer with the new BertLayerWithAdaption
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

    def forward(self, input_ids, attention_mask, task):
        """
        input_ids: [batch_size, seq_len], seq_len is the max length of the batch
        attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
        """
        # Get the embedding for each input token.
        embedding_output = self.embed(input_ids=input_ids)

        # Feed to a transformer (a stack of BertLayers).
        sequence_output = self.encode(embedding_output, attention_mask, task)

        # Get cls token hidden state.
        first_tk = sequence_output[:, 0]
        first_tk = self.pooler_dense(first_tk)
        first_tk = self.pooler_af(first_tk)

        return {'last_hidden_state': sequence_output, 'pooler_output': first_tk}

