import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralkg.model.KGEModel.model import Model
from inspect import stack


class FNetBlock(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    re_x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
    return re_x

class FNetEReal(Model):
    def __init__(self, args):
        super(FNetEReal, self).__init__(args)
        self.args = args      
        self.emb_ent = None
        self.emb_rel = None
        self.init_emb(args)

    def init_emb(self, args):

        """Initialize the convolution layer and embeddings .

        Args:
            conv1: The convolution layer.
            fc: The full connection layer.
            bn0, bn1, bn2: The batch Normalization layer.
            inp_drop, hid_drop, feg_drop: The dropout layer.
            emb_ent: Entity embedding, shape:[num_ent, emb_dim].
            emb_rel: Relation_embedding, shape:[num_rel, emb_dim].
        """
        self.emb_dim = self.args.emb_dim
        self.inp_drop = self.args.inp_drop
        self.emb_ent = nn.Embedding(self.args.num_ent, self.emb_dim, padding_idx=0)
        self.emb_rel = nn.Embedding(self.args.num_rel, self.emb_dim, padding_idx=0)
        nn.init.xavier_normal_(self.emb_ent.weight.data)
        nn.init.xavier_normal_(self.emb_rel.weight.data)
        self.register_parameter('b', nn.Parameter(torch.zeros(self.args.num_ent)))
        
        ### initial layer norm
        self.attn_layernorms_ent = nn.ModuleList()
        self.attn_layernorms_rel = nn.ModuleList()

        ### attnetion layer
        self.attn_layers = nn.ModuleList()
        
        ### skip_connection layer norm
        self.sc_layernorms = nn.ModuleList()
        
        for layer in range(self.args.nblocks):
            layer_norm_ent = nn.LayerNorm(self.emb_dim, eps=1e-8)
            layer_norm_rel = nn.LayerNorm(self.emb_dim, eps=1e-8)
            self.attn_layernorms_ent.append(layer_norm_ent)
            self.attn_layernorms_rel.append(layer_norm_rel)
            attn_real = FNetBlock()
            self.attn_layers.append(attn_real)
            sc_layernorm = nn.LayerNorm(self.emb_dim, eps=1e-8)
            self.sc_layernorms.append(sc_layernorm)
        
        self.ffn_output = nn.Sequential(nn.Linear(self.emb_dim, self.args.dim_feedforward),
                                        nn.GELU(),
                                        nn.Dropout(self.args.hid_drop),
                                        nn.Linear(self.args.dim_feedforward, self.emb_dim),
                                        nn.Dropout(self.args.hid_drop))
        self.last_layernorms = nn.LayerNorm(self.emb_dim, eps=1e-8)


    def score_func(self, head_emb, relation_emb, choose_emb = None):

        """Calculate the score of the triple embedding.

        This function calculate the score of the embedding.
        First, the entity and relation embeddings are reshaped
        and concatenated; the resulting matrix is then used as
        input to a convolutional layer; the resulting feature
        map tensor is vectorised and projected into a k-dimensional
        space.

        Args:
            head_emb: The embedding of head entity.
            relation_emb:The embedding of relation.

        Returns:
            score: Final score of the embedding.
        """
        # import pdb;pdb.set_trace()
        for layer in range(self.args.nblocks):
            Q_ent = self.attn_layernorms_ent[layer](head_emb)
            Q_rel = self.attn_layernorms_rel[layer](relation_emb)
            concat_emb = torch.cat([Q_ent, Q_rel], dim=1)
            real_attn_out = self.attn_layers[layer](concat_emb) # [B, 2, emb_dim]
            real_attn_out = self.sc_layernorms[layer](real_attn_out + concat_emb) # [B, 2, emb_dim]
            if layer <= (self.args.nblocks - 1):
                head_emb, relation_emb = torch.chunk(real_attn_out, 2, dim=1)

        merged_out = head_emb * relation_emb
        x = self.ffn_output(merged_out.squeeze())
        x = self.last_layernorms(x)
        x = torch.mm(x, self.emb_ent.weight.transpose(1,0)) if choose_emb == None else torch.mm(x, choose_emb.transpose(1, 0)) 
        x += self.b.expand_as(x)
        x = torch.sigmoid(x)
        return x

    def forward(self, triples):

        """The functions used in the training phase

        Args:
            triples: The triples ids, as (h, r, t), shape:[batch_size, 3].

        Returns:
            score: The score of triples.
        """
        head_emb = self.emb_ent(triples[:, 0]).view(-1, 1, self.emb_dim)
        rela_emb = self.emb_rel(triples[:, 1]).view(-1, 1, self.emb_dim)
        score = self.score_func(head_emb, rela_emb)
        return score

    def get_score(self, batch, mode="tail_predict"):

        """The functions used in the testing phase

        Args:
            batch: A batch of data.

        Returns:
            score: The score of triples.
        """
        
        triples = batch["positive_sample"]
        head_emb = self.emb_ent(triples[:, 0]).view(-1, 1, self.emb_dim)
        rela_emb = self.emb_rel(triples[:, 1]).view(-1, 1, self.emb_dim)
        score = self.score_func(head_emb, rela_emb)
        return score