import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralkg.model.KGEModel.model import Model
from inspect import stack


class FNetBlock(nn.Module):
  def __init__(self, dropout=0.1):
    super().__init__()
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    re_x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
    img_x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).imag
    re_x = self.dropout(re_x)
    imag_x = self.dropout(img_x)
    return re_x, imag_x

class FNetE2(Model):
    def __init__(self, args):
        super(FNetE2, self).__init__(args)
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
        self.attn_layernorms = nn.ModuleList()
        ### skip_connection layer norm
        self.sc_layernorms = nn.ModuleList()
        ### attnetion layer
        self.attn_layers_real_imag = nn.ModuleList()
        
        for layer in range(self.args.nblocks):
            layer_norm = nn.LayerNorm(self.emb_dim, eps=1e-8)
            self.attn_layernorms.append(layer_norm)
            attn_real_imag = FNetBlock(dropout=self.args.inp_drop)
            self.attn_layers_real_imag.append(attn_real_imag)
            sc_layernorm = nn.LayerNorm(self.emb_dim, eps=1e-8)
            self.sc_layernorms.append(sc_layernorm)
        
        self.hid_drop = nn.Dropout(self.args.hid_drop)
        self.ffn_output = nn.Sequential(nn.Linear(self.emb_dim, self.args.dim_feedforward),
                                        nn.GELU(),
                                        nn.Linear(self.args.dim_feedforward, self.emb_dim))
        self.last_layernorms = nn.LayerNorm(self.emb_dim, eps=1e-8)
        if self.args.fnete_opn == "tucker":
            self.W_td_Re = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (self.emb_dim, self.emb_dim, self.emb_dim)), dtype=torch.float32), requires_grad=True)
            self.W_td_img = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (self.emb_dim, self.emb_dim, self.emb_dim)), dtype=torch.float32), requires_grad=True)
            self.bn1 = nn.BatchNorm1d(self.emb_dim)

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
        for layer in range(self.args.nblocks):
            Q_ent = self.attn_layernorms[layer](head_emb)
            Q_rel = self.attn_layernorms[layer](relation_emb)
            concat_emb = torch.cat([Q_ent, Q_rel], dim=1)
            real_attn_out, imag_attn_out = self.attn_layers_real_imag[layer](concat_emb) # [B, 2, emb_dim]
            real_attn_out = self.sc_layernorms[layer](real_attn_out + concat_emb) # [B, 2, emb_dim]
            imag_attn_out = self.sc_layernorms[layer](imag_attn_out + concat_emb) # [B, 2, emb_dim]
            real_attn_ent, real_attn_rel = torch.chunk(real_attn_out, 2, dim=1) # [B, 1, emb_dim]
            imag_attn_ent, imag_attn_rel = torch.chunk(imag_attn_out, 2, dim=1) # [B, 1, emb_dim]
            if layer < (self.args.nblocks - 1):
                # head_emb, relation_emb = torch.chunk(merged_out, 2, dim=-1)
                head_emb = torch.fft.ifft(torch.complex(real_attn_ent, imag_attn_ent)).real # [B, 1, emb_dim]
                relation_emb = torch.fft.ifft(torch.complex(real_attn_rel, imag_attn_rel)).real # [B, 1, emb_dim]

        if self.args.fnete_opn == "mult": # (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            re_score = real_attn_ent * real_attn_rel - imag_attn_ent * imag_attn_rel # [B, 1, emb_dim]
            im_score = real_attn_ent * imag_attn_rel + imag_attn_ent * real_attn_rel # [B, 1, emb_dim]
            re_score = re_score.squeeze()
            im_score = im_score.squeeze()
        elif self.args.fnete_opn == "add": # (a+bi) + (c+di) = (a+c) + (b+d)i
            re_score = real_attn_ent + real_attn_rel # [B, 1, emb_dim]
            im_score = imag_attn_ent + imag_attn_rel # [B, 1, emb_dim]
            re_score = re_score.squeeze()
            im_score = im_score.squeeze()
        elif self.args.fnete_opn == "tucker":
            W_mat_Re = torch.mm(real_attn_rel.squeeze(), self.W_td_Re.view(self.emb_dim, -1)).view(-1, self.emb_dim, self.emb_dim)
            W_mat_img = torch.mm(imag_attn_rel.squeeze(), self.W_td_img.view(self.emb_dim, -1)).view(-1, self.emb_dim, self.emb_dim)
            W_mat_Re = self.hid_drop(W_mat_Re)
            W_mat_img = self.hid_drop(W_mat_img)
            re_score = torch.bmm(real_attn_ent, W_mat_Re) # [B, 1, emb_dim]
            im_score = torch.bmm(imag_attn_ent, W_mat_Re) # [B, 1, emb_dim]
            re_score = self.bn1(re_score.squeeze())
            im_score = self.bn1(im_score.squeeze())
        else:
            raise NotImplementedError("Unknown fnete_opn: {}".format(self.args.fnete_opn))
        
        merged_out = torch.complex(re_score, im_score)
        merged_out = torch.fft.ifft(merged_out).real
        # import pdb;pdb.set_trace()
        x = self.ffn_output(merged_out)
        x = self.last_layernorms(x)
        x = self.hid_drop(x)
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