import torch.nn as nn
import torch as th
import numpy as np
from torch.autograd import Function
from einops import rearrange, repeat, reduce

from typing import Tuple, Union, List
import cv2

__author__ = "Manuel Traub"

class EpropLSTMFunction(Function):
    @staticmethod
    def forward(ctx, x, w_ix, w_ih, b_i, w_fx, w_fh, b_f, w_ox, w_oh, b_o, w_cx, w_ch, b_c, args):

        h_last, c_last, e_w_ix, e_w_ih, e_b_i, e_w_fx, e_w_fh, e_b_f, e_w_cx, e_w_ch, e_b_c = args

        i     = th.sigmoid(x.mm(w_ix.t()) + h_last.mm(w_ih.t()) + b_i)
        f     = th.sigmoid(x.mm(w_fx.t()) + h_last.mm(w_fh.t()) + b_f)
        o     = th.sigmoid(x.mm(w_ox.t()) + h_last.mm(w_oh.t()) + b_o)
        c_hat = th.tanh(   x.mm(w_cx.t()) + h_last.mm(w_ch.t()) + b_c)

        c = f * c_last + i * c_hat
        h = o * c

        di = i * (1 - i)
        df = f * (1 - f)
        do = o * (1 - o)
        dc_hat = (1 - c_hat**2)

        i_j = i.unsqueeze(dim=2)
        f_j = f.unsqueeze(dim=2)
        o_j = o.unsqueeze(dim=2)

        di_j = di.unsqueeze(dim=2)
        df_j = df.unsqueeze(dim=2)
        do_j = do.unsqueeze(dim=2)
        dc_hat_j = dc_hat.unsqueeze(dim=2)

        x_i      = x.unsqueeze(dim=1)
        h_last_i = h_last.unsqueeze(dim=1)
        c_last_j = c_last.unsqueeze(dim=2)
        c_hat_j  = c_hat.unsqueeze(dim=2)
        c_j      = c.unsqueeze(dim=2)

        e_w_ih.copy_(e_w_ih * f_j + di_j * h_last_i * c_hat_j)
        e_w_ix.copy_(e_w_ix * f_j + di_j * x_i      * c_hat_j)
        e_b_i.copy_( e_b_i  * f   + di              * c_hat  )

        e_w_fh.copy_(e_w_fh * f_j + df_j * h_last_i * c_last_j)
        e_w_fx.copy_(e_w_fx * f_j + df_j * x_i      * c_last_j)
        e_b_f.copy_( e_b_f  * f   + df              * c_last  )

        e_w_ch.copy_(e_w_ch * f_j + dc_hat_j * h_last_i * i_j)
        e_w_cx.copy_(e_w_cx * f_j + dc_hat_j * x_i      * i_j)
        e_b_c.copy_( e_b_c  * f   + dc_hat              * i  )

        ctx.e_w_ix = e_w_ix 
        ctx.e_w_ih = e_w_ih 
        ctx.e_b_i  = e_b_i  
        ctx.e_w_fx = e_w_fx 
        ctx.e_w_fh = e_w_fh 
        ctx.e_b_f  = e_b_f  
        ctx.e_w_cx = e_w_cx 
        ctx.e_w_ch = e_w_ch 
        ctx.e_b_c  = e_b_c  

        ctx.i        = i       
        ctx.o        = o        
        ctx.c        = c        
        ctx.o_j      = o_j      
        ctx.c_j      = c_j      
        ctx.do       = do       
        ctx.df       = df       
        ctx.di       = di       
        ctx.dc_hat   = dc_hat   
        ctx.di_j     = di_j     
        ctx.do_j     = do_j     
        ctx.h_last_i = h_last_i 
        ctx.c_last   = c_last  
        ctx.c_hat    = c_hat    
        ctx.x_i      = x_i      
        ctx.w_ox     = w_ox     
        ctx.w_fx     = w_fx    
        ctx.w_ix     = w_ix    
        ctx.w_cx     = w_cx    

        return h, c

    @staticmethod
    def backward(ctx, dh, _):

        e_w_ix = ctx.e_w_ix 
        e_w_ih = ctx.e_w_ih 
        e_b_i  = ctx.e_b_i  
        e_w_fx = ctx.e_w_fx 
        e_w_fh = ctx.e_w_fh 
        e_b_f  = ctx.e_b_f  
        e_w_cx = ctx.e_w_cx 
        e_w_ch = ctx.e_w_ch 
        e_b_c  = ctx.e_b_c  

        i        = ctx.i       
        o        = ctx.o        
        c        = ctx.c        
        o_j      = ctx.o_j      
        c_j      = ctx.c_j      
        do       = ctx.do       
        df       = ctx.df       
        di       = ctx.di       
        dc_hat   = ctx.dc_hat   
        di_j     = ctx.di_j     
        do_j     = ctx.do_j     
        h_last_i = ctx.h_last_i 
        c_last   = ctx.c_last  
        c_hat    = ctx.c_hat    
        x_i      = ctx.x_i      
        w_ox     = ctx.w_ox     
        w_fx     = ctx.w_fx    
        w_ix     = ctx.w_ix    
        w_cx     = ctx.w_cx    

        dh_j = dh.unsqueeze(dim=2)

        dw_ix = th.sum(dh_j * o_j * e_w_ix, dim=0)
        dw_ih = th.sum(dh_j * o_j * e_w_ih, dim=0)
        db_i  = th.sum(dh   * o   * e_b_i , dim=0)

        dw_fx = th.sum(dh_j * o_j * e_w_fx, dim=0)
        dw_fh = th.sum(dh_j * o_j * e_w_fh, dim=0)
        db_f  = th.sum(dh   * o   * e_b_f , dim=0)

        dw_cx = th.sum(dh_j * o_j * e_w_cx, dim=0)
        dw_ch = th.sum(dh_j * o_j * e_w_ch, dim=0)
        db_c  = th.sum(dh   * o   * e_b_c , dim=0)

        dw_oh = th.sum(dh_j * do_j * h_last_i * c_j, dim=0)
        dw_ox = th.sum(dh_j * do_j * x_i      * c_j, dim=0)
        db_o  = th.sum(dh   * do              * c  , dim=0)

        dh_do = dh * do * c
        dh_df = dh * df * c_last * o
        dh_di = dh * di * c_hat * o
        dh_dc_hat = dh * dc_hat * i * o

        dx = dh_do.mm(w_ox) + dh_df.mm(w_fx) + dh_di.mm(w_ix) + dh_dc_hat.mm(w_cx)

        return dx, dw_ix, dw_ih, db_i, dw_fx, dw_fh, db_f, dw_ox, dw_oh, db_o, dw_cx, dw_ch, db_c, None

class EpropLSTM(nn.Module):
    def __init__(
        self, 
        num_inputs,
        num_hidden,
        batch_size
    ):
        super(EpropLSTM, self).__init__()

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden

        self.fcn = EpropLSTMFunction().apply

        # input gate weights and biases
        self.w_ix = nn.Parameter(th.empty(num_hidden, num_inputs))
        self.w_ih = nn.Parameter(th.empty(num_hidden, num_hidden))
        self.b_i  = nn.Parameter(th.ones(num_hidden))

        # forget gate weights and biases
        self.w_fx = nn.Parameter(th.empty(num_hidden, num_inputs))
        self.w_fh = nn.Parameter(th.empty(num_hidden, num_hidden))
        self.b_f  = nn.Parameter(th.ones(num_hidden))

        # output gate weights and biases
        self.w_ox = nn.Parameter(th.empty(num_hidden, num_inputs))
        self.w_oh = nn.Parameter(th.empty(num_hidden, num_hidden))
        self.b_o  = nn.Parameter(th.ones(num_hidden))

        # cell weights and biases
        self.w_cx = nn.Parameter(th.empty(num_hidden, num_inputs))
        self.w_ch = nn.Parameter(th.empty(num_hidden, num_hidden))
        self.b_c  = nn.Parameter(th.zeros(num_hidden))

        # input gate eligibilitiy traces
        self.register_buffer("e_w_ix", th.zeros(batch_size, num_hidden, num_inputs), persistent=False)
        self.register_buffer("e_w_ih", th.zeros(batch_size, num_hidden, num_hidden), persistent=False)
        self.register_buffer("e_b_i",  th.zeros(batch_size, num_hidden),             persistent=False)

        # forget gate eligibilitiy traces
        self.register_buffer("e_w_fx", th.zeros(batch_size, num_hidden, num_inputs), persistent=False)
        self.register_buffer("e_w_fh", th.zeros(batch_size, num_hidden, num_hidden), persistent=False)
        self.register_buffer("e_b_f",  th.zeros(batch_size, num_hidden),             persistent=False)

        # cell eligibilitiy traces
        self.register_buffer("e_w_cx", th.zeros(batch_size, num_hidden, num_inputs), persistent=False)
        self.register_buffer("e_w_ch", th.zeros(batch_size, num_hidden, num_hidden), persistent=False)
        self.register_buffer("e_b_c",  th.zeros(batch_size, num_hidden),             persistent=False)

        # cell sate and hidden state
        self.register_buffer("h_last", th.zeros(batch_size, num_hidden),             persistent=False)
        self.register_buffer("c_last", th.zeros(batch_size, num_hidden),             persistent=False)

        # initialize weights
        stdv_i = np.sqrt(6/(self.num_inputs + self.num_hidden))
        stdv_h = np.sqrt(3/self.num_hidden)

        nn.init.uniform_(self.w_ix, -stdv_i, stdv_i)
        nn.init.uniform_(self.w_ih, -stdv_h, stdv_h)

        nn.init.uniform_(self.w_fx, -stdv_i, stdv_i)
        nn.init.uniform_(self.w_fh, -stdv_h, stdv_h)

        nn.init.uniform_(self.w_ox, -stdv_i, stdv_i)
        nn.init.uniform_(self.w_oh, -stdv_h, stdv_h)

        nn.init.uniform_(self.w_cx, -stdv_i, stdv_i)
        nn.init.uniform_(self.w_ch, -stdv_h, stdv_h)

    def reset_state(self):
        self.h_last.zero_() 
        self.c_last.zero_() 
        self.e_w_ix.zero_() 
        self.e_w_ih.zero_() 
        self.e_b_i.zero_() 
        self.e_w_fx.zero_() 
        self.e_w_fh.zero_() 
        self.e_b_f.zero_() 
        self.e_w_cx.zero_() 
        self.e_w_ch.zero_() 
        self.e_b_c.zero_()

    def forward(self, x: th.Tensor):
        self.h_last.detach_()
        self.c_last.detach_()

        h, c = self.fcn(
            x, 
            self.w_ix, self.w_ih, self.b_i, 
            self.w_fx, self.w_fh, self.b_f, 
            self.w_ox, self.w_oh, self.b_o, 
            self.w_cx, self.w_ch, self.b_c,
            (
                self.h_last, self.c_last, 
                self.e_w_ix, self.e_w_ih, self.e_b_i, 
                self.e_w_fx, self.e_w_fh, self.e_b_f, 
                self.e_w_cx, self.e_w_ch, self.e_b_c
            )
        )

        self.h_last = h
        self.c_last = c

        return h
