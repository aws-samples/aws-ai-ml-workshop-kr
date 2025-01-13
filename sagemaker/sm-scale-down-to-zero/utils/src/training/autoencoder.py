import torch
import torch.nn as nn

class AutoEncoder(nn.Module):

    def __init__(self, encoder, decoder, emb_size):
        
        super(AutoEncoder, self).__init__()
        
        self.emb = nn.Embedding(num_embeddings=24, embedding_dim=emb_size)
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        # |x| = (batch_size, #c, w, h)
        z = self.encoder(x)
        return z.view(x.size(0), -1)

    def decode(self, z):
        # |z| = (batch_size, btl_size)
        y = self.decoder(z)
        return y

    def forward(self, time, x):
        
        t_emb = self.emb(time)
        x = torch.cat([t_emb, x], dim=1)
        z = self.encode(x)
        x_hat = self.decode(z)
        x_hat = x_hat
        
        return t_emb, x_hat.view(x.size(0), -1)
    
class FCLayer(nn.Module):
    
    def __init__(self, input_size, output_size=1, bias=True, last_act=True, bn=False, dropout_p=0):
        
        super().__init__()
        self.layer = nn.Linear(input_size, output_size, bias)
        self.bn = nn.BatchNorm1d(output_size) if bn else None
        self.dropout = nn.Dropout(dropout_p) if dropout_p else None
        if last_act: self.act = None
        else: self.act = nn.LeakyReLU(.2)
    
    def forward(self, x):
         
        y = self.act(self.layer(x)) if self.act else self.layer(x)
        y = self.bn(y) if self.bn else y
        y = self.dropout(y) if self.dropout else y

        return y

class FCModule(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=None, use_batch_norm=True, dropout_p=0):

        super().__init__()
        self.layer_list = []

        if use_batch_norm and dropout_p > 0:
            raise Exception("Either batch_norm or dropout is allowed, not both")

        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for idx, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            if idx < len(hidden_sizes):
                layer = FCLayer(
                    input_size=in_size,
                    output_size=out_size,
                    last_act=False,
                    bn=use_batch_norm,
                    dropout_p=dropout_p
                )
            else:
                layer = FCLayer(
                    input_size=in_size,
                    output_size=out_size,
                    last_act=True,
                )

            self.layer_list.append(layer)
        self.net = nn.Sequential(*self.layer_list)

    def forward(self, x):
        return self.net(x)
    
def get_model(input_dim, hidden_sizes, btl_size, emb_size):
    
    # args
    input_size = input_dim
    
    encoder = FCModule(
        input_size=input_size,
        output_size=btl_size,
        hidden_sizes=hidden_sizes, 
        use_batch_norm=True
    )
    
    decoder = FCModule(
        input_size=btl_size,
        output_size=input_size,
        hidden_sizes=list(reversed(hidden_sizes)),
        use_batch_norm=True,
    )

    model = AutoEncoder(
        encoder=encoder,
        decoder=decoder,
        emb_size=emb_size
    )

    return model
