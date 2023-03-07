import torch
import torch.nn as nn
import torch.nn.functional as F 
# XLA imports
    
class NCF(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers,
                    dropout, model, GMF_model=None, MLP_model=None):
        super(NCF, self).__init__()
        self.dropout = dropout
        self.model = model
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model

        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        self.embed_user_MLP = nn.Embedding(
                user_num, factor_num * (2 ** (num_layers - 1)))
        self.embed_item_MLP = nn.Embedding(
                item_num, factor_num * (2 ** (num_layers - 1)))

        self.linear = torch.nn.Linear(4, 4)
        self.MLP_layers = nn.Sequential(
            nn.Dropout(p=0.0, inplace=False), 
            nn.Linear(in_features=256,out_features=128, bias=True), 
            nn.ReLU(), 
            nn.Dropout(p=0.0, inplace=False), 
            nn.Linear(in_features=128, out_features=64, bias=True), 
            nn.ReLU(), 
            nn.Dropout(p=0.0, inplace=False), 
            nn.Linear(in_features=64, out_features=32, bias=True), 
            nn.ReLU()            
        )
    
        predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 1)

        self._init_weight_()
        
    def forward(self, user, item):
        '''
        # PyTorch --> Triton 수정 사항
        - if statement 제거
        '''
        embed_user_GMF = self.embed_user_GMF(user)
        embed_item_GMF = self.embed_item_GMF(item)
        output_GMF = embed_user_GMF * embed_item_GMF

        embed_user_MLP = self.embed_user_MLP(user)
        embed_item_MLP = self.embed_item_MLP(item)
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
        output_MLP = self.MLP_layers(interaction)



        concat = torch.cat((output_GMF, output_MLP), -1)


        prediction = self.predict_layer(concat)
  
        print("#### test 1 #####")
        print("#### prediction size: \n", prediction.size())        
        print("#### prediction: \n", prediction)    
        return torch.cat((user,item),1)            
#         return prediction
#         return torch.cat((prediction,prediction),1)


    def _init_weight_(self):
        """ We leave the weights initialization here. """
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, 
                                a=1, nonlinearity='sigmoid')

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

