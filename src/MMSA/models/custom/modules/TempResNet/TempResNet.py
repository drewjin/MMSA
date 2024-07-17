import torch
import torch.nn as nn

from easydict import EasyDict as edict
from transformers import ResNetModel

__all__ = ['TempResNet']

class TempResNet(nn.Module):
    def __init__(self, args) -> None:
        super(TempResNet, self).__init__()
        self.resnet_weight = r'/home/drew/Desktop/Research/weights/resnet/resnet-18'
        self.resnet = ResNetModel.from_pretrained(self.resnet_weight)
        self.v_o_size = 512
        self.hidden_size = self.v_o_size * 2
        self.lstm = nn.LSTMCell(input_size=self.v_o_size, hidden_size=self.hidden_size)    

    def forward(self, x):
        device = x.device
        hidden = torch.zeros(x.shape[0], self.hidden_size).to(device)
        cell = torch.zeros(x.shape[0], self.hidden_size).to(device)

        x = x.permute(1, 0, 4, 2, 3)
        hidden_states = []
        cell_states = []
        for x_i in x:
            resnet_output = self.resnet(x_i)
            v_feature = resnet_output.pooler_output.squeeze()
            prev_hidden, prev_cell = hidden, cell
            new_hidden, new_cell = self.lstm(v_feature, (prev_hidden, prev_cell))
            hidden, cell = new_hidden, new_cell
            hidden_states.append(new_hidden) 
            cell_states.append(new_cell)
        
        last_hidden_state = hidden_states[-1]
        hidden_states = torch.stack(hidden_states).permute(1, 0, 2)
        cell_states = torch.stack(cell_states).permute(1, 0, 2)
        return edict({
            'last_hidden_state': last_hidden_state,
            'hidden_states': hidden_states,
            'cell_states': cell_states
        })


if __name__ == '__main__':
    model = TempResNet(None).cuda()
    ans = model(torch.randn(32, 15, 64, 64, 3).cuda())
    print(ans)