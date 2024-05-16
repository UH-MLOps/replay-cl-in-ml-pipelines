import torch
import torch.nn as nn
import lightning as L

INPUT_DIM = 14
OUTPUT_DIM = 1
HIDDEN_SIZE = 10

class ShipPerformanceModel(L.LightningModule):
    def __init__(self, loss_fn, optimizer_config):
        super().__init__()
        self.loss_fn = loss_fn
        self.optimizer = optimizer_config['optimizer']
        self.lr = optimizer_config['lr']
        
        # layers
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc4 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc5 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc6 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc7 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc8 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc9 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc10 = nn.Linear(HIDDEN_SIZE, OUTPUT_DIM)
        self.bn = nn.BatchNorm1d(HIDDEN_SIZE)

        
        # activation
        self.lrelu = nn.LeakyReLU(0.2)
        
        # initialize the weights
        self.init_weights()


    def init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.kaiming_normal_(self.fc4.weight)
        nn.init.kaiming_normal_(self.fc5.weight)
        nn.init.kaiming_normal_(self.fc6.weight)
        nn.init.kaiming_normal_(self.fc7.weight)
        nn.init.kaiming_normal_(self.fc8.weight)
        nn.init.kaiming_normal_(self.fc9.weight)
        nn.init.kaiming_normal_(self.fc10.weight)

        
    def forward(self, features):
        output = self.fc1(features)
        output = self.lrelu(output)  
        output = self.fc2(output)
        output = self.lrelu(output)
        output = self.fc3(output)
        output = self.lrelu(output)
        output = self.fc4(output)
        output = self.lrelu(output)
        output = self.fc5(output)
        output = self.lrelu(output)       
        output = self.fc6(output)
        output = self.lrelu(output)
        output = self.fc7(output)
        output = self.lrelu(output)
        output = self.fc8(output)
        output = self.lrelu(output)
        output = self.fc9(output)

        output = self.lrelu(output)
        output = self.fc10(output)
        return output

    
    def training_step(self, batch):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log('train_loss', loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log('val_loss', loss, on_epoch=True)
        return loss

    def predict_step(self, batch):
        x, y = batch
        return self(x)


    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)
    
    
    def save_model(self, optimizer, path):
        state = {'state_dict': self.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state, path)

