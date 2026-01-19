import torch
import pytorch_lightning as pl

class Bottleneck(torch.nn.Module):
    def __init__(self,In_channel,Med_channel,Out_channel,downsample=False):
        super(Bottleneck, self).__init__()
        self.stride = 1
        if downsample == True:
            self.stride = 2

        self.layer = torch.nn.Sequential(
            torch.nn.Conv1d(In_channel, Med_channel, 1, self.stride),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Med_channel, 3, padding=1),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Out_channel, 1),
            torch.nn.BatchNorm1d(Out_channel),
            torch.nn.ReLU(),
        )

        if In_channel != Out_channel:
            self.res_layer = torch.nn.Conv1d(In_channel, Out_channel,1,self.stride)
        else:
            self.res_layer = None

    def forward(self,x):
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        return self.layer(x)+residual


class ResNet50(torch.nn.Module):
    def __init__(self,in_channels=2,classes=125):
        super(ResNet50, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels,64,kernel_size=7,stride=2,padding=3),
            torch.nn.MaxPool1d(3,2,1),

            Bottleneck(64,64,256,False),
            Bottleneck(256,64,256,False),
            Bottleneck(256,64,256,False),
            #
            Bottleneck(256,128,512, True),
            Bottleneck(512,128,512, False),
            Bottleneck(512,128,512, False),
            Bottleneck(512,128,512, False),
            #
            Bottleneck(512,256,1024, True),
            Bottleneck(1024,256,1024, False),
            Bottleneck(1024,256,1024, False),
            Bottleneck(1024,256,1024, False),
            Bottleneck(1024,256,1024, False),
            Bottleneck(1024,256,1024, False),
            #
            Bottleneck(1024,512,2048, True),
            Bottleneck(2048,512,2048, False),
            Bottleneck(2048,512,2048, False),

            torch.nn.AdaptiveAvgPool1d(1)
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(2048,classes)
        )

    def forward(self,x):
        x = self.features(x)
        x = x.view(-1,2048)
        x = self.classifer(x)
        return x
    

class ResNetPL(pl.LightningModule):

    def __init__(self, model, optimizer):
        super(ResNetPL, self).__init__()
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        targets = targets.long()
        loss = self.criterion(outputs, targets)
        self.train_losses.append(loss.item())
        self.log('train_loss', loss, prog_bar=True)
        acc = (outputs.argmax(dim=1) == targets).float().mean()
        self.train_accs.append(acc.item())
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        targets = targets.long()
        loss = self.criterion(outputs, targets)
        self.val_losses.append(loss.item())
        acc = (outputs.argmax(dim=1) == targets).float().mean()
        self.val_accs.append(acc.item())
        return loss
    
    def on_train_epoch_start(self):
        self.train_losses = []
        self.train_accs = []
        return super().on_train_epoch_start()

    def on_validation_epoch_start(self):
        self.val_losses = []
        self.val_accs = []
        return super().on_validation_epoch_start()
    
    def on_train_epoch_end(self):
        avg_train_loss = sum(self.train_losses) / len(self.train_losses)
        avg_train_acc = sum(self.train_accs) / len(self.train_accs)
        self.log('avg_train_loss', avg_train_loss, prog_bar=True, on_epoch=True)
        self.log('avg_train_acc', avg_train_acc, prog_bar=True, on_epoch=True)
        return super().on_train_epoch_end()
    
    def on_validation_epoch_end(self):
        avg_val_loss = sum(self.val_losses) / len(self.val_losses)
        avg_val_acc = sum(self.val_accs) / len(self.val_accs)
        self.log('avg_val_acc', avg_val_acc, prog_bar=True, on_epoch=True)
        self.log('avg_val_loss', avg_val_loss, prog_bar=True, on_epoch=True)
        return super().on_validation_epoch_end()
    
    def configure_optimizers(self):
        optimizer = self.optimizer
        return optimizer

    def predict(self, dataloader):
        self.eval()
        all_outputs = []
        with torch.no_grad():
            for batch in dataloader:
                inputs, _ = batch
                outputs = self(inputs)
                all_outputs.append(outputs)
        return torch.cat(all_outputs, dim=0)
