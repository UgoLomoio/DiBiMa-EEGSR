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
        x = x.to(self.parameters().__next__().device)
        x = self.features(x)
        x = x.view(-1,2048)
        x = self.classifer(x)
        return x
    
class ResNetPL(pl.LightningModule):

    def __init__(self, model, optimizer, criterion):
        super(ResNetPL, self).__init__()
        self.model = model.to(torch.float32)  # ALL params float32
        self.model = self.model.float()
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def forward(self, x, return_embeddings=False):
        x = x.to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=-1)   # multiclass
        preds = torch.argmax(logits, dim=-1)
        if return_embeddings:
            embeddings = self.model.features(x)
            embeddings = embeddings.view(embeddings.size(0), -1)
            return logits, probs, preds, embeddings
        else:
            return logits, probs, preds  # tuple

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        logits, probs, preds = self(inputs)
        loss = self.criterion(logits, targets)
        self.log('train_loss', loss, prog_bar=True)
        acc = (preds == targets).float().mean()
        self.log('train_acc', acc, prog_bar=True)
        self.train_losses.append(loss.item())  # safe now
        self.train_accs.append(acc.item())
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        logits, probs, preds = self(inputs)
        loss = self.criterion(logits, targets)
        acc = (preds == targets).float().mean()
        self.val_losses.append(loss.item())
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
        avg_train_loss = torch.mean(torch.tensor(self.train_losses))
        avg_train_acc = torch.mean(torch.tensor(self.train_accs))
        self.log('avg_train_loss', avg_train_loss, prog_bar=True, on_epoch=True)
        self.log('avg_train_acc', avg_train_acc, prog_bar=True, on_epoch=True)
        return super().on_train_epoch_end()
    
    def on_validation_epoch_end(self):
        avg_val_loss = torch.mean(torch.tensor(self.val_losses))
        avg_val_acc = torch.mean(torch.tensor(self.val_accs))
        self.log('avg_val_acc', avg_val_acc, prog_bar=True, on_epoch=True)
        self.log('avg_val_loss', avg_val_loss, prog_bar=True, on_epoch=True)
        return super().on_validation_epoch_end()
    
    def configure_optimizers(self):
        optimizer = self.optimizer
        return optimizer

    def predict(self, dataloader):
        self.eval()
        all_logits = []
        all_probs = []
        all_preds = []
        with torch.no_grad():
            for batch in dataloader:
                inputs, _ = batch
                inputs = inputs.float().to(self.device)
                logits, probs, preds = self(inputs)
                all_logits.append(logits.cpu())
                all_probs.append(probs.cpu())
                all_preds.append(preds.cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_probs = torch.cat(all_probs, dim=0)
        all_preds = torch.cat(all_preds, dim=0)
        return all_logits, all_probs, all_preds