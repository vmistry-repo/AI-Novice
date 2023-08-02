from pytorch_lightning import LightningModule
import torch

class CustomResnet(LightningModule):
    def __init__(self, data_dir=PATH_DATASETS, hidden_size=16, learning_rate=2e-4, dropout_value=0.05):
        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_value

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.transform = test_transforms


        # PrepLayer
        self.preplayer    = self.conv_block(in_channels=3, out_channels=64, kernel_size=(3,3), stride=1, padding=1)
        # Layer 1
        self.conv1_layer1 = self.conv_block(in_channels=64, out_channels=128, kernel_size=(3,3),
                                            stride=1, padding=1, max_pool=True)
        self.resblock1    = self.res_block(in_channels=128, out_channels=128, kernel_size=(3,3))
        # Layer 2
        self.conv2_layer2 = self.conv_block(in_channels=128, out_channels=256, kernel_size=(3,3),
                                            stride=1, padding=1, max_pool=True)
        # Layer 3
        self.conv3_layer3 = self.conv_block(in_channels=256, out_channels=512, kernel_size=(3,3),
                                            stride=1, padding=1, max_pool=True)
        self.resblock2    = self.res_block(in_channels=512, out_channels=512, kernel_size=(3,3))
        # Max Pool
        self.maxpool      = nn.MaxPool2d(kernel_size=(4,4))
        # FC Layer
        self.fc           = nn.Linear(512, 10, bias=False)

        # Define PyTorch model
        self.model = nn.Sequential(
            self.preplayer,
            self.conv1_layer1,
            self.resblock1,
            self.conv2_layer2,
            self.conv3_layer3,
            self.resblock2,
            self.maxpool,
            self.fc
        )

        self.accuracy = Accuracy(task="MULTICLASS", num_classes=10)

    def forward(self, x):
        x  = self.preplayer(x)
        x  = self.conv1_layer1(x)
        r1 = self.resblock1(x)
        x  = x + r1
        x  = self.conv2_layer2(x)
        x  = self.conv3_layer3(x)
        r2 = self.resblock2(x)
        x  = x + r2
        x  = self.maxpool(x)
        x  = x.view(-1, 512)
        x  = self.fc(x)

        return F.log_softmax(x, dim=-1)

    def conv_block(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                   dropout_value=0, groups=1, dilation=1, max_pool=False):
        x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                       kernel_size=kernel_size, padding=padding, dilation=dilation,
                       groups=groups, bias=False)
        x2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        x3 = nn.BatchNorm2d(out_channels)
        x4 = nn.ReLU()
        x5 = nn.Dropout(dropout_value)
        if max_pool == True:
            return nn.Sequential(x1, x2, x3, x4, x5)
        return nn.Sequential(x1, x3, x4, x5)

    def res_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
                self.conv_block(in_channels, out_channels, kernel_size, stride=1, padding=1),
                self.conv_block(in_channels, out_channels, kernel_size, stride=1, padding=1)
        )

    def dial_conv_block(self, in_channels, out_channels, kernel_size, padding, dropout_value, groups=1, dilation=1):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=groups, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels),
            nn.Dropout(dropout_value),
        )

    def output_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.AvgPool2d(kernel_size=26),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), padding=0, bias=False)
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.logger.experiment.add_scalar("Loss/Train",
                                           loss,
                                           self.current_epoch)

        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)
        self.logger.experiment.add_scalar("Accuracy/Train",
                                           self.accuracy.compute().item(),
                                           self.current_epoch)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)

        self.logger.experiment.add_scalar("Loss/Test",
                                           loss,
                                           self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Test",
                                           self.accuracy.compute().item(),
                                           self.current_epoch)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        self.train_data, self.test_data = trainset, testset

    def setup(self, stage=None):
        self.train_data, self.test_data = trainset, testset
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.cifar_train, self.cifar_train_val = trainset, valset

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar_test = self.test_data

    def train_dataloader(self):
        return train_loader

    def val_dataloader(self):
        return val_loader

    def test_dataloader(self):
        return test_loader
