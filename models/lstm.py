from torch import nn, optim
import lightning as L
import torchmetrics
import wandb
import torch


class LSTM(L.LightningModule):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, seq_len, n_class, learning_rate, max_epochs, class_weights):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.n_class = n_class

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )
        self.fc = nn.Linear(seq_len * hidden_dim, n_class)

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.test_step_losses = []
        self.test_step_y_hats = []
        self.test_step_y_hat_probs = []
        self.test_step_ys = []

    def _log_metrics(self, type, loss, y_hat, y):
        if self.n_class == 2:
            _, y_hat = torch.max(y_hat, 1)
            
            accuracy = torchmetrics.Accuracy("binary", num_classes=2)( y_hat.detach().cpu(), y.detach().cpu())
            f1_score = torchmetrics.F1Score("binary", num_classes=2)(y_hat.detach().cpu(), y.detach().cpu())
            roc_auc = torchmetrics.AUROC("binary", num_classes=2)(y_hat.detach().cpu(), y.detach().cpu())
        else:
            y_hat_prob = y_hat
            _, y_hat = torch.max(y_hat, 1)
            accuracy = torchmetrics.Accuracy("multiclass", num_classes=3)( y_hat.detach().cpu(), y.detach().cpu())
            f1_score = torchmetrics.F1Score("multiclass", num_classes=3)( y_hat.detach().cpu(), y.detach().cpu())
            roc_auc = torchmetrics.AUROC( "multiclass", num_classes=3)( y_hat_prob.detach().cpu(), y.detach().cpu())

        self.log_dict(
            {
                f"{type}_accuracy": accuracy,
                f"{type}_loss": loss,
                f"{type}_auc": roc_auc,
                f"{type}_f1": f1_score,
            },
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=True
        )

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _common_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch)
        self._log_metrics("train", loss, y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch)
        self._log_metrics("val", loss, y_hat, y)
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch)
        y_hat_prob = y_hat
        _, y_hat = torch.max(y_hat, 1)
        self.test_step_losses.append(loss)
        self.test_step_y_hats.append(y_hat)
        self.test_step_y_hat_probs.append(y_hat_prob)
        self.test_step_ys.append(y)
        return loss

    def on_test_epoch_end(self):
        avg_loss = torch.stack(self.test_step_losses).mean()
        y_hat = torch.cat(self.test_step_y_hats)
        y_hat_prob = torch.cat(self.test_step_y_hat_probs)
        y = torch.cat(self.test_step_ys)

        if self.n_class == 2:
            accuracy = torchmetrics.Accuracy("binary", num_classes=2)(y_hat.detach().cpu(), y.detach().cpu())
            f1_score = torchmetrics.F1Score("binary", num_classes=2)(y_hat.detach().cpu(), y.detach().cpu())
            roc_auc = torchmetrics.AUROC("binary", num_classes=2)(y_hat.detach().cpu(), y.detach().cpu())
        else:
            accuracy = torchmetrics.Accuracy("multiclass", num_classes=3)(y_hat.detach().cpu(), y.detach().cpu())
            f1_score = torchmetrics.F1Score("multiclass", num_classes=3)(y_hat.detach().cpu(), y.detach().cpu())
            roc_auc = torchmetrics.AUROC("multiclass", num_classes=3)(y_hat_prob.detach().cpu(), y.detach().cpu())

        wandb.run.summary["test_loss"] = avg_loss.item()
        wandb.run.summary["test_accuracy"] = accuracy
        wandb.run.summary["test_auc"] = roc_auc
        wandb.run.summary["test_f1"] = f1_score
        
        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
            preds=y_hat.detach().cpu().numpy(),
            y_true=y.detach().cpu().numpy(),
            class_names=["negative", "positive"] if self.n_class == 2 else ["negative", "neutral", "positive"]
        )})

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return [optimizer], [scheduler]
