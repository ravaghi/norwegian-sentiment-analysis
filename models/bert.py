from transformers import AutoModelForSequenceClassification
from torchmetrics import Accuracy, F1Score, AUROC
from torch import optim
import lightning as L
import wandb
import torch


class BERT(L.LightningModule):
    def __init__(self, model_name: str, learning_rate: int, max_epochs: int, n_classes: int):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.n_classes = n_classes

        if n_classes == 2:
            id2label = {0: "negative", 1: "positive"}
            label2id = {"negative": 0, "positive": 1}
        else:
            id2label = {0: "negative", 1: "neutral", 2: "positive"}
            label2id = {"negative": 0, "neutral": 1, "positive": 2}

        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            id2label=id2label,
            label2id=label2id
        )

        self.test_step_losses = []
        self.test_step_y_hats = []
        self.test_step_y_hat_probs = []
        self.test_step_ys = []

    def _log_metrics(self, type, loss, y_hat, y):
        if self.n_classes == 2:
            _, y_hat = torch.max(y_hat, 1)

            accuracy = Accuracy("binary", num_classes=2)( y_hat.detach().cpu(), y.detach().cpu())
            f1_score = F1Score("binary", num_classes=2)(y_hat.detach().cpu(), y.detach().cpu())
            roc_auc = AUROC("binary", num_classes=2)(y_hat.detach().cpu(), y.detach().cpu())
        else:
            y_hat_prob = y_hat
            _, y_hat = torch.max(y_hat, 1)
            accuracy = Accuracy("multiclass", num_classes=3)( y_hat.detach().cpu(), y.detach().cpu())
            f1_score = F1Score("multiclass", num_classes=3)( y_hat.detach().cpu(), y.detach().cpu())
            roc_auc = AUROC( "multiclass", num_classes=3)( y_hat_prob.detach().cpu(), y.detach().cpu())

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

    def _common_step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        y = batch['labels']
        output = self.bert(input_ids, attention_mask=attention_mask, labels=y)
        loss = output.loss
        y_hat = output.logits
        _, y = torch.max(y, 1)
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

        if self.n_classes == 2:
            accuracy = Accuracy("binary", num_classes=2)(y_hat.detach().cpu(), y.detach().cpu())
            f1_score = F1Score("binary", num_classes=2)(y_hat.detach().cpu(), y.detach().cpu())
            roc_auc = AUROC("binary", num_classes=2)(y_hat.detach().cpu(), y.detach().cpu())
        else:
            accuracy = Accuracy("multiclass", num_classes=3)(y_hat.detach().cpu(), y.detach().cpu())
            f1_score = F1Score("multiclass", num_classes=3)(y_hat.detach().cpu(), y.detach().cpu())
            roc_auc = AUROC("multiclass", num_classes=3)(y_hat_prob.detach().cpu(), y.detach().cpu())

        wandb.run.summary["test_loss"] = avg_loss.item()
        wandb.run.summary["test_accuracy"] = accuracy
        wandb.run.summary["test_auc"] = roc_auc
        wandb.run.summary["test_f1"] = f1_score

        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
            preds=y_hat.detach().cpu().numpy(),
            y_true=y.detach().cpu().numpy(),
            class_names=["negative", "positive"] if self.n_classes == 2 else ["negative", "neutral", "positive"]
        )})

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return [optimizer], [scheduler]