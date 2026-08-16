'Internal utilities for the iSCALE research workflow.'
import os
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.optim.lr_scheduler import ReduceLROnPlateau


class ProteinRNATrainer:
    'Implementation of ProteinRNATrainer.'

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader=None,
        learning_rate=0.0001,
        weight_decay=1e-5,
        device=None,
        output_dir='./output',
        max_epochs=150,
        patience=20,
        checkpoint_interval=10,
        resume_from=None
    ):
        'Init.'

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device


        self.model = model.to(self.device)


        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader


        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )


        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.8,
            patience=15,
            verbose=True
        )


        self.output_dir = output_dir
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint_interval = checkpoint_interval


        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)


        self.start_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        self.best_val_pcc = -float('inf')
        self.best_epoch = 0
        self.patience_counter = 0


        self.best_model_path = os.path.join(output_dir, 'best_model.pt')


        if resume_from is not None:
            self.resume_training(resume_from)

    def resume_training(self, checkpoint_path):
        'Resume training.'
        print(f"Resuming from checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)


        self.model.load_state_dict(checkpoint['model_state_dict'])


        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        self.best_val_pcc = checkpoint.get('best_val_pcc', -float('inf'))
        self.best_epoch = checkpoint.get('best_epoch', 0)
        self.patience_counter = checkpoint.get('patience_counter', 0)

        print(f"Resumed at epoch {self.start_epoch}; best PCC={self.best_val_pcc:.4f}; patience={self.patience_counter}")

    def train_epoch(self):
        'Train epoch.'
        self.model.train()
        total_loss = 0
        aux_losses = {'contact': 0, 'intensity': 0}
        batch_count = 0


        ddg_loss_sum = 0
        aux_loss_sum = 0


        total_batches = len(self.train_loader)

        start_time = time.time()

        for i, (wild_data, mutant_data, rna_data, ddg) in enumerate(self.train_loader):
            try:

                wild_data = wild_data.to(self.device)
                mutant_data = mutant_data.to(self.device)
                rna_data = rna_data.to(self.device)


                if isinstance(ddg, (list, tuple)):
                    ddg = ddg[0]

                if isinstance(ddg, torch.Tensor):
                    target = ddg.to(self.device)
                    if target.dim() > 1:
                        target = target.squeeze()
                else:
                    target = torch.tensor([ddg], dtype=torch.float, device=self.device)


                self.optimizer.zero_grad()


                if hasattr(self.model, 'compute_loss') and callable(getattr(self.model, 'compute_loss')):

                    loss, loss_info = self.model.compute_loss(wild_data, mutant_data, rna_data, target)


                    if isinstance(loss_info, dict):
                        if 'ddg_loss' in loss_info:
                            ddg_loss_sum += loss_info['ddg_loss'].item()
                        if 'aux_loss' in loss_info:
                            aux_loss_sum += loss_info['aux_loss'].item()


                    if isinstance(loss_info, dict):
                        for key, value in loss_info.items():
                            if key != 'total_loss' and key != 'ddg_loss' and key in aux_losses:
                                aux_losses[key] += value.item()
                else:

                    output = self.model(wild_data, mutant_data, rna_data)


                    if isinstance(output, torch.Tensor) and output.dim() == 0:
                        output = output.unsqueeze(0)


                    if isinstance(output, tuple):
                        output = output[0]


                    loss = F.mse_loss(output, target)
                    ddg_loss_sum += loss.item()


                if torch.isnan(loss).any().item():
                    print("Warning: NaN loss detected; skipping this batch.")
                    continue


                loss.backward()


                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)


                self.optimizer.step()


                total_loss += loss.item()
                batch_count += 1


                if i == total_batches - 1 and hasattr(self.model, 'compute_loss'):
                    aux_weight = getattr(self.model, 'aux_weight', 0.5)
                    avg_ddg_loss = ddg_loss_sum / batch_count
                    avg_aux_loss = aux_loss_sum / batch_count

                    print("\nMean loss components for this epoch:")
                    print(f"  Primary DDG loss (MSE): {avg_ddg_loss:.6f}")
                    print(f"  Auxiliary loss: {avg_aux_loss:.6f}")
                    print(f"  Auxiliary weight: {aux_weight:.2f}")
                    print(f"  Weighted auxiliary loss: {(aux_weight * avg_aux_loss):.6f}")
                    print(f"  Total loss: {(avg_ddg_loss + aux_weight * avg_aux_loss):.6f}")
                    print(
                        f"  Auxiliary/primary ratio: {(aux_weight * avg_aux_loss / avg_ddg_loss if avg_ddg_loss > 0 else 0):.6f}")


                    for key, value in aux_losses.items():
                        if value > 0:
                            print(f"  {key} loss: {value / batch_count:.6f}")

            except Exception as e:
                print(f"Training batch failed: {e}")
                import traceback
                traceback.print_exc()
                continue


        avg_loss = total_loss / max(1, batch_count)
        epoch_time = time.time() - start_time


        aux_info = ""
        if batch_count > 0:
            aux_losses = {k: v / batch_count for k, v in aux_losses.items() if v > 0}
            if aux_losses:
                aux_info = ", " + ", ".join([f"{k} loss: {v:.4f}" for k, v in aux_losses.items()])

        print(f"Epoch complete: batches={batch_count}, mean loss={avg_loss:.4f}{aux_info}, time={epoch_time:.2f}s")

        return avg_loss

    def train_epoch_1(self):
        'Train epoch 1.'
        self.model.train()
        total_loss = 0
        aux_losses = {'contact': 0, 'intensity': 0}
        batch_count = 0

        start_time = time.time()

        for wild_data, mutant_data, rna_data, ddg in self.train_loader:
            try:

                wild_data = wild_data.to(self.device)
                mutant_data = mutant_data.to(self.device)
                rna_data = rna_data.to(self.device)


                if isinstance(ddg, (list, tuple)):
                    ddg = ddg[0]

                if isinstance(ddg, torch.Tensor):
                    target = ddg.to(self.device)
                    if target.dim() > 1:
                        target = target.squeeze()
                else:
                    target = torch.tensor([ddg], dtype=torch.float, device=self.device)


                self.optimizer.zero_grad()


                if hasattr(self.model, 'compute_loss') and callable(getattr(self.model, 'compute_loss')):

                    loss, loss_info = self.model.compute_loss(wild_data, mutant_data, rna_data, target)


                    if isinstance(loss_info, dict):
                        for key, value in loss_info.items():
                            if key != 'total_loss' and key != 'ddg_loss' and key in aux_losses:
                                aux_losses[key] += value.item()
                else:

                    output = self.model(wild_data, mutant_data, rna_data)


                    if isinstance(output, torch.Tensor) and output.dim() == 0:
                        output = output.unsqueeze(0)


                    if isinstance(output, tuple):
                        output = output[0]


                    loss = F.mse_loss(output, target)


                if torch.isnan(loss).any().item():
                    print("Warning: NaN loss detected; skipping this batch.")
                    continue


                loss.backward()


                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)


                self.optimizer.step()


                total_loss += loss.item()
                batch_count += 1

            except Exception as e:
                print(f"Training batch failed: {e}")
                import traceback
                traceback.print_exc()
                continue


        avg_loss = total_loss / max(1, batch_count)
        epoch_time = time.time() - start_time


        aux_info = ""
        if batch_count > 0:
            aux_losses = {k: v / batch_count for k, v in aux_losses.items() if v > 0}
            if aux_losses:
                aux_info = ", " + ", ".join([f"{k} loss: {v:.4f}" for k, v in aux_losses.items()])

        print(f"Epoch complete: batches={batch_count}, mean loss={avg_loss:.4f}{aux_info}, time={epoch_time:.2f}s")

        return avg_loss

    def save_checkpoint(self, epoch, is_best=False):
        'Save checkpoint.'
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics,
            'best_val_pcc': self.best_val_pcc,
            'best_epoch': self.best_epoch,
            'patience_counter': self.patience_counter,
            'model_info': {
                'name': self.model.__class__.__name__,
                'module': self.model.__class__.__module__
            }
        }


        if not is_best:
            checkpoint_path = os.path.join(self.output_dir, 'checkpoints', f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, checkpoint_path)


            checkpoints = sorted([
                f for f in os.listdir(os.path.join(self.output_dir, 'checkpoints'))
                if f.startswith('checkpoint_epoch_')
            ])

            if len(checkpoints) > 5:
                oldest_checkpoint = os.path.join(self.output_dir, 'checkpoints', checkpoints[0])
                if os.path.exists(oldest_checkpoint):
                    os.remove(oldest_checkpoint)


        if is_best:
            torch.save(checkpoint, self.best_model_path)


            model_info = {
                'name': self.model.__class__.__name__,
                'module': self.model.__class__.__module__,
                'best_epoch': epoch,
                'best_pcc': self.best_val_pcc,
                'val_mse': self.val_metrics[-1]['mse'],
                'val_mae': self.val_metrics[-1]['mae']
            }

            with open(os.path.join(self.output_dir, 'model_info.json'), 'w') as f:
                json.dump(model_info, f, indent=2)

            print(f"Saved best model (PCC={self.best_val_pcc:.4f}) to {self.best_model_path}")

    def plot_training_curves(self):
        'Plot training curves.'
        plots_dir = os.path.join(self.output_dir, 'plots')


        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, 'loss_curve.png'), dpi=300)
        plt.close()


        plt.figure(figsize=(10, 6))
        plt.plot(epochs, [m['mae'] for m in self.val_metrics], 'g-', label='MAE')
        plt.plot(epochs, [m['pcc'] for m in self.val_metrics], 'c-', label='PCC')
        plt.title('Validation Metrics')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, 'metrics_curve.png'), dpi=300)
        plt.close()

    def train(self):
        'Train.'
        print(f"Training model: {self.model.__class__.__name__}")
        print(f"Device: {self.device}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")

        start_time = time.time()

        for epoch in range(self.start_epoch, self.max_epochs):
            epoch_start_time = time.time()

            print(f"\nEpoch {epoch+1}/{self.max_epochs}")


            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)


            val_metrics = self.evaluate()
            self.val_metrics.append(val_metrics)
            self.val_losses.append(val_metrics['loss'])

            epoch_time = time.time() - epoch_start_time


            self.scheduler.step(val_metrics['mse'])


            print(f"Epoch {epoch+1}/{self.max_epochs} | "
                  f"train loss: {train_loss:.4f} | "
                  f"validation MSE: {val_metrics['mse']:.4f} | "
                  f"validation MAE: {val_metrics['mae']:.4f} | "
                  f"validation PCC: {val_metrics['pcc']:.4f} | "
                  f"patience: {self.patience_counter}/{self.patience} | "
                  f"time: {epoch_time:.2f}s")


            is_best = val_metrics['pcc'] > self.best_val_pcc

            if is_best:
                self.best_val_pcc = val_metrics['pcc']
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1


            if (epoch + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(epoch)


            self.plot_training_curves()


            if self.patience_counter >= self.patience:
                print(f"Early stopping after {self.patience} epochs without improvement.")
                break

        total_time = time.time() - start_time

        print("\nTraining complete.")
        print(f"Total training time: {total_time:.2f}s")
        print(f"Best validation PCC: {self.best_val_pcc:.4f} (epoch {self.best_epoch+1})")


        if self.best_epoch < self.max_epochs - 1:
            print(f"Loading the best model from epoch {self.best_epoch+1}.")
            best_checkpoint = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(best_checkpoint['model_state_dict'])

        return self.best_val_pcc

    def evaluate(self):
        'Evaluate.'
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        batch_count = 0

        with torch.no_grad():
            for wild_data, mutant_data, rna_data, ddg in self.val_loader:
                try:

                    wild_data = wild_data.to(self.device)
                    mutant_data = mutant_data.to(self.device)
                    rna_data = rna_data.to(self.device)

                    if isinstance(ddg, (list, tuple)):
                        ddg = ddg[0]


                    if isinstance(ddg, torch.Tensor):
                        target = ddg.to(self.device)
                        if target.dim() > 1:
                            target = target.squeeze()
                    else:
                        target = torch.tensor([ddg], dtype=torch.float, device=self.device)


                    output = self.model(wild_data, mutant_data, rna_data)


                    if isinstance(output, dict):

                        prediction = output['ddg']
                    elif isinstance(output, torch.Tensor):

                        prediction = output
                    elif isinstance(output, tuple):

                        prediction = output[0]


                    if prediction.dim() == 0:
                        prediction = prediction.unsqueeze(0)


                    loss = F.mse_loss(prediction, target)


                    total_loss += loss.item()
                    batch_count += 1
                    predictions.append(prediction.cpu())
                    targets.append(target.cpu())

                except Exception as e:
                    print(f"Validation batch failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue


        if batch_count == 0:
            print("Warning: no valid validation batches were produced.")
            return {'loss': float('inf'), 'mse': float('inf'), 'mae': float('inf'), 'pcc': float('nan')}


        avg_loss = total_loss / batch_count
        predictions = torch.cat(predictions)
        targets = torch.cat(targets)

        mse = F.mse_loss(predictions, targets).item()
        mae = F.l1_loss(predictions, targets).item()


        x = predictions - predictions.mean()
        y = targets - targets.mean()
        pcc = torch.sum(x * y) / (torch.sqrt(torch.sum(x ** 2) * torch.sum(y ** 2)) + 1e-8)

        metrics = {
            'loss': avg_loss,
            'mse': mse,
            'mae': mae,
            'pcc': pcc.item()
        }

        return metrics

    def test(self):
        'Test.'
        if self.test_loader is None:
            print("No test set was provided.")
            return None, None, None

        print("\nEvaluating on the test set...")

        self.model.eval()
        predictions = []
        targets = []

        with torch.no_grad():

            has_rna_separate = True

            for wild_data, mutant_data, rna_data, ddg in self.test_loader:
                try:

                    wild_data = wild_data.to(self.device)
                    mutant_data = mutant_data.to(self.device)
                    rna_data = rna_data.to(self.device)

                    if isinstance(ddg, (list, tuple)):
                        ddg = ddg[0]


                    if isinstance(ddg, torch.Tensor):
                        target = ddg.to(self.device)
                        if target.dim() > 1:
                            target = target.squeeze()
                    else:
                        target = torch.tensor([ddg], dtype=torch.float, device=self.device)


                    output = self.model(wild_data, mutant_data, rna_data)


                    if isinstance(output, dict):

                        prediction = output['ddg']
                    elif isinstance(output, torch.Tensor):

                        prediction = output
                    elif isinstance(output, tuple):

                        prediction = output[0]


                    if prediction.dim() == 0:
                        prediction = prediction.unsqueeze(0)


                    predictions.append(prediction.cpu())
                    targets.append(target.cpu())

                except Exception as e:
                    print(f"Test batch failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        if not predictions:
            print("Warning: no valid test predictions were produced.")
            return None, None, None


        predictions_tensor = torch.cat(predictions)
        targets_tensor = torch.cat(targets)


        predictions_np = predictions_tensor.numpy()
        targets_np = targets_tensor.numpy()


        mse = np.mean((predictions_np - targets_np) ** 2)
        mae = np.mean(np.abs(predictions_np - targets_np))
        pcc = np.corrcoef(predictions_np.flatten(), targets_np.flatten())[0, 1]

        test_metrics = {
            'mse': mse,
            'mae': mae,
            'pcc': pcc
        }

        print("Test results:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  PCC: {pcc:.4f}")


        self.plot_predictions(predictions_np, targets_np)


        test_results = {
            'metrics': {
                'mse': float(mse),
                'mae': float(mae),
                'pcc': float(pcc)
            },
            'predictions': predictions_np.tolist(),
            'targets': targets_np.tolist()
        }

        with open(os.path.join(self.output_dir, 'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)

        return test_metrics, predictions_np, targets_np


    def plot_predictions(self, predictions, targets):
        """
        Plot prediction scatter plot

        Parameters:
        - predictions: prediction value array
        - targets: true value array
        """
        plots_dir = os.path.join(self.output_dir, 'plots')


        plt.figure(figsize=(10, 8))
        plt.scatter(targets, predictions, alpha=0.7)


        min_val = min(min(targets), min(predictions))
        max_val = max(max(targets), max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)


        z = np.polyfit(targets.flatten(), predictions.flatten(), 1)
        p = np.poly1d(z)
        plt.plot(targets, p(targets), "r--", alpha=0.7)


        plt.xlabel('Actual DDG Values')
        plt.ylabel('Predicted DDG Values')
        plt.title(f'Test Set Predictions (PCC: {np.corrcoef(predictions.flatten(), targets.flatten())[0, 1]:.4f})')


        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        pcc = np.corrcoef(predictions.flatten(), targets.flatten())[0, 1]

        plt.text(0.05, 0.95,
                 f'MSE: {mse:.4f}\n'
                 f'MAE: {mae:.4f}\n'
                 f'PCC: {pcc:.4f}',
                 transform=plt.gca().transAxes,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))


        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, 'test_predictions.png'), dpi=300)
        plt.close()


        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.hist(targets, bins=20, alpha=0.7, label='Actual Values')
        plt.hist(predictions, bins=20, alpha=0.7, label='Predicted Values')
        plt.xlabel('DDG Values')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('Distribution Comparison: Predicted vs Actual')

        plt.subplot(1, 2, 2)
        plt.hist(predictions - targets, bins=20, alpha=0.7)
        plt.xlabel('Prediction Error (Predicted - Actual)')
        plt.ylabel('Frequency')
        plt.title('Prediction Error Distribution')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'test_distribution.png'), dpi=300)
        plt.close()


def train_model(model, train_loader, val_loader, test_loader=None, **kwargs):
    'Train model.'
    trainer = ProteinRNATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        **kwargs
    )


    trainer.train()


    if test_loader is not None:
        trainer.test()

    return trainer



class SemiSupervisedProteinRNATrainer(ProteinRNATrainer):
    'Implementation of SemiSupervisedProteinRNATrainer.'

    def __init__(self, unlabeled_loader=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unlabeled_loader = unlabeled_loader
        self.structure_loss_history = []

    def train_epoch_semi_supervised(self):
        'Train epoch semi supervised.'
        self.model.train()
        total_supervised_loss = 0
        total_structure_loss = 0
        supervised_batches = 0
        structure_batches = 0

        start_time = time.time()


        print("Running supervised training...")
        supervised_loss = self.train_epoch_supervised()


        if self.unlabeled_loader is not None and hasattr(self.model, 'compute_structure_only_loss'):
            print("Running unsupervised structure learning...")

            for wild_data_unlabeled in self.unlabeled_loader:
                try:
                    wild_data_unlabeled = wild_data_unlabeled.to(self.device)

                    self.optimizer.zero_grad()


                    structure_loss = self.model.compute_structure_only_loss(wild_data_unlabeled)


                    weighted_structure_loss = self.model.structure_loss_weight * structure_loss

                    if not torch.isnan(weighted_structure_loss):
                        weighted_structure_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()

                        total_structure_loss += weighted_structure_loss.item()
                        structure_batches += 1

                except Exception as e:
                    print(f"Unsupervised batch failed: {e}")
                    continue


        avg_structure_loss = total_structure_loss / max(1, structure_batches)
        self.structure_loss_history.append(avg_structure_loss)

        epoch_time = time.time() - start_time

        print("Semi-supervised epoch complete:")
        print(f"  Supervised loss: {supervised_loss:.4f}")
        print(f"  Structure loss: {avg_structure_loss:.4f}")
        print(f"  Time: {epoch_time:.2f}s")

        return supervised_loss

    def train_epoch_supervised(self):
        'Train epoch supervised.'
        return super().train_epoch()

    def train(self):
        'Train.'
        print(f"Training semi-supervised model: {self.model.__class__.__name__}")


        original_train_epoch = self.train_epoch
        self.train_epoch = self.train_epoch_semi_supervised


        result = super().train()


        self.train_epoch = original_train_epoch

        return result

    def plot_training_curves(self):
        'Plot training curves.'
        super().plot_training_curves()


        if self.structure_loss_history:
            plots_dir = os.path.join(self.output_dir, 'plots')

            plt.figure(figsize=(10, 6))
            epochs = range(1, len(self.structure_loss_history) + 1)
            plt.plot(epochs, self.structure_loss_history, 'g-', label='Structure Loss')
            plt.title('Unsupervised Structure Learning Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Structure Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, 'structure_loss_curve.png'), dpi=300)
            plt.close()
