# code based on code from https://www.learnpytorch.io/05_pytorch_going_modular/
"""
Contains functions for training and testing a PyTorch model.
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from jj_nn_framework.utils import plot_mnist_preds
from jj_nn_framework.nn_funcs import get_class_and_cnts

def batch_update(X:torch.Tensor, y:torch.Tensor,
  model: torch.nn.Module,
  loss_fn: torch.nn.Module,
  optimizer: torch.optim.Optimizer

):
  ''''''
  if optimizer != None:
    # 1. Forward pass
    pred = model(X)
    
    # 2. Calculate
    loss = loss_fn(pred, y)

    # 3. Optimizer zero grad
    optimizer.zero_grad(set_to_none=True)

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()
  else:
    # 1. Forward pass
    pred = model(X)

    # 2. Calculate
    loss = loss_fn(pred, y)

    # 3. Loss backward
    loss.backward()

  return loss,pred


def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module,
               metrics: List[any], 
               optimizer: torch.optim.Optimizer,
               scheduler: torch.optim.lr_scheduler,
               writer: any, #torch.utils.tensorboard.writer.SummaryWriter,
               device: torch.device,
               epoch: int) -> Tuple[float, float]:
      """Trains a PyTorch model for a single epoch.

      Turns a target PyTorch model to training mode and then
      runs through all of the required training steps (forward
      pass, loss calculation, optimizer step).

      Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        scheduler: A PyTorch learning scheduler to help minimize the loss function
        device: A target device to compute on (e.g. "cuda" or "cpu").

      Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy). For example:

        (0.1112, 0.8743)
      """
      # Put model in train mode
      model.train()

      # Setup train loss and train accuracy values
      train_loss, train_acc = 0, 0

      # Loop through data loader data batches
      for batch, (X, y) in tqdm(enumerate(dataloader)):
          
          # 1-5 Batch Update
          loss, y_pred = batch_update(X=X,y=y,model=model,loss_fn=loss_fn,optimizer=optimizer)

          # 6. Accumulate loss
          train_loss += loss.item()

          # Calculate Metrics
          for metric_fn in metrics:
            metric_val = metric_fn(y_pred,y).detach().cpu().numpy()
            #metrics_meters = [metric_fn.__name__].add(metric_val)
            
          #metrics_logs = {k: v.mean for k,v in metrics_meters.items()}

          train_acc += 1

          '''
          # calculate current batch
          curr_batch = epoch * len(dataloader) + batch
          #print(curr_batch)
          if curr_batch % 1000 == 999: # every 1000 batches
            # ...log running loss to tensorboard
            writer.add_scalar('Loss/training loss', train_loss / 1000, curr_batch)
            #writer.add_scalar('Loss/training loss', train_loss, curr_batch)
            writer.add_scalar('Accuracy/training accuracy', train_acc /1000, curr_batch)
            #writer.add_scalar('Accuracy/training accuracy', train_acc, curr_batch)

          if curr_batch == 0 or curr_batch == 99 or curr_batch == 999 or curr_batch == 4999 or curr_batch == 9999\
            or curr_batch == 19999 or curr_batch == 39999 or curr_batch == 59999: #< 100 and (curr_batch < 10 or (curr_batch >=10 and curr_batch % 10 == 9)): #or curr_batch % 1000 == 999:
            print(f"BATCH {curr_batch}!!:\nSaving sample images to TensorBoard...\n")
            # ...log Matplotlib Figure
            writer.add_figure('predictions vs actuals',
              plot_mnist_preds(y_pred,X,y,y_pred_class),
              global_step=curr_batch
            )
          '''

      # 7 Learning Rate Step
      #scheduler.step()

      # Adjust metrics to get average loss and accuracy per batch 
      train_loss = train_loss / len(dataloader)
      train_acc = train_acc / len(dataloader)
      return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              writer: any, #torch.utils.tensorboard.writer.SummaryWriter,
              device: torch.device,
              epoch: int = -1) -> Tuple[float, float]:
      """Tests a PyTorch model for a single epoch.

      Turns a target PyTorch model to "eval" mode and then performs
      a forward pass on a testing dataset.

      Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute on (e.g. "cuda" or "cpu").

      Returns:
        A tuple of testing loss and testing accuracy metrics.
        In the form (test_loss, test_accuracy). For example:

        (0.0223, 0.8985)
      """
      # Put model in eval mode
      model.eval() 

      # Setup test loss and test accuracy values
      test_loss, test_acc = 0, 0

      # Turn on inference context manager
      with torch.inference_mode():
          # Loop through DataLoader batches
          for batch, (X, y) in tqdm(enumerate(dataloader)):
              # Send data to target device
              #X, y = X.to(device), y.to(device)
              
              #X,y = val_transform((X,y))

              y_class, y_cnts = get_class_and_cnts(y)
              y_one_hot = F.one_hot(y_class,num_classes=2).to(torch.float32)
              
              #for i in range (12*16):
              
              # 1. Forward pass
              test_pred_logits = model(X)

              # 2. Calculate and accumulate loss
              loss = loss_fn(test_pred_logits, y_one_hot)
              test_loss += loss.item()

              # Calculate and accumulate accuracy
              test_pred_labels = test_pred_logits.argmax(dim=1)

              y_comp = y_one_hot.argmax(dim=-1)

              test_acc += ((test_pred_labels == y_comp).sum().item()/len(test_pred_labels))

              # Collect missed labels

              if epoch != -1:
                # calculate current batch
                curr_batch = epoch * len(dataloader) + batch
                if curr_batch % 100 == 99: # every 1000 batches
                  # ...log running loss to tensorboard
                  writer.add_scalar('Loss/testing loss', test_loss / 100, curr_batch)
                  #writer.add_scalar('Loss/testing loss', test_loss, curr_batch)
                  writer.add_scalar('Accuracy/testing accuracy', test_acc / 100, curr_batch)
                  #writer.add_scalar('Accuracy/testing accuracy', test_acc, curr_batch)

                  # ...log Matplotlib Figure
              else:
                pass

      # Adjust metrics to get average loss and accuracy per batch 
      test_loss = test_loss / len(dataloader)
      test_acc = test_acc / len(dataloader)
      return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          val_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler,
          writer: any, #torch.utils.tensorboard.writer.SummaryWriter,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          early_stop: int,
          chkpt_dir: str,
          sess_name: str) -> Dict[str, List]:
      """Trains and tests a PyTorch model.

      Passes a target PyTorch models through train_step() and test_step()
      functions for a number of epochs, training and testing the model
      in the same epoch loop.

      Calculates, prints and stores evaluation metrics throughout.

      Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        val_dataloader: A DataLoader instance for the model to be tested on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        scheduler: A PyTorch learning scheduler to help minimize the loss function
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda" or "cpu").
        early_stop: An integer indicating the number of epochs to train without validation improvement
        chkpt_dir: A path to where model checpoints should be saved
        sess_name: The session name used to save model chekpoints 

      Returns:
        A dictionary of training and validation loss as well as training and
        validation accuracy metrics. Each metric has a value in a list for 
        each epoch.
        In the form: {train_loss: [...],
                      train_acc: [...],
                      val_loss: [...],
                      val_acc: [...]} 
        For example if training for epochs=2: st
                     {train_loss: [2.0616, 1.0537],
                      train_acc: [0.3945, 0.3945],
                      val_loss: [1.2641, 1.5706],
                      val_acc: [0.3400, 0.2973]} 
      """
      # Initialize results dictionary
      results = {"train_loss": [],
          "train_acc": [],
          "val_loss": [],
          "val_acc": [],
          "optim_train_loss" : 1,
          "best_train_acc": 1,
          "optim_val_loss": 1,
          "best_val_acc": 0
      }

      # early stop count
      estc = 0
      file_name = f"{sess_name}_checkpoint.pt"
      best_val_loss = f"{sess_name}_vl_chkpt.pt"
      best_val_acc = f"{sess_name}_va_chkpt.pt"
      best_vl_va = f"{sess_name}_vl_va_chkpt.pt"
      end_check = f"{sess_name}_end_chkpt.pt"
      chkpt_file = Path(chkpt_dir) / file_name
      vl_chkpt_file = Path(chkpt_dir) / best_val_loss
      va_chkpt_file = Path(chkpt_dir) / best_val_acc
      end_chkpt_file = Path(chkpt_dir) / end_check
      vl_va_chkpt_file = Path(chkpt_dir) / best_vl_va

      # Loop through training and testing steps for a number of epochs
      for epoch in tqdm(range(epochs)):
          train_loss, train_acc = train_step(model=model,
                                              dataloader=train_dataloader,
                                              loss_fn=loss_fn,
                                              optimizer=optimizer,
                                              scheduler=scheduler,
                                              writer=writer,
                                              device=device,
                                              epoch=epoch)
          val_loss, val_acc = test_step(model=model,
              dataloader=val_dataloader,
              loss_fn=loss_fn,
              writer=writer,
              device=device,
              epoch=epoch)

          # Print out what's happening
          print(
              f"Epoch: {epoch+1} | "
              f"train_loss: {train_loss:.4f} | "
              f"train_acc: {train_acc:.4f} | "
              f"val_loss: {val_loss:.4f} | "
              f"val_acc: {val_acc:.4f}"
          )

          # Compare Test/Validation loss to prior loss to determine if overfitting is likely
          if val_loss >= results["optim_val_loss"]:
            # Update early stopping counter
            estc += 1
          else:
            pass
            

          # Compare current loss to prior loss and save checkpoint if improved
          if epoch == 0:
            # Save initial checkpoint
            torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'train_loss': results["optim_train_loss"],
              'train_acc': results["best_train_acc"],
              'val_loss': results["optim_val_loss"],
              'val_acc':results["best_val_acc"]
            }, chkpt_file)

            print(f"Saving initial checkpoint to:\n{chkpt_file}\nval_loss improved from {None} to {val_loss}\n"
              f"val_acc: {val_acc}\n"
            )

            # Update results
            results["optim_train_loss"] = train_loss
            results["best_train_acc"] = train_acc
            results["optim_val_loss"] = val_loss
            results["best_val_acc"] = val_acc

            

          else:
            #if train_loss < results["optim_train_loss"] and val_loss == results["optim_val_loss"]:
            if val_loss < results["optim_val_loss"] and val_acc > results["best_val_acc"]:
              # Update checkpoint
              torch.save({
              'epoch': epoch,
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'train_loss': results["optim_train_loss"],
              'train_acc': results["best_train_acc"],
              'val_loss': results["optim_val_loss"],
              'val_acc':results["best_val_acc"]
            }, vl_va_chkpt_file)

              print(f"Updating checkpoint @:\n{vl_chkpt_file}\nval_loss improved from {results['optim_val_loss']} to {val_loss}\n\nand\n\n"
                f"val_acc improved from {results['best_val_acc']} to {val_acc}\n"
              )

              # Update reults dictionary
              results["best_val_acc"] = val_acc
              results['optim_val_loss'] = val_loss
              estc = 0

            elif val_loss < results["optim_val_loss"]:
              # Update checkpoint
              torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'train_loss': results["optim_train_loss"],
              'train_acc': results["best_train_acc"],
              'val_loss': results["optim_val_loss"],
              'val_acc':results["best_val_acc"]
            }, vl_chkpt_file)

              print(f"Updating checkpoint @:\n{vl_chkpt_file}\nval_loss improved from {results['optim_val_loss']} to {val_loss}\n"
                f"val_acc: {val_acc}\n"
              )


              # Update reults dictionary
              results["optim_val_loss"] = val_loss
              estc = 0

            elif val_acc > results["best_val_acc"]:
              # Update checkpoint
              torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'train_loss': results["optim_train_loss"],
              'train_acc': results["best_train_acc"],
              'val_loss': results["optim_val_loss"],
              'val_acc':results["best_val_acc"]
            }, va_chkpt_file)

              print(f"Updating checkpoint @:\n{va_chkpt_file}\nval_acc improved from {results['best_val_acc']} to {val_acc}\n"
                f"val_loss: {val_loss}\n"
              )

              # Update reults dictionary
              results["best_val_acc"] = val_acc
              estc = 0
            else:
              pass

            if train_loss < results["optim_train_loss"]:
              # Update reults dictionary
              results["optim_train_loss"] = train_loss
            else:
              pass

            if train_acc > results["best_train_acc"]:
              # Update reults dictionary
              results["best_train_acc"] = train_acc
            else:
              pass

          # Update results dictionary
          results["train_loss"].append(train_loss)
          results["train_acc"].append(train_acc)
          results["val_loss"].append(val_loss)
          results["val_acc"].append(val_acc)

          # Early stopping
          if estc >= early_stop:

            # Update checkpoint
            torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'train_loss': results["optim_train_loss"],
              'train_acc': results["best_train_acc"],
              'val_loss': results["optim_val_loss"],
              'val_acc':results["best_val_acc"]
            }, end_chkpt_file)
            return results

      # Update checkpoint
      torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': results["optim_train_loss"],
        'train_acc': results["best_train_acc"],
        'val_loss': results["optim_val_loss"],
        'val_acc':results["best_val_acc"]
      }, end_chkpt_file)

      # Return the filled results at the end of the epochs
      return results
