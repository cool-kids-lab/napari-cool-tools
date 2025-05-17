# code based on code from https://www.learnpytorch.io/05_pytorch_going_modular/
"""
Contains functions for training and testing a PyTorch model.
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from torch import nn
from torchvision.ops import sigmoid_focal_loss
from jj_nn_framework.utils import plot_mnist_preds, reactivate_checkpoint

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               scheduler: torch.optim.lr_scheduler,
               writer: torch.utils.tensorboard.writer.SummaryWriter,
               transform: torch.nn.Module,
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
          # Send data to target device
          X, y = X.to(device), y.to(device)

          X,y = transform((X,y))


          torch.manual_seed(0)
          # 1. Forward pass
          y_pred = model(X)
          #print(f"y_pred:\n{y_pred}\n")

          #m = nn.Sigmoid()

          #y_pred = torch.where(y_pred>.9,1,0).to(torch.int64)
          #y_pred = y_pred.permute(0,3,1,2)
          #y = y.permute(0,3,1,2)     ########################### Maybe Remember to reinstate after testing focal loss!!

          #m(y_pred)

          # 2. Calculate  and accumulate loss    ########################### Remember to reinstate after testing focal loss!!
          #loss = loss_fn(y_pred, y)

          # test focal loss
          loss = sigmoid_focal_loss(y_pred,y,reduction='mean')

          #print(f"loss: {loss}\n")
          train_loss += loss.item() 

          acc = 1 - loss
          # 3. Optimizer zero grad
          #optimizer.zero_grad()
          optimizer.zero_grad(set_to_none=True)
          #for param in model.parameters(): # optimization for optimizer.zero_grad()
            #param.grad = None

          # 4. Loss backward
          loss.backward()

          # 5. Optimizer step  
          optimizer.step()

          # Calculate and accumulate accuracy metric across all batches
          #y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)

          #y_comp = y.argmax(dim=-1)

          #print(f"y_pred:\n{y_pred}")
            
          #print(f"y_comp:\n{y_comp}\n")

          #print(f"y_pred_class:\n{y_pred_class}\n")
          #train_acc += (y_pred_class == y_comp).sum().item()/len(y_pred)
          train_acc += acc.item()

          # calculate current batch

          
          curr_batch = epoch * len(dataloader) + batch
          #print(curr_batch)
          if True: #curr_batch % 10 == 9: # every 1000 batches
            # ...log running loss to tensorboard
            writer.add_scalar('Loss/training loss', train_loss / 1, curr_batch)
            #writer.add_scalar('Loss/training loss', train_loss, curr_batch)
            writer.add_scalar('Accuracy/training accuracy', train_acc /1, curr_batch)
            #writer.add_scalar('Accuracy/training accuracy', train_acc, curr_batch)

          '''
          if curr_batch == 0 or curr_batch == 99 or curr_batch == 999 or curr_batch == 4999 or curr_batch == 9999\
            or curr_batch == 19999 or curr_batch == 39999 or curr_batch == 59999: #< 100 and (curr_batch < 10 or (curr_batch >=10 and curr_batch % 10 == 9)): #or curr_batch % 1000 == 999:
            print(f"BATCH {curr_batch}!!:\nSaving sample images to TensorBoard...\n")
            # ...log Matplotlib Figure
            writer.add_figure('predictions vs actuals',
              plot_mnist_preds(y_pred,X,y,y_pred_class),
              global_step=curr_batch
            )

          '''
          

      # 6 Learning Rate Step   ############################# Testing without it may want to reinstate or replace with different scheduler
      #scheduler.step()

      # Adjust metrics to get average loss and accuracy per batch 
      train_loss = train_loss / len(dataloader)
      train_acc = train_acc / len(dataloader)
      return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              writer: torch.utils.tensorboard.writer.SummaryWriter,
              transform:torch.nn.Module,
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
              X, y = X.to(device), y.to(device)

              X,y = transform((X,y))


              torch.manual_seed(0)
              # 1. Forward pass
              test_pred_logits = model(X)

              #test_pred_logits = test_pred_logits.permute(0,3,1,2)
              #y = y.permute(0,3,1,2)         ########################### Maybe Remember to reinstate after testing focal loss!!

              # 2. Calculate and accumulate loss
              #loss = loss_fn(test_pred_logits, y)    ########################### Remember to reinstate after testing focal loss!!


              loss = sigmoid_focal_loss(test_pred_logits,y,reduction='mean')


              test_loss += loss.item()

              # Calculate and accumulate accuracy
              #test_pred_labels = test_pred_logits.argmax(dim=1)
            
              acc = 1 - loss

              #y_comp = y.argmax(dim=-1)
            
              #test_acc += ((test_pred_labels == y_comp).sum().item()/len(test_pred_labels))

              test_acc += acc.item() 
              # Collect missed labels

              if epoch != -1:
                # calculate current batch
                curr_batch = epoch * len(dataloader) + batch
                if True: #curr_batch % 10 == 9: # every 10 batches
                  # ...log running loss to tensorboard
                  writer.add_scalar('Loss/testing loss', test_loss / 1, curr_batch)
                  #writer.add_scalar('Loss/testing loss', test_loss, curr_batch)
                  writer.add_scalar('Accuracy/testing accuracy', test_acc / 1, curr_batch)
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
          writer: torch.utils.tensorboard.writer.SummaryWriter,
          transforms: Dict[torch.nn.Module,torch.nn.Module],
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          early_stop: int,
          chkpt_dir: str,
          sess_name: str,
          load_chkpt: str) -> Dict[str, List]:
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

      print(f"init values:\nsess_name: {sess_name}\nchkpt_dir{chkpt_dir}\n")


      if load_chkpt == '':
        # Initialize results dictionary ##### consider merits of initilizing this to 1 or .5
        results = {
          "train_loss": [.5],
          "train_acc": [.5],
          "val_loss": [.5],
          "val_acc": [.5],
          "optim_train_loss" : 1,
          "best_train_acc": 1,
          "optim_val_loss": 1,
          "best_val_acc": 0
        }

        # early stop count epoch init
        estc = 0
        epoch_start = 0
        epoch_stop = epochs
      else:
        # Initialize with checkpooint data
        data = reactivate_checkpoint(load_chkpt,model,optimizer)
        model = data['model']
        optimizer = data['optimizer']
        estc = data['early_stop']
        epoch_start = data['epoch']
        epoch_stop = epochs

        results = {
          'train_loss':[data['train_loss']],
          'train_acc':[data['train_acc']],
          'val_loss':[data['val_loss']],
          'val_acc':[data['val_acc']],
          'optim_train_loss':data['optim_train_loss'],
          'best_train_acc':data['best_train_acc'],
          'optim_val_loss':data['optim_val_loss'],
          'best_val_acc':data['best_train_acc']
        }


        sess_name = data['sess_name']
        chkpt_dir = data['chkpt_path'].as_posix()

        print(
          f"loaded values:\nsess_name: {sess_name}\nchkpt_dir{chkpt_dir}\n"
          f"epoch_start:{epoch_start}, early_stop_cnt:{estc}\n"
          f"results:\n{results}\n"
        )

      # file name init
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

      # tranforms
      train_transform = transforms['train']
      val_transform = transforms['val']

      # Loop through training and testing steps for a number of epochs
      for epoch in tqdm(range(epoch_start,epoch_stop)):
          train_loss, train_acc = train_step(model=model,
                                              dataloader=train_dataloader,
                                              loss_fn=loss_fn,
                                              optimizer=optimizer,
                                              scheduler=scheduler,
                                              writer=writer,
                                              transform=train_transform,
                                              device=device,
                                              epoch=epoch)
          val_loss, val_acc = test_step(model=model,
              dataloader=val_dataloader,
              loss_fn=loss_fn,
              writer=writer,
              transform=val_transform,
              device=device,
              epoch=epoch)

          #print(f"What are train and val losses?\n\n{type(train_loss)}\n\n{type(val_loss)}\n\n")

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
              'train_loss': sum(results['train_loss'])/len(results['train_loss']),
              'train_acc': sum(results['train_acc'])/len(results['train_acc']),
              'val_loss': sum(results['val_loss'])/len(results['val_loss']),
              'val_acc': sum(results['val_acc'])/len(results['val_acc']),
              'optim_val_loss': results['optim_val_loss'],
              'optim_train_loss': results['optim_train_loss'],
              'best_val_acc': results["best_val_acc"],
              'best_train_acc': results["best_train_acc"],
              'early_stop': estc,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict()
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
                'train_loss': sum(results['train_loss'])/len(results['train_loss']),
                'train_acc': sum(results['train_acc'])/len(results['train_acc']),
                'val_loss': sum(results['val_loss'])/len(results['val_loss']),
                'val_acc': sum(results['val_acc'])/len(results['val_acc']),
                'optim_val_loss': results['optim_val_loss'],
                'optim_train_loss': results['optim_train_loss'],
                'best_val_acc': results["best_val_acc"],
                'best_train_acc': results["best_train_acc"],
                'early_stop': estc,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
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
                'train_loss': sum(results['train_loss'])/len(results['train_loss']),
                'train_acc': sum(results['train_acc'])/len(results['train_acc']),
                'val_loss': sum(results['val_loss'])/len(results['val_loss']),
                'val_acc': sum(results['val_acc'])/len(results['val_acc']),
                'optim_val_loss': results['optim_val_loss'],
                'optim_train_loss': results['optim_train_loss'],
                'best_val_acc': results["best_val_acc"],
                'best_train_acc': results["best_train_acc"],
                'early_stop': estc,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
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
                'train_loss': sum(results['train_loss'])/len(results['train_loss']),
                'train_acc': sum(results['train_acc'])/len(results['train_acc']),
                'val_loss': sum(results['val_loss'])/len(results['val_loss']),
                'val_acc': sum(results['val_acc'])/len(results['val_acc']),
                'optim_val_loss': results['optim_val_loss'],
                'optim_train_loss': results['optim_train_loss'],
                'best_val_acc': results["best_val_acc"],
                'best_train_acc': results["best_train_acc"],
                'early_stop': estc,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
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
              'train_loss': sum(results['train_loss'])/len(results['train_loss']),
              'train_acc': sum(results['train_acc'])/len(results['train_acc']),
              'val_loss': sum(results['val_loss'])/len(results['val_loss']),
              'val_acc': sum(results['val_acc'])/len(results['val_acc']),
              'optim_val_loss': results['optim_val_loss'],
              'optim_train_loss': results['optim_train_loss'],
              'best_val_acc': results["best_val_acc"],
              'best_train_acc': results["best_train_acc"],
              'early_stop': estc,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict()
            }, end_chkpt_file)

            print(f"Updating checkpoint @:\n{end_chkpt_file}\n")

            return results

      # Update checkpoint
      torch.save({
        'epoch': epoch,
        'train_loss': sum(results['train_loss'])/len(results['train_loss']),
        'train_acc': sum(results['train_acc'])/len(results['train_acc']),
        'val_loss': sum(results['val_loss'])/len(results['val_loss']),
        'val_acc': sum(results['val_acc'])/len(results['val_acc']),
        'optim_val_loss': results['optim_val_loss'],
        'optim_train_loss': results['optim_train_loss'],
        'best_val_acc': results["best_val_acc"],
        'best_train_acc': results["best_train_acc"],
        'early_stop': estc,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
      }, end_chkpt_file)

      print(f"Updating checkpoint @:\n{end_chkpt_file}\n")

      # Return the filled results at the end of the epochs
      return results
