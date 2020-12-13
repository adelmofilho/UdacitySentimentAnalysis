import time
import pylab as pl
from IPython import display
from tqdm import tqdm
import pandas as pd

def train(model, train_loader, valid_loader, epochs, optimizer, loss_fn, device):
    report_train = pd.DataFrame(columns=["epoch", "trainError", "validError"])
    best_valid_BCELoss = 9999999999
    BCELoss_list = []
    valid_BCELoss_list = []
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        total_loss = 0
        total_valid_loss = 0        
        for batch in train_loader:         
            batch_X, batch_y = batch
            
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # TODO: Complete this train method to train the model provided.
            optimizer.zero_grad()
            output = model(batch_X)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()            
            total_loss += loss.data.item()
        for block in valid_loader:     
            block_X, block_y = block
            
            block_X = block_X.to(device)
            block_y = block_y.to(device)
            output_valid = model(block_X)
            valid_loss = loss_fn(output_valid, block_y)
            total_valid_loss += valid_loss.data.item()
        BCELoss = total_loss/len(train_loader)
        BCELoss_list.append(BCELoss)
        valid_BCELoss = total_valid_loss/len(valid_loader)
        valid_BCELoss_list.append(valid_BCELoss)
        if valid_BCELoss < best_valid_BCELoss: 
            #dummy_input = torch.tensor(block_X).to(device).long()
            #torch.onnx.export(model, dummy_input, f"models/best_model.onnx")
            best_valid_BCELoss = BCELoss
        
        desc = (f'Epoch: {epoch}, train_loss: {BCELoss}, valid_loss: {valid_BCELoss}')
        print(desc)
        to_append = [epoch, BCELoss, valid_BCELoss]
        report_train_length = len(report_train)
        report_train.loc[report_train_length] = to_append
        #print("Epoch: {}, BCELoss: {}".format(epoch, total_loss / len(train_loader)))
        axisy = list(report_train.trainError.values) + list(report_train.validError.values)
        display.clear_output(wait=True)
        if epoch == 1:
            pl.plot(BCELoss_list, '-b', label="TrainError")
            pl.plot(valid_BCELoss_list, '-r', label="ValidationError")
            pl.legend(loc='upper right')
        else:
            pl.plot(BCELoss_list, '-b')
            pl.plot(valid_BCELoss_list, '-r')
        pl.xlim(0, epochs-1)
        pl.ylim(min(axisy)*0.9, max(axisy)*1.1)
        display.display(pl.gcf())
        time.sleep(1.0)
        display.clear_output(wait=True)
    return report_train, model
