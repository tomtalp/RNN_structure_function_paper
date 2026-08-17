import sys
import datetime

from core.truth_table import TruthTableDataset
from core.pytorch_network import SparseMaskedRNN

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

def run_training_loop(pytorch_net, TT, lr, epochs, device, lr_decay_fraction=None, lr_decay_epoch_frequency=None):

    TT_dataset = TruthTableDataset(TT)
    dataloader = DataLoader(TT_dataset, batch_size=1, shuffle=False)
    optimizer = optim.Adam(pytorch_net.get_pytorch_params(), lr=lr)
    if lr_decay_fraction and lr_decay_epoch_frequency:
        print(f"Learning Rate decay is active! decaying every {lr_decay_epoch_frequency} epochs by {lr_decay_fraction*100}%")
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_epoch_frequency, gamma=lr_decay_fraction)


    loss_func = torch.nn.BCEWithLogitsLoss()

    training_log = {}
    training_log["lr"] = lr
    training_log["max_epochs"] = epochs
    training_log["start_time"] = datetime.datetime.now()
    training_log["epoch_logs"] = []

    initial_acc = TT.get_similarity(pytorch_net.get_binarized_truth_table())
    training_log["initial_acc"] = initial_acc


    for epoch in range(epochs):
        running_loss = 0
        for (X, y) in dataloader:
            x = X.squeeze(0).double()
            x = x.to(device)

            y = y.squeeze(0).double()
            y = y.to(device)

            y_hat = pytorch_net(x).unsqueeze(0)

            if any(y_hat.isnan()):
                continue
            loss = loss_func(y_hat, y)

            running_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pytorch_net.apply_mask()

            # print(f"Epoch #{epoch+1} / {epochs}. Loss = {running_loss} , acc = {TT.get_similarity(pytorch_net.get_binarized_truth_table())}")

        if lr_decay_epoch_frequency is not None:
            scheduler.step()
            if epoch % lr_decay_epoch_frequency == 0:
                print(f"Epoch = {epoch}, lr = {optimizer.param_groups[0]['lr']}")

        acc = TT.get_similarity(pytorch_net.get_binarized_truth_table())
        if acc == 1:
            avg_loss_in_epoch = running_loss / len(dataloader)
            training_log["epoch_logs"].append(
                {"epoch": epoch, "loss": avg_loss_in_epoch, "acc": acc})
            break
        if epoch % 100 == 0:
            avg_loss_in_epoch = running_loss / len(dataloader)
            # print(f"Epoch #{epoch+1} / {epochs}. Loss = {avg_loss_in_epoch} , acc = {acc}")
            training_log["epoch_logs"].append({"epoch": epoch, "loss": avg_loss_in_epoch, "acc": acc})

    avg_loss_in_epoch = running_loss / len(dataloader)
    acc = TT.get_similarity(pytorch_net.get_binarized_truth_table())
    # print(f"\t\t\t## Done! Loss = {avg_loss_in_epoch} , acc = {acc}")

    training_log["end_time"] = datetime.datetime.now()
    training_log["final_acc"] = TT.get_similarity(
        pytorch_net.get_binarized_truth_table())
    training_log["network_params"] = list(pytorch_net.get_pytorch_params())

    return training_log

def train_model(n_attempts, input_size, hidden_size, output_size, num_steps, recurrent_mask, epochs, learning_rate, threshold, device, dataloader, TT_dataset):
        best_accuracy = 0
        best_model = None

        for attempt in range(n_attempts):
            model = SparseMaskedRNN(input_size, hidden_size, output_size, num_steps, recurrent_mask)
            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            loss_func = torch.nn.BCEWithLogitsLoss()
            # print(f"Attempt {attempt+1} / {n_attempts} - ")
            for epoch in range(epochs):
                running_loss = 0
                for (X, y) in dataloader:
                    X = X.to(device)
                    y = y.to(device)
                    optimizer.zero_grad()
                    output = model(X)
                    loss = loss_func(output, y)
                    loss.backward()
                    optimizer.step()

                    model.apply_masks()

                    running_loss += loss.item()

                model.eval()
                correct = 0
                with torch.no_grad():
                    for (X, y) in dataloader:
                        X = X.to(device)
                        y = y.to(device)

                        output = model(X)
                        output = torch.sigmoid(output)
                        output[output >= threshold] = 1
                        output[output < threshold] = 0

                        correct += torch.sum(output == y).item()

                    accuracy = correct / len(TT_dataset)

                if epoch % 100 == 0:
                    print(f"\tEpoch {epoch} - ")
                    print("\t\tLoss: ", running_loss)
                    print("\t\tAccuracy: ", accuracy)

                sys.stdout.flush()
                sys.stderr.flush()

                if accuracy == 1:
                    print(f"\t\tConverged at step {epoch}")
                    break

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                # print(f"New best accuracy: {best_accuracy}")

            if best_accuracy == 1:
                break

        return best_accuracy, best_model
