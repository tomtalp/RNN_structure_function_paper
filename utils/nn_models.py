import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def calculated_R_squared(y, y_hat):
    y = np.array(y)
    y_hat = np.array(y_hat)
    y_mean = np.mean(y)
    SS_tot = np.sum((y - y_mean)**2)
    SS_res = np.sum((y - y_hat)**2)
    return 1 - (SS_res / SS_tot)

def epsilon_mask(x, mask):
    x[x == 0] = mask
    return x

def pad_1d_tensor(x, desired_dim):
    how_much_padding = desired_dim - len(x)
    pad = torch.empty(how_much_padding)

    result = epsilon_mask(torch.concat((x, pad)), 1e-10)
    return result

class LinReg(nn.Module):
    def __init__(self, input_dim):
        super(LinReg, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)

class FF_1Layer(nn.Module):
    def __init__(self, input_dim):
        super(FF_1Layer, self).__init__()

        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class FF_2Layer(nn.Module):
    def __init__(self, input_dim):
        super(FF_2Layer, self).__init__()

        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.fc3 = nn.Linear(input_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        x = self.fc3(x)
        # x = nn.Sigmoid()(x)
        return x

class DeepFF(nn.Module):
    def __init__(self, input_dim):
        super(DeepFF, self).__init__()

        self.fc1 = nn.Linear(input_dim, 2*input_dim)
        self.fc2 = nn.Linear(2*input_dim, 2*input_dim)
        self.fc3 = nn.Linear(2*input_dim, input_dim)
        self.fc4 = nn.Linear(input_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        x = self.fc4(x)
        # x = nn.Sigmoid()(x)
        return x

class PairsDataset(Dataset):
    def __init__(self, X_list, y_list):
        self.X = torch.stack(X_list)  # Convert list of tensors to a single tensor
        self.y = torch.tensor(y_list, dtype=torch.float32).view(-1, 1)  # Convert to tensor and reshape for MSE loss

    def __len__(self):
        return len(self.X)  # Number of samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]  # Return one sample at index idx

def perform_network_success_fit(model_class, representation_dim, X, y, n_splits, test_size=0.2, num_epochs=100, batch_size=16, lr=0.001):
    best_split_data = {}

    best_r_2_score = -np.inf
    r_2_scores = []

    X_list = [torch.tensor(x, dtype=torch.float32) for x in X]
    y_list = [torch.tensor(y_val) for y_val in y]

    dataset = PairsDataset(X_list, y_list)

    train_size = int((1 - test_size) * len(dataset))
    test_size_n = len(dataset) - train_size

    for i in range(n_splits):
        print(f"Split #{i}")
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size_n])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, shuffle=False)

        print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

        model = model_class(input_dim=representation_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for batch_X, batch_y in train_loader:
                predictions = model(batch_X.view(-1, representation_dim))
                loss = criterion(predictions, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

        model.eval()
        test_loss = 0.0
        y_y_hat = []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                test_loss += loss.item()

                y_y_hat.append((float(batch_y[0][0]), float(predictions[0])))

        print(f'Test Loss: {test_loss/len(test_loader):.4f}')
        y_vals = [y for y, _ in y_y_hat]
        y_hat_vals = [y_hat for _, y_hat in y_y_hat]
        r_2 = calculated_R_squared(y_vals, y_hat_vals)
        print(f"Test R^2 score: {r_2}")

        r_2_scores.append(r_2)

        if r_2 > best_r_2_score:
            best_r_2_score = r_2
            best_split_data = {
                "model": model,
                "y_yhat": y_y_hat,
            }

    print(f"Performed {n_splits} splits, \n\t best R^2 score: {best_r_2_score} \n\t mean R^2 score: {np.mean(r_2_scores)} \n\t std R^2 score: {np.std(r_2_scores)}")
    return {
        "best_split_data": best_split_data,
        "r_2_scores": r_2_scores,
        "mean_r_2": np.mean(r_2_scores),
        "std_r_2": np.std(r_2_scores),
        "best_r_2": best_r_2_score,
    }

def perform_NetSolvesFunc_prediction(model_class, criterion, representation_dim, X_train, X_test, y_train, y_test, device, num_epochs=100, batch_size=16, lr=0.001):
    train_dataset = PairsDataset(
        [torch.tensor(x).type(torch.float32) for x in X_train],
        [torch.tensor(x).type(torch.float32) for x in y_train]
    )

    test_dataset = PairsDataset(
        [torch.tensor(x).type(torch.float32) for x in X_test],
        [torch.tensor(x).type(torch.float32) for x in y_test]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=len(test_dataset))

    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

    model = model_class(input_dim=representation_dim)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if (epoch+1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

    test_evaluation_metrics = {}
    if isinstance(criterion, nn.BCEWithLogitsLoss):
        model.eval()
        y_y_hat = []
        test_loss = 0.0
        threshold = 0.5

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                raw_predictions = model(batch_X)
                probabilities = torch.sigmoid(raw_predictions)
                predictions = (probabilities > threshold).int()

                y_true = batch_y.cpu().numpy()
                y_pred = (torch.sigmoid(raw_predictions) > threshold).int().cpu().numpy()
                y_probs = torch.sigmoid(raw_predictions).detach().cpu().numpy()

                # Compute metrics
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                roc_auc = roc_auc_score(y_true, y_probs)

                print(f"Accuracy: {accuracy:.2f}")
                print(f"Precision: {precision:.2f}")
                print(f"Recall: {recall:.2f}")
                print(f"F1-Score: {f1:.2f}")
                print(f"ROC-AUC: {roc_auc:.2f}")

                test_evaluation_metrics = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "roc_auc": roc_auc

                }



        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FN = np.sum((y_true == 1) & (y_pred == 0))

        confusion_matrix = pd.DataFrame(
            [[TP, FP], [FN, TN]],
            columns=["Predicted `Solved`", "Predicted `NotSolved`"],
            index=["True `Solved`", "True `NotSolved`"]
        )

        print("Confusion Matrix (TP, FP, TN, FN):")
        print(confusion_matrix)
        test_evaluation_metrics["confusion_matrix"] = confusion_matrix

    elif isinstance(criterion, nn.MSELoss):
        model.eval()
        y_y_hat = []
        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                test_loss += loss.item()

                y_y_hat.append((float(batch_y[0][0]), float(predictions[0])))


        print(f'Test Loss: {test_loss/len(test_loader):.4f}')
        y_vals = [y for y, _ in y_y_hat]
        y_hat_vals = [y_hat for _, y_hat in y_y_hat]
        r_2 = calculated_R_squared(y_vals, y_hat_vals)
        print(f"Test R^2 score: {r_2}")

        test_evaluation_metrics["r_2"] = r_2

    else:
        raise ValueError("Only BCEWithLogitsLoss and MSELoss are supported")


    return test_evaluation_metrics
