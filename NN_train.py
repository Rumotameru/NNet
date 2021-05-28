import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import data_loader as dl
import network
import visualization


# data paths
train_path = r'C-NMC_Leukemia/training_data'
test_info_path = r'C-NMC_Leukemia/validation_data'
test_path = r'C-NMC_Leukemia/validation_data/C-NMC_test_prelim_phase_data'

# Hyper parameters
num_epochs = 10
batch_size = 10
learning_rate = 0.003


def model_proccesing():
    # load-data
    train_info = dl.train_info_load(train_path)
    test_info = dl.test_info_load(test_info_path, 'C-NMC_test_prelim_phase_data_labels.csv')

    ## vizualize
    # visualization.class_balance(test_info)
    # visualization.preview(train_info.sample(n=50), train_path)

    # datasets
    train_data, valid_data, test_data = dl.dataset_prepare(train_info, test_info, train_path, test_path)

    # CPU or GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # to-torch-tensor
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    model = network.CNN()
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    train_losses = []
    valid_losses = []
    valid_acc = []
    test_acc = []

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}:")

        # keep-track-of-training-and-validation-loss
        train_loss = 0.0
        valid_loss = 0.0

        # training-the-model
        model.train()
        model, train_loss, optimizer = network.train(model, train_loader, train_loss, optimizer, criterion, device)

        # validate-the-model
        model.eval()
        model, valid_loss, acc = network.valid(model, valid_loader, valid_loss, criterion, device)
        valid_acc.append(acc)

        # calculate-average-losses
        train_loss = train_loss / len(train_loader.sampler)
        valid_loss = valid_loss / len(valid_loader.sampler)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # print-training/validation-statistics
        print('\tTraining Loss: {:.6f} \tValidation Loss: {:.6f} '.format(
            train_loss, valid_loss))

        # test-the-model
        with torch.no_grad():
            acc = network.test(model, test_loader, device)
            test_acc.append(acc)

    ## Save
    # torch.save(model, "model.pt")
    torch.save(model.state_dict(), "model_state.pt")

    return train_losses, valid_losses, valid_acc, test_acc


if __name__ == "__main__":
    t_losses, v_losses, valid_accuracy, test_accuracy = model_proccesing()
    visualization.print_losses(t_losses, v_losses)
    visualization.print_acc(valid_accuracy, test_accuracy)


