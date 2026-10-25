from prepare_data import *
from network import *

import torch.nn
import torch.optim

def main():
    train_dataset, val_dataset, test_dataset = create_mnist_datasets("./data", val_set_size = 1000, mnist_set_size=10000)
    train_loader,  val_loader,  test_loader  = create_dataloaders(train_dataset, val_dataset, test_dataset)
    # show_mnist([train_dataset[i] for i in range(10)])

    model = Net()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train(model, train_loader, criterion, optimizer, num_epoch=10)
    
if __name__ == "__main__":
    main()