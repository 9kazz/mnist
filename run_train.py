from src.prepare_data import *
from src.network import *
from src.train import *

import torch.optim

def main():
    train_dataset, _, _ = create_mnist_datasets("./data", val_set_size = 5000, mnist_set_size=45000)

    train_ldr = Loader(train_dataset, batch_size=64)

    model = Net()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with open("5^10.log", "w") as train_log:
        train(model, train_ldr, criterion, optimizer, num_epoch=50, log_file=train_log)

    save_model(model, "saved_models/5^10.pt")
    
    
if __name__ == "__main__":
    main()