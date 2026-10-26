from src.prepare_data import *
from src.network import *
from src.benchmark import *

def main():
    _, _, test_dataset = create_mnist_datasets("./data")
    loader = Loader(test_dataset, batch_size=64)

    show_mnist([test_dataset[i] for i in range(15)])

    model     = load_model("saved_models/16*16.pt")
    criterion = torch.nn.CrossEntropyLoss()

    run_benchmark(model, loader, criterion)

if __name__ == "__main__":
    main()