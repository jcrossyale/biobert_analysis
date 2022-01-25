from dataset import MedDataset
from models import COReClassifier, BERTClassifier, AvgMeter
from utilities import get_loader, load_data_as_dfs, get_model_tokenizer



def main():
    # print version specs
    print("Huggingface version:", transformers.__version__)
    print("CUDA is available:", torch.cuda.is_available())
    print("CPU count:", os.cpu_count())

    # empty cuda cache
    torch.cuda.empty_cache()

    # set desired hyperparameter

    hps = ["model_name","batch_size","learning_rate","dropout", "epochs"]
    hp_combos = [
        # BERT
        ["bert-base-uncased", 4, 3e-5, 0.25, 1], # demonstration example

        # CORe (BioBERT)
        # ["COReClassifier", 4, 3e-5, 0.25, 5], # val= 0.672 !
        # ["COReClassifier", 8, 3e-5, 0.25, 5], # val=0.634 
        # ["COReClassifier", 16, 1e-5, 0.25, 7], # val = 0.612
        # ["COReClassifier", 16, 3e-5, 0.25, 5], # val = 0.56
        # ["COReClassifier", 16, 1e-5, 0.25, 7], # val = 0.60
    ]

    hp_combos_dict = [dict(zip(hps,combo)) for combo in hp_combos]

    for hp_combo in hp_combos_dict:
        print("*" * 30)
        print("Running trial w/ HPs: ", hp_combo)
        options = Options()
        options.epochs = hp_combo["epochs"]
        options.model_name = hp_combo["model_name"]
        options.batch_size = hp_combo["batch_size"]
        options.learning_rate = hp_combo["learning_rate"]
        options.dropout = hp_combo["dropout"]

        # Start training!
        run_trial(options=options, train_dataframe=df_train, valid_dataframe=df_val)

        pass


def one_epoch(model, criterion, loader, device, optimizer=None, lr_scheduler=None, mode="train", step="batch"):
    loss_meter = AvgMeter()
    acc_meter = AvgMeter()
    
    # Fancy progress bar
    tqdm_object = tqdm(loader, total=len(loader))
    
    # Training loop
    for batch in tqdm_object:
        
        # move batch to GPU
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # get logits, compute loss, etc.
        logits = model(batch)
        loss = criterion(logits, batch['labels'])

        # If training, do gradient descent step 
        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step == "batch":
                lr_scheduler.step()
        
        # update metrics
        count = batch['input_ids'].size(0)
        loss_meter.update(loss.item(), count)
        
        accuracy = get_accuracy(logits.detach(), batch['labels'])
        acc_meter.update(accuracy.item(), count)

        # Update progress bar labels
        if mode == "train":
            tqdm_object.set_postfix(loss=loss_meter.avg, accuracy=acc_meter.avg, lr=get_lr(optimizer))
        else:
            tqdm_object.set_postfix(loss=loss_meter.avg, accuracy=acc_meter.avg)
    
    return loss_meter, acc_meter


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def get_accuracy(logits, labels):
    """
    logits shape: (batch_size, num_labels)
    labels shape: (batch_size)
    """
    preds = logits.argmax(dim=1)
    acc = (preds == labels).float().mean()
    return acc


def train_eval(epochs, model, train_loader, valid_loader, 
               criterion, optimizer, device, options, lr_scheduler=None):
    
    best_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    
    for epoch in range(epochs):
        print("*" * 30)
        print(f"Epoch {epoch + 1}")
        current_lr = get_lr(optimizer)
        
        # Training
        model.train()
        train_loss, train_acc = one_epoch(model, 
                                          criterion, 
                                          train_loader, 
                                          device,
                                          optimizer=optimizer,
                                          lr_scheduler=lr_scheduler,
                                          mode="train",
                                          step=options.step)  
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            valid_loss, valid_acc = one_epoch(model, 
                                              criterion, 
                                              valid_loader, 
                                              device,
                                              optimizer=None,
                                              lr_scheduler=None,
                                              mode="valid")
        
        # if average validation loss less than best previous, save the weights
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            best_model_weights = copy.deepcopy(model.state_dict())
            save_path = os.path.join(options.models_dir,options.model_save_name)
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")
        
        # or you could do: if step == "epoch":
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(valid_loss.avg)
            # if the learning rate changes by ReduceLROnPlateau, we are going to
            # reload our previous best model weights and start from there with a lower LR
            if current_lr != get_lr(optimizer):
                print("Loading best model weights...")
                model.load_state_dict(torch.load(os.path.join(options.models_dir,options.model_save_name), 
                                                 map_location=device))
        
        # print the metrics
        print(f"Train Loss: {train_loss.avg:.5f}")
        print(f"Train Accuracy: {train_acc.avg:.5f}")
        
        print(f"Valid Loss: {valid_loss.avg:.5f}")
        print(f"Valid Accuracy: {valid_acc.avg:.5f}")
        print("*" * 30)

def run_trial(options, train_dataframe, valid_dataframe):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load the correct model and associated tokenizer
    if options.model_name == "bert-base-uncased":
      
        # get tokenizer from HF
        tokenizer = AutoTokenizer.from_pretrained(options.model_name, use_fast=True)

        # get model class from above
        model = BERTClassifier(num_labels=options.num_labels, dropout=options.dropout).to(device)

    elif options.model_name == 'COReClassifier':
        tokenizer = AutoTokenizer.from_pretrained("bvanaken/CORe-clinical-diagnosis-prediction")
        model = COReClassifier(num_labels=options.num_labels, dropout=options.dropout).to(device)

    else:
      print("Invalid model_name!")
      
    # define train and validation data, create data loaders
    train_loader = get_loader(train_dataframe, tokenizer, "train", options.max_length)
    valid_loader = get_loader(valid_dataframe, tokenizer, "valid", options.max_length)

    # define optimizer, loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # set parameters of the LR scheduler
    if options.scheduler == "ReduceLROnPlateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="min", 
            factor=0.5, 
            patience=2
        )
        
        # when to step the scheduler: after an epoch or after a batch
        options.step = "epoch"
    
    # define save name of model
    options.model_save_name = f"model_{options.model_name}_LR_{options.learning_rate}_dropout_{options.dropout}_bs_{options.batch_size}.pt"
    
    # train model on train/val sets
    train_eval(
        epochs = options.epochs,
        model = model,
        train_loader = train_loader,
        valid_loader = valid_loader,
        criterion = criterion,
        optimizer=optimizer,
        device=device,
        options=options,
        lr_scheduler=lr_scheduler
    )



if __name__ == 'main':
    main()

