class Options:
    epochs = 5
    models_dir = "/content/drive/MyDrive/project/models"
    model_name = "bert-base-uncased"
    batch_size = 8
    num_labels = 4
    num_workers=0  #num_workers=os.cpu_count() leads to CUDA out of memory issues
    learning_rate = 3e-5 
    scheduler = "ReduceLROnPlateau"
    dropout = 0.25
    max_length = 512
    model_save_name = "model.pt" # filler
