from utilities import *

class MedDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset object for our medical texts dataset
    """
    def __init__(self, dataframe, tokenizer, mode="train", max_length=512):
        self.texts = list(dataframe['text'].values)
        self.mode = mode

        if mode != "test":
            self.labels = dataframe['label'].values

        self.encodings = tokenizer(
            self.texts, 
            padding=True, 
            truncation=True, 
            max_length=max_length,
            return_tensors="pt"
        )
        
    def __getitem__(self, idx):
        item = {key: values[idx] for key, values in self.encodings.items()}
        
        if self.mode != "test":
            item['labels'] = torch.tensor(self.labels[idx])
            
        return item
    
    def __len__(self):
        return len(self.texts)
