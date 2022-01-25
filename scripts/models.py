from utilities import *

class COReClassifier(nn.Module):
    def __init__(self, num_labels, bert_hidden_dim=768, classifier_hidden_dim=768, dropout=None):
        super().__init__()

        # pull the CORe model
        self.model = AutoModelForSequenceClassification.from_pretrained("bvanaken/CORe-clinical-diagnosis-prediction")
        
        # change dropout to desired p value
        self.model.dropout = nn.Dropout(dropout) if dropout is not None else nn.Identity()

        # change classifier to output 4 classes, and add extra layer
        # self.model.classifier = nn.Linear(768,4)
        self.model.classifier = nn.Sequential(nn.Linear(bert_hidden_dim, classifier_hidden_dim),
                                  nn.ReLU(),
                                  nn.Dropout(dropout) if dropout is not None else nn.Identity(),
                                  nn.Linear(classifier_hidden_dim, num_labels))
        
    def forward(self, batch):
        # feed batch into model, returns SequenceOutput that has logits attribute
        output = self.model(input_ids=batch['input_ids'],
                            attention_mask = batch['attention_mask'],
                            token_type_ids = batch['token_type_ids']) 
        
        return output.logits


class BERTClassifier(nn.Module):
    def __init__(self, num_labels, bert_hidden_dim=768, classifier_hidden_dim=768, dropout=None):
        
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

        self.model.dropout = torch.nn.Dropout(dropout)

        # nn.Identity does nothing if the dropout is set to None
        self.model.classifier = nn.Sequential(nn.Linear(bert_hidden_dim, classifier_hidden_dim),
                                  nn.ReLU(),
                                  nn.Dropout(dropout) if dropout is not None else nn.Identity(),
                                  nn.Linear(classifier_hidden_dim, num_labels))


    def forward(self, batch):
        # feed batch into model, returns SequenceOutput that has logits attribute
        output = self.model(input_ids=batch['input_ids'],
                            attention_mask = batch['attention_mask'],
                            token_type_ids = batch['token_type_ids']) 
        
        return output.logits

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()
    
    def reset(self):
        self.avg, self.sum, self.count = [0]*3
    
    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count
    
    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

