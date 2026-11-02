from transformers import AlbertForSequenceClassification, AlbertForQuestionAnswering, AlbertConfig
from torch import nn

def get_albert_model(task, albert_dropout, classifier_dropout, num_labels=None):
    config = AlbertConfig.from_pretrained('albert-base-v2', hidden_dropout_prob=albert_dropout, 
                                         attention_probs_dropout_prob=albert_dropout)
    if num_labels is not None:
        config.num_labels = num_labels

    if task in ['squad_v1', 'squad_v2']:
        model = AlbertForQuestionAnswering.from_pretrained('albert-base-v2', config=config)
    else:
        model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', config=config)
    
    # Apply classifier dropout
    if hasattr(model, 'classifier'):
        model.classifier.dropout = nn.Dropout(classifier_dropout)
    
    return model