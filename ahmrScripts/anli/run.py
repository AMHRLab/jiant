import pandas as pd
from simpletransformers.classification import ClassificationModel
import sklearn

# read the input files
df = pd.read_json("train.jsonl", lines=True)
test = pd.read_json("dev.jsonl", lines=True)

labels = None
with open("train-labels.lst") as l:
    labels = l.readlines()
    labels = [int(int(line.rstrip()) == 2) for line in labels]

train_labels = pd.DataFrame(labels, columns =['label'])

with open("dev-labels.lst") as l:
    labels = l.readlines()
    labels = [int(int(line.rstrip()) == 2) for line in labels]

dev_labels = pd.DataFrame(labels, columns =['label'])

# process the input data
df = pd.DataFrame({
        'text_a': '[CLS] ' + df['obs1'].replace(r'\n', ' ', regex= True) + ' [SEP] ' + df['obs2'].replace(r'\n', ' ', regex= True) + ' [SEP] ' + df['hyp1'].replace(r'\n', ' ', regex= True),
        'text_b': '[CLS] ' + df['obs1'].replace(r'\n', ' ', regex= True) + ' [SEP] ' + df['obs2'].replace(r'\n', ' ', regex= True) + ' [SEP] ' + df['hyp2'].replace(r'\n', ' ', regex = True),
        'labels': train_labels['label']
    })


test = pd.DataFrame({ 
        'text_a': '[CLS] ' + test['obs1'].replace(r'\n', ' ', regex= True) + ' [SEP] ' + test['obs2'].replace(r'\n', ' ', regex= True) + ' [SEP] ' + test['hyp1'].replace(r'\n', ' ', regex= True),
        'text_b': '[CLS] ' + test['obs1'].replace(r'\n', ' ', regex= True) + ' [SEP] ' + test['obs2'].replace(r'\n', ' ', regex= True) + ' [SEP] ' + test['hyp2'].replace(r'\n', ' ', regex = True),
        'labels': dev_labels['label']
    })


# shuffle the samples
df = df.sample(frac=1).reset_index(drop=True)
test = test.sample(frac=1).reset_index(drop=True)

# Create a TransformerModel
model = ClassificationModel(
                'roberta', 
                'roberta-large', 
                    num_labels=2, 
                        args=(
                                    {'overwrite_output_dir': True,
                                     'fp16': False,
                                     'num_train_epochs': 2,  
                                     'reprocess_input_data': False,
                                     "learning_rate": 5e-6,                                       
                                     "train_batch_size": 16,
                                     "eval_batch_size": 16,
                                     "weight_decay": 0.2,
                                     "max_seq_length": 128,
                                     "evaluate_during_training_verbose": True,
                                     "evaluate_during_training": True,
                                     "weight_decay": 0,
                                     "do_lower_case": False,
                                     "n_gpu": 4, # can be 1 if you have enough memory
                                     })
                              )
# Train the model
model.train_model(df, eval_df=test)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(test, acc=sklearn.metrics.accuracy_score)
print(result)
# save the results
outputs = open("outputs.txt", "w")
outputs.write(str(model_outputs))
preds = open("preds.txt", "a")
for prediction in wrong_predictions:
    preds.write("text_a: " + prediction.text_a + "\n" + "text_b: " + prediction.text_b + "\n" + "label: " + str(prediction.label) + "\n\n\n")
