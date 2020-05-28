import pandas as pd
from simpletransformers.classification import ClassificationModel
import sklearn
import numpy as np
import os
import itertools
import logging
import torch
import wandb # for logging and graph gen

wandb.init(project="anli")

df = pd.read_json("trainBertScore.jsonl", lines=True)
dev = pd.read_json("devBertScore.jsonl", lines=True)

logging.basicConfig(level=logging.ERROR)
transformers_logger = logging.getLogger("transformers")

labels = None
with open("train-labels.lst") as l:
    labels = l.readlines()
    labels = [int(int(line.rstrip()) == 2) for line in labels]

train_labels = pd.DataFrame(labels, columns =['label'])

with open("dev-labels.lst") as l:
    labels = l.readlines()
    labels = [int(int(line.rstrip()) == 2) for line in labels]

dev_labels = pd.DataFrame(labels, columns =['label'])



sorted_inds = np.argsort(df.bertScore)    #TODO
df = df.iloc[sorted_inds]
train_labels = train_labels.iloc[sorted_inds]


sorted_inds = np.argsort(dev.bertScore)   #TODO
dev = dev.iloc[sorted_inds]
dev_labels = dev_labels.iloc[sorted_inds]

df = pd.DataFrame({
        'text_a': '[CLS] ' + df['obs1'].replace(r'\n', ' ', regex= True) + ' [SEP] ' + df['obs2'].replace(r'\n', ' ', regex= True) + ' [SEP] ' + df['hyp1'].replace(r'\n', ' ', regex= True),
        'text_b': '[CLS] ' + df['obs1'].replace(r'\n', ' ', regex= True) + ' [SEP] ' + df['obs2'].replace(r'\n', ' ', regex= True) + ' [SEP] ' + df['hyp2'].replace(r'\n', ' ', regex = True),
        'labels': train_labels['label'],
        'bertScore': df['bertScore'] #TODO
    })


dev = pd.DataFrame({ 
        'text_a': '[CLS] ' + dev['obs1'].replace(r'\n', ' ', regex= True) + ' [SEP] ' + dev['obs2'].replace(r'\n', ' ', regex= True) + ' [SEP] ' + dev['hyp1'].replace(r'\n', ' ', regex= True),
        'text_b': '[CLS] ' + dev['obs1'].replace(r'\n', ' ', regex= True) + ' [SEP] ' + dev['obs2'].replace(r'\n', ' ', regex= True) + ' [SEP] ' + dev['hyp2'].replace(r'\n', ' ', regex = True),
        'labels': dev_labels['label'],
        
    })

# define grid search parameters
b_size = [32]
n_bins = [5]
models = [('roberta','roberta-large')] #TODO

f = open("experiments.txt", "w+")
for b,n,m in itertools.product(b_size,n_bins,models):
    f.writelines(f"Batch Size {b}, Number Bins {n}, Model {m}")
    f.write("\n")
    if n == 2:
        df_1 = df[(df['bertScore'] < 0.5)]
        df_2 = df
        dfTrain = [df_1.drop(columns='bertScore'),df_2.drop(columns='bertScore')]
    elif n == 3:
        df_1 = df[(df['bertScore'] < 0.33)]
        df_2 = df[(df['bertScore'] < 0.66)]
        df_3 = df
        dfTrain = [df_1.drop(columns='bertScore'),df_2.drop(columns='bertScore'),df_3.drop(columns='bertScore')]
    elif n == 4:
        df_1 = df[(df['bertScore'] < 0.25)]
        df_2 = df[(df['bertScore'] < 0.5)]
        df_3 = df[(df['bertScore'] < 0.75)]
        df_4 = df
        dfTrain = [df_1.drop(columns='bertScore'), df_2.drop(columns='bertScore'), df_3.drop(columns='bertScore'), df_4.drop(columns='bertScore')]
    elif n == 5:
        df_1 = df[(df['bertScore'] < 0.2)]
        df_2 = df[(df['bertScore'] < 0.4)]
        df_3 = df[(df['bertScore'] < 0.6)]
        df_4 = df[(df['bertScore'] < 0.8)]
        df_5 = df
        dfTrain = [df_1.drop(columns='bertScore'), df_2.drop(columns='bertScore'), df_3.drop(columns='bertScore'), df_4.drop(columns='bertScore'), df_5.drop(columns='bertScore')]

    model_path = 'outputs/curriculum_anli'
    for i in range(0, n):
        if os.path.isdir('./outputs/curriculum_anli'):
            print("Found the model! Continuing training with bin " + str(i+1))
        else:
            print("Could not find the model! Starting training with bin " + str(i+1))
            model_path = m[1]

        model = ClassificationModel(m[0], model_path, num_labels=2, 
                                    args={'overwrite_output_dir': True,
                                            'output_dir': 'outputs/curriculum_anli',
                                            'fp16': True,
                                            'num_train_epochs': 1,  
                                            'reprocess_input_data': False,
                                            "learning_rate": 1e-5,                                       
                                            "train_batch_size": b,
                                            "eval_batch_size": b,
                                            "weight_decay": 0.01,
                                            "max_seq_length": 128,
                                            "evaluate_during_training_verbose": True,
                                            "evaluate_during_training": True,
                                            "evaluate_during_training_steps": 2500,
                                            "do_lower_case": True,
                                            "n_gpu": 1, # can be 1 if you have enough memory
                                            'save_eval_checkpoints': False,
                                            'save_model_every_epoch': False,
                                            'warmup_ratio': 0.20,
                                            'wandb_project': "anli",
                                        })

                                    
        # Train the model
        model.train_model(dfTrain[i], eval_df=dev)
        # Evaluate the model
        result, model_outputs, wrong_predictions = model.eval_model(dev, acc=sklearn.metrics.accuracy_score)
        wandb.log({"Accuracy":result})
        f.writelines(str(result))
        f.write("\n")
        print(result)
    
    # delete the saved model and start the next hyperparameter set
    # os.system('rm -r outputs/curriculum_anli')
    model = None
    del model
    torch.cuda.empty_cache()

f.close()
