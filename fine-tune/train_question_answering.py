from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoConfig
from transformers import TrainingArguments, Trainer, DefaultDataCollator
from evaluate import load
from tqdm import tqdm
import torch
import json
import numpy as np
import collections
import argparse
import sys, os
import matplotlib.pyplot as plt
import time
import random

#sys.path.insert(0,'/home/yangy96/BabyBERTa/crfsrl')


#following huggingface tutorial of question answering: https://huggingface.co/course/chapter7/7?fw=pt
# to call the script, python3 train_question_answering.py --path roberta-base --train_model True --evaluate_path None --evaluate_model True

#compress tokenizer warning 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"


def preprocess_function(examples):
    max_seq_length = 120
    questions = [q.lstrip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=120,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
        
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=120,
        truncation="only_second",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else (None,None) for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


def evaluate_question_answering(model, n_best=20,max_answer_length =30, split="validation"):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model  = model.to(device)
    model.eval()
    predicted_answers=[]
    references=[]
    
    result_list = []
    
    for i in tqdm(range(0,len(squad[split]),100)):
        small_eval_set = squad[split].select(range(i,min(len(squad[split]),i+100)))
        eval_set = small_eval_set.map(preprocess_validation_examples,batched=True,remove_columns=small_eval_set.column_names,)
        eval_set_for_model = eval_set.remove_columns(["example_id", "offset_mapping"])
        eval_set_for_model.set_format("torch")
        
        batch = {k: eval_set_for_model[k].to(device) for k in eval_set_for_model.column_names}
        with torch.no_grad():
            predictions =model(**batch)

        start_logits = predictions.start_logits.cpu().numpy()
        end_logits = predictions.end_logits.cpu().numpy()
        example_to_features = collections.defaultdict(list)
        offset_list = np.array(eval_set["offset_mapping"])


        for idx, feature in enumerate(eval_set):
            example_to_features[feature["example_id"]].append(idx)

        start = 0

        for example in small_eval_set:
            
            example_id = example["id"]
            context = example["context"]
            answers = []
            
            for feature_index in example_to_features[example_id]:
                
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = offset_list[feature_index]
                
                start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
                end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
                
                
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Skip answers that are not fully in the context
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # Skip answers with a length that is either < 0 or > max_answer_length.
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                        ):
                            continue

                        answers.append(
                            {
                                "text": context[offsets[start_index][0] : offsets[end_index][1]],
                                "logit_score": start_logit[start_index] + end_logit[end_index],
                            }
                        )
            #if len(answers) > 0:
            #    best_answer = max(answers, key=lambda x: x["logit_score"])
            #    predicted_answers.append({"id": example_id, "prediction_text": best_answer["text"]})
            #else:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append({"id": example_id, "prediction_text": best_answer["text"]})
            references.append({"id":example_id,"answers":example["answers"]})
            #print(best_answer["text"],example["answers"])
            correct = False
            for gold in example["answers"]['text']:
                if best_answer["text"] == gold:
                    correct = True
                    result_list.append(1)
                    break
            if not correct: 
                result_list.append(0)
            #print('-----------------------------------------------')
            #print(context)
            #print("answers: ",{"id": example_id, "prediction_text": best_answer["text"]}, len(context),example["question"])
            #print("reference: ", {"id":example_id,"answers":example["answers"]})

    return predicted_answers,references



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess qamr file into huggingface format')
    parser.add_argument('--path',type=str, default="roberta-base", help='path for reading pretrained model')
    parser.add_argument('--train_model',type=bool, default=False, help='whether to train a new question answering model')
    parser.add_argument('--evaluate_path',type=str, default=None, help='path for fine-tuned model')
    parser.add_argument('--evaluate_model',type=bool, default=False, help='whether to evaluate ')
    parser.add_argument('--result_folder',type=str, default=None, help='folder of training logs ')
    parser.add_argument('--num_of_epochs',type=int, default=3, help='number of epochs ')
    parser.add_argument('--choice', type=str, default = "qasrl")
    parser.add_argument('--split', type=str, default = "test")
    parser.add_argument('--steps',type=int, default=20000, help='number of steps ')
    parser.add_argument('--seed',type=int, default=523, help='number of steps ')
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    path = args.path
    evaluate_path = args.evaluate_path
    print(args)

    if args.choice == "qasrl":
        squad = load_dataset('json',data_files={'train':'../qasrl/preprocess_qasrl.train.json','validation':'../qasrl/preprocess_qasrl.dev.json','test':'../qasrl/preprocess_qasrl.test.json'},field='data')
    elif args.choice == "qamr":
        squad = load_dataset('json',data_files={'train':'../qamr/preprocess_train.json','validation':'../qamr/preprocess_dev.json','test':'../qamr/preprocess_test.json'},field='data')
    elif args.choice == "squad":
        squad = load_dataset('squad',keep_in_memory=True)
        
    #   print(squad)

    if args.train_model:
        
        if ("RoBERTa" in path.split("/")[-1]) or ("roberta" in path.split("/")[-1]) :
            tokenizer = AutoTokenizer.from_pretrained(path,use_fast=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained('phueb/BabyBERTa-3',use_fast=True, add_prefix_space=True)

        tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)
    
        model = AutoModelForQuestionAnswering.from_pretrained(path)
        print(model)

        
        training_args = TrainingArguments(output_dir="./results_"+args.choice+"_"+path.split('/')[-1], save_strategy="epoch",learning_rate=2e-4,per_device_train_batch_size=16,
            per_device_eval_batch_size=32,num_train_epochs=args.num_of_epochs,weight_decay=0.01, save_total_limit=20,  seed=args.seed)

        trainer = Trainer(model=model,args=training_args,train_dataset=tokenized_squad["train"],eval_dataset=tokenized_squad["validation"],tokenizer=tokenizer,data_collator=DefaultDataCollator())
        trainer.train()
        trainer.save_model("./"+args.choice+"_"+path.split('/')[-1])
        #print("./"+args.choice+"_"+path.split('/')[-1])
        if evaluate_path is None:
            evaluate_path = "./"+args.choice+"_"+path.split('/')[-1]


    if args.evaluate_model:
        split = args.split
        if evaluate_path:
            if ("RoBERTa" in evaluate_path.split("/")[-1]) or ("roberta" in evaluate_path.split("/")[-1]) :
                tokenizer = AutoTokenizer.from_pretrained(evaluate_path,use_fast=True)
            else:
                tokenizer = AutoTokenizer.from_pretrained(evaluate_path,use_fast=True, add_prefix_space=True)
    
            eval_model = AutoModelForQuestionAnswering.from_pretrained(evaluate_path)
            predicted_answers,references = evaluate_question_answering(eval_model,split=split)
            squad_metric = load('squad')
            results = squad_metric.compute(predictions=predicted_answers, references=references)
            with open(evaluate_path+os.sep+args.choice+'_'+split+'_result.json', 'w') as f:
                json.dump(results, f)
            print(results)
        
        
        