# BabyBERTa Continual Training

## Pre-train the model
1. `` cd  pre-train``
2. Need to copy the *data_collator.py* to *transformers/data* (installed transformers package) to enable choice of unmasked removal policy
3. The processed data is located in the *data*
4. To continually pre-train the model on more wikipeida data, run 
- `` python3 continue_train.py --text_path {text_path} --save_path {save_model_path} --read_path {starting_point_path} --unmasked_removal True``
- Note that the text file should be located in the *data* 


## Fine-tune BabyBERTa on downstream tasks

- The code for fine-tuning BabyBERTa on downstream tasks are in the folder *fine-tune*
- To fine-tune the model, `` cd fine-tune``

### SRL

- `` cd crfsrl ``
- Check the README in the folder

### QAMR and QASRL

- The processed data of qamr and qasrl are located in the folder *qamr* and *qasrl*, please download the [qamr data] (https://drive.google.com/file/d/1VHGWuxqMn0sFmpUQUB_UhYbAcZjHDRb-/view?usp=sharing), [qasrl data] (https://drive.google.com/file/d/1cXRcum-t50_ARIVZz1Gu1CwOJbKLdRE6/view?usp=sharing) and unzip the file
- To run training the model on QAMR or QASRL, `` python3 train_question_answering.py --path {model_path} --train_model True --evaluate_model True --choice {task_choice} --num_of_epochs {number of epochs}``
- To run evaluation only `` python3 train_question_answering.py --evaluate_path {evaluation_model_path} --evaluate_model True --choice {task_choice}``
- After finished fine-tuning, the model will be saved in the save director with a prefix task_choice, e.g. *qasrl_{model_name}*
- The evaluation results will be saved in  *{task_choice}_{split}_result*


## Model for BabyLM Challenge

- Please find the model for BabyLM-challenge (strict-small task): https://huggingface.co/yangy96/BabyLM_strict_small_Penn-BGU-BabyBERTa
- More details could be found in https://cogcomp.seas.upenn.edu/papers/YSLR23b.pdf
