import time
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import random

import sys
sys.path.append('/home/yangy96/BabyBERTa/BabyBERTa/')

import transformers
from transformers.models.roberta import RobertaForMaskedLM, RobertaConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from babyberta import configs
from babyberta.io import load_sentences_from_file, load_tokenizer
from babyberta.params import Params
from babyberta.utils import split, make_sequences, forward_mlm
from babyberta.probing import do_probing
from babyberta.dataset import DataSet


from tqdm import tqdm


def main(param2val):

    #assert transformers.__version__ == '4.3.3'
    #assert torch.__version__.startswith('1.6.0')

    # params
    params = Params.from_param2val(param2val)
    params.framework = 'huggingface'
    params.is_huggingface_recommended = False
    print(params, flush=True)

    #  path to root folder on shared drive where results are saved, and data is loaded
    project_path = Path(param2val['project_path'])

    # probing path - contains probing sentences
    probing_path = configs.Dirs.probing_sentences
    if not probing_path.exists():
        raise FileNotFoundError(f'Path to probing sentences does not exist: {probing_path}.'
                                'Probing sentences can be downloaded from github.com/phueb/Zorro/sentences')

    # save_path - locations where probing results are saved
    save_path = Path(param2val['save_path'])
    if not save_path.exists():
        save_path.mkdir(parents=True)

    # Byte-level BPE tokenizer
    path_tokenizer_config = project_path / 'data' / 'tokenizers' / f'{params.tokenizer}.json'
    tokenizer = load_tokenizer(path_tokenizer_config, params.max_input_length)
    #tokenizer = transformers.RobertaTokenizer.from_pretrained("roberta-base")
    vocab_size = len(tokenizer.get_vocab())
    print(f'Vocab size={vocab_size}')

    # load sentences from corpora
    sentences = []
    for corpus_name in params.corpora:
        data_path = project_path / 'data' / 'corpora' / f'{corpus_name}.txt'
        sentences_in_corpus = load_sentences_from_file(data_path,
                                                       include_punctuation=params.include_punctuation,
                                                       allow_discard=True)
        print(f'Loaded {len(sentences_in_corpus):>12,} sentences from {corpus_name}')
        sentences += sentences_in_corpus

    # training order (do this after loading all corpora, otherwise re-ordering is performed within each corpus
    if params.training_order == 'shuffled':
        random.shuffle(sentences)
    elif params.training_order == 'original':
        pass
    elif params.training_order == 'reversed':
        sentences = sentences[::-1]
    else:
        raise AttributeError('Invalid arg to training_order.')

    all_sequences = make_sequences(sentences, params.num_sentences_per_input)
    train_sequences, devel_sequences = split(all_sequences)
    print(len(train_sequences),len(devel_sequences))

    # BabyBERTa
    print('Preparing BabyBERTa...')

    """config = RobertaConfig(vocab_size=vocab_size,
                           pad_token_id=tokenizer.token_to_id(configs.Data.pad_symbol),
                           bos_token_id=tokenizer.token_to_id(configs.Data.bos_symbol),
                           eos_token_id=tokenizer.token_to_id(configs.Data.eos_symbol),
                           return_dict=True,
                           is_decoder=False,
                           is_encoder_decoder=False,
                           add_cross_attention=False,
                           layer_norm_eps=params.layer_norm_eps,  # 1e-5 used in fairseq
                           max_position_embeddings=params.max_input_length + 2,
                           hidden_size=params.hidden_size,
                           num_hidden_layers=params.num_layers,
                           num_attention_heads=params.num_attention_heads,
                           intermediate_size=params.intermediate_size,
                           initializer_range=params.initializer_range,
                           )
    # load weights from previous checkpoint
    if params.load_from_checkpoint.startswith('param'):
        path_tmp = Path(param2val['project_path']) / 'runs' / params.load_from_checkpoint
        model_files = list(path_tmp.rglob('**/saves/*.bin'))
        print(f'Found {len(model_files)} saved models')
        path_cpt = random.choice(model_files)
        print(f'Trying to load model from {path_cpt.parent}')
        model = RobertaForMaskedLM.from_pretrained(path_cpt.parent)
    # initialize random weights
    else:
        model = RobertaForMaskedLM(config=config)"""

    model = RobertaForMaskedLM.from_pretrained("/home/yangy96/BabyBERTa/BabyBERTa/saved_models/BabyBERTa_AO-CHILDES")
    print('Number of parameters: {:,}'.format(model.num_parameters()), flush=True)
    model.cuda(0)

    train_dataset = DataSet(train_sequences, tokenizer, params)
    devel_dataset = DataSet(devel_sequences, tokenizer, params)
    # pre-compute batches once (otherwise StopIteration after epoch=1)
    train_batches = [batch for batch in train_dataset]
    devel_batches = [batch for batch in devel_dataset] if devel_dataset.data else []

    # count number of steps in training data
    max_step = train_dataset.num_batches * params.num_epochs
    print(f'max step={max_step:,}', flush=True)

    # optimizer + lr schedule
    optimizer = AdamW(model.parameters(),
                      lr=params.lr,
                      weight_decay=params.weight_decay,
                      correct_bias=False)  # does not implement lr scheduling
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=params.num_warmup_steps,
                                                num_training_steps=max_step)

    # init
    name2xy = {}
    train_start = time.time()
    loss = None
    step = 0
    is_evaluated_at_current_step = False
    is_first_time_in_loop = True

    # train + eval loop
    for epoch_id in tqdm(range(params.num_epochs)):
        for x, y, mm in train_batches:

            if not is_first_time_in_loop:  # do not influence first evaluation by training on first batch
                # forward
                model.train()
                loss = forward_mlm(model, mm, x, y)
                # backward + update
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # otherwise only punctuation is predicted
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()  # needed ?
                model.zero_grad()
                step += 1

            is_first_time_in_loop = False

            # eval
            if step % configs.Eval.interval == 0:
                is_evaluated_at_current_step = True

                # pp
                if configs.Data.train_prob < 1.0:  # if there are eval and test data
                    model.eval()
                    pp_sum = 0
                    num_steps = 0
                    for x_eval, y_eval, mm_eval in devel_batches:
                        loss = forward_mlm(model, mm_eval, x_eval, y_eval)
                        pp = torch.exp(loss).detach().cpu().numpy().item()
                        pp_sum += pp
                        num_steps += 1
                        model.zero_grad()
                    pp = pp_sum / num_steps
                    name2xy.setdefault(f'dev_pps', []).append((step, pp))
                    print(f'dev pp={pp}', flush=True)

                # probing on grammar test suite
                for paradigm_path in probing_path.rglob(f'*.txt'):
                    do_probing(save_path, paradigm_path, model, step,
                               params.include_punctuation, tokenizer=tokenizer)

                if max_step - step < configs.Eval.interval:  # no point in continuing training
                    print('Detected last eval step. Exiting training loop', flush=True)
                    break

                if configs.Training.max_step is not None and step >= configs.Training.max_step:
                    print('Reached manually set max_step. Exiting training loop', flush=True)
                    break

            # console
            if is_evaluated_at_current_step or step % configs.Training.feedback_interval == 0:
                min_elapsed = (time.time() - train_start) // 60
                pp = torch.exp(loss) if loss is not None else np.nan
                print(f'epoch={epoch_id + 1:>3,}/{params.num_epochs} step={step:>9,}/{max_step:>9,}\n'
                      f'pp={pp :2.4f} \n'
                      f'lr={scheduler.get_lr()[0]} \n'
                      f'total minutes elapsed={min_elapsed:<3}\n', flush=True)
                is_evaluated_at_current_step = False

    # prepare collected data for returning to Ludwig
    performance_curves = []
    for name, xy in name2xy.items():
        print(f'Making pandas series with name={name} and length={len(xy)}', flush=True)
        x, y = zip(*xy)
        s = pd.Series(y, index=x)
        s.name = name
        performance_curves.append(s)

    # save  # TODO these are not saved at the same training step across differently sized corpora
    model.save_pretrained(save_path)
    #config.save_pretrained(save_path)
    tokenizer.save(str(save_path / 'tokenizer.json'), pretty=True)

    print('Reached end of babyberta.job.main', flush=True)

    return performance_curves

param2default = {
    # data
    'sample_with_replacement': False,  # this must be False if corpus order is to be preserved during training
    'training_order': 'original',  # original or shuffled, use this alongside consecutive_masking=True
    'consecutive_masking': False,  # better dev pp and grammatical accuracy when false
    'num_sentences_per_input': 1,  # if too large -> may exceed CUDA memory, 1 is best for good number-agreement
    'include_punctuation': True,
    'allow_truncated_sentences': False,
    'num_mask_patterns': 10,  # 10 is better than fewer
    'mask_pattern_size': 2,  # used only if probabilistic_masking = False
    'probabilistic_masking': True,
    'mask_probability': 0.15,  # used only if probabilistic_masking = true
    'leave_unmasked_prob_start': 0.1,   # better performance if no unmasking
    'leave_unmasked_prob': 0.1,  # better performance if no unmasking
    'random_token_prob': 0.1,
    'corpora': ('aochildes',),
    'tokenizer': 'babyberta',  # larger than 8k slightly reduces performance
    'add_prefix_space': True,  # better if True, whether to treat first token like any other token (False in GPT-2)
    'max_input_length': 128,  # unacceptable performance if lower than ~32

    # training
    'batch_size': 16,
    'lr': 1e-4,  # 1e-4 is used in fairseq (and performs better here), and 1e-3 is default in huggingface
    'num_epochs': 10,  # use 1 epoch to use dynamic masking
    'num_warmup_steps': 24_000,  # 24K used in Roberta-base
    'weight_decay': 0.0,

    # model change for roberta
    'load_from_checkpoint': 'none',
    'hidden_size': 256,
    'num_layers': 8,
    'num_attention_heads': 8,
    'intermediate_size': 1024,
    'initializer_range': 0.02,  # stdev of trunc normal for initializing all weights
    'layer_norm_eps': 1e-5,  # 1e-5 default in fairseq (and slightly better performance), 1e-12 default in hgugingface,
    'project_path': '/home/yangy96/BabyBERTa/BabyBERTa/',
    'save_path':'./saved_wiki'
}

if __name__ == "__main__":
    main(param2default)

