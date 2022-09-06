import json
import os
import random
from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from tqdm import tqdm, trange
from src.data import TextDataset
import wandb
from utils.model import save_model
from utils.utils import logger
from sklearn.metrics import accuracy_score, f1_score, precision_score

from src.build_IdBench_dataset import load_data
from utils.calculateProb import *
import scipy.stats as st
import pandas as pd


def pretrain(args, train_dataset: TextDataset, encoder, model, tokenizer, eval_interval=7000):
    """ Train the model """
    # Init wandb
    wandb.init(project='id2vec-TextCNN', config=args, name=args.exp_name)
    # wandb.watch(model)

    train_size = train_dataset.__len__()
    train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size, num_workers=4, pin_memory=True)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    max_steps = len(train_dataloader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps * 0.1,
                                                num_training_steps=max_steps)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  batch size = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", max_steps)
    best_acc, best_loss, best_pre, best_f1 = -1, 1e8, -1, -1
    model.zero_grad()

    global_step = 0
    tr_loss = 0.0
    for idx in range(args.num_train_epochs):  # epoch
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        losses = []
        for step, batch in enumerate(bar):  # step
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            id_len = batch[2].to(args.device)
            mask_indices = batch[3]
            m_label = batch[4].to(args.device)
            indices = batch[5]
            # print(inputs.shape)
            hidden_states, embeddings, mean_embeddings = encoder(inputs, id_len, labels, mask_indices)  # calls `forward()`

            loss = model(hidden_states, embeddings, m_label)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            global_step += 1
            if global_step % 10 == 0:
                train_loss = tr_loss / 10
                wandb.log({"train/loss": train_loss}, step=global_step)
                tr_loss = 0.0
            losses.append(loss.item())
            bar.set_description("epoch {} loss {}".format(idx, round(np.mean(losses), 3)))
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if (step + 1) % eval_interval == 0:
                results = evaluate_pretrained_model(args, encoder, model, tokenizer)
                for key, value in results.items():
                    logger.info("  %s = %s", key, round(value, 4))

                    # Save model checkpoint
                if results['eval_acc'] > best_acc:
                    best_acc = results['eval_acc']
                    logger.info("  " + "*" * 20)
                    logger.info("  Best acc:%s", round(best_acc, 4))
                    logger.info("  " + "*" * 20)

                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-acc', 'epoch_' + str(idx))
                    save_model(model, output_dir)

                # Save model checkpoint
                if results['eval_loss'] < best_loss:
                    best_loss = results['eval_loss']

                    logger.info("  " + "*" * 20)
                    logger.info("  Best loss:%s", round(best_loss, 4))
                    logger.info("  " + "*" * 20)

                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-loss', 'epoch_' + str(idx))
                    save_model(model, output_dir)

                if results['eval_f1'] > best_f1:
                    best_f1 = results['eval_f1']

                    logger.info("  " + "*" * 20)
                    logger.info("  Best f1:%s", round(best_f1, 4))
                    logger.info("  " + "*" * 20)

                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-f1', 'epoch_' + str(idx))
                    save_model(model, output_dir)

                if results['eval_pres'] > best_pre:
                    best_pre = results['eval_pres']

                    logger.info("  " + "*" * 20)
                    logger.info("  Best f1:%s", round(best_pre, 4))
                    logger.info("  " + "*" * 20)

                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-precision', 'epoch_' + str(idx))
                    save_model(model, output_dir)

        wandb.log({'train/loss_per_epoch': np.mean(losses)})
    if args.exp_name == 'no_pretrain':
        print('save model directly')
        output_dir = os.path.join(args.output_dir, 'codebert-base-mlm')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        output_dir = os.path.join(output_dir, '{}'.format('model_'+str(idx)+'.bin'))
        torch.save(model_to_save.state_dict(), output_dir)


def evaluate_pretrained_model(args, encoder, model, tokenizer):
    eval_output_dir = args.output_dir

    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4,
                                 pin_memory=True)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    labels = []
    preds = []
    for i, batch in enumerate(eval_dataloader):
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        id_len = batch[2].to(args.device)
        mask_indices = batch[3]
        m_label = batch[4].to(args.device)
        indices = batch[5]

        with torch.no_grad():
            hidden_states, embeddings, mean_embeddings = encoder(inputs, id_len, label, mask_indices)
            seq_embeddings = model.predict(hidden_states)
            identifier_ebd = []
            for row_sub_tokens_ebd in embeddings:
                row_sub_tokens_ebd = torch.reshape(row_sub_tokens_ebd, (1, -1, 768))
                row_identifier_ebd = model.predict(row_sub_tokens_ebd)
                identifier_ebd.append(row_identifier_ebd)
            batched_identifiers_ebd = torch.cat(identifier_ebd, dim=0)

            out = torch.mul(seq_embeddings, batched_identifiers_ebd)
            out = model.fc2(out)
            out = model.sig(out)

            # print(out)
            for _, x in enumerate(out):
                if x[0].item() > x[1].item():
                    # print(True)
                    preds.append(1)
                else:
                    # print(False)
                    preds.append(0)

            # print(m_label)
            for _, x in enumerate(m_label):
                labels.append(int(x[0].item()))

            out = torch.reshape(out, (-1,))
            m_label = torch.reshape(m_label, (-1,))
            loss = model.loss(out, m_label)
            eval_loss += loss.mean().item()

        nb_eval_steps += 1

    eval_acc = accuracy_score(labels, preds)
    eval_pres = precision_score(labels, preds)
    eval_f1 = f1_score(labels, preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    result = {
        "eval_loss": float(perplexity),
        "eval_acc": round(eval_acc, 4),
        "eval_pres": float(eval_pres),
        "eval_f1": round(eval_f1, 4)
    }
    wandb.log({
        'val/loss_per_epoch': float(perplexity),
        'val/acc_per_epoch': eval_acc,
        "val/eval_pres": float(eval_pres),
        "val/eval_f1": round(eval_f1, 4)
    })
    return result


def evaluate_task_1_idbench(args, encoder, model, tokenizer):
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    eval_data = load_data()

    # Eval!
    logger.info("***** Running evaluation task IdBench*****")
    model.eval()
    cnt = 0
    dPrime = dict()
    score = dict()
    cs_score = dict()
    cos_similarities = []
    large_score = []
    record_result = []
    for row in eval_data:
        cnt += 1
        var1 = row['var1']
        var2 = row['var2']
        context1 = row['context1']
        context2 = row['context2']
        hit1 = hit2 = false1 = false2 = 0
        for ct in context1:
            code = ct[0]
            mask_location = ct[1]
            choice, cos_similarity = compareTwoVar(args, var1=var1, var2=var2, context=code, maskLocations=mask_location,
                                   tokenizer=tokenizer, model=model, encoder=encoder)
            hit1 += choice[0]
            false1 += choice[1]
            cos_similarities.append(cos_similarity)

        for ct in context2:
            code = ct[0]
            mask_location = ct[1]
            choice, cos_similarity = compareTwoVar(args, var1=var1, var2=var2, context=code, maskLocations=mask_location,
                                   tokenizer=tokenizer, model=model, encoder=encoder)
            hit2 += choice[1]
            false2 += choice[0]
            cos_similarities.append(cos_similarity)

        cs = np.array(cos_similarities).mean()
        cos_similarities = []
        distance = st.norm.ppf(hit1 / (hit1 + false1)) - st.norm.ppf(false2 / (hit2 + false2))
        print(distance)
        dPrime.setdefault((var1, var2), 0)
        dPrime[(var1, var2)] = distance

        cs_score.setdefault((var1, var2), 0)
        cs_score[(var1, var2)] = cs
        large_score.append(cs)

        record_result.append([var1, var2, cs])
        # assert 0

    result_path = os.path.join('./results', 'idbench_res_' + args.exp_name + '_' + args.textcnn_path.split('/')[-2] + '.json')
    with open(result_path, 'w') as f:
        f.write(json.dumps(record_result))
    minD = 100000
    maxD = -100000
    for _, dist in dPrime.items():
        minD = min(minD, dist)
        maxD = max(maxD, dist)
    outputData = {
        'var1': [],
        'var2': [],
        'contextual_similarity_score': [],
        'dprime': []
    }
    for pair, dist in dPrime.items():
        score.setdefault(pair, 1.0 - (float(dist) - float(minD)) / (float(maxD) - float(minD)))
        outputData['var1'].append(pair[0])
        outputData['var2'].append(pair[1])
        outputData['contextual_similarity_score'].append(
            1.0 - (float(dist) - float(minD)) / (float(maxD) - float(minD)))
        outputData['dprime'].append(dist)

    df = pd.DataFrame(outputData)
    if args.exp_name == 'id2vec-textcnn_js_2':
        output_path = './results/idbench_res_js_2.jsonl'
    elif args.exp_name == 'id2vec-textcnn_5':
        output_path = './results/idbench_res_tc5.jsonl'
    else:
        output_path = './results/idbench_res_3.jsonl'
    df.to_csv(output_path, index=True)
    print('large: ', np.array(large_score).mean(), len(large_score))

    mediumIdBench = pd.read_csv('data/IdBench/medium_pair_wise.csv')
    medium_score = []
    for idx in range(mediumIdBench.shape[0]):
        row = mediumIdBench.iloc[idx]
        if (row['contextual_similarity'] == 'NAN'):
            continue
        var1 = row['id1']
        var2 = row['id2']
        if (not (var1, var2) in cs_score):
            continue
        medium_score.append(cs_score[(var1, var2)])

    print('medium: ', np.array(medium_score).mean(), len(medium_score))

    smallData = {
        'var1': [],
        'var2': [],
        'contextual_similarity_score': []
    }
    small_score = []
    smallIdBench = pd.read_csv('data/IdBench/small_pair_wise.csv')
    for idx in range(smallIdBench.shape[0]):
        row = smallIdBench.iloc[idx]
        if (row['contextual_similarity'] == 'NAN'):
            continue
        var1 = row['id1']
        var2 = row['id2']
        if (not (var1, var2) in cs_score):
            continue
        small_score.append(cs_score[(var1, var2)])

    print('short: ', np.array(small_score).mean(), len(small_score))

    return 0


def evaluate_task_2_rename(args, encoder, model, tokenizer):
    output_path = os.path.join('./results', 'name_ref_' + args.exp_name + '_' + args.textcnn_path.split('/')[-2])
    eval_output_dir = args.output_dir
    name_ref_survey_path = os.path.join('./results', 'name_ref_survey_' + args.exp_name + '_' + args.textcnn_path.split('/')[-2])
    print(output_path)
    print(name_ref_survey_path)
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    eval_data_path = args.eval_rename_data_file
    with open(eval_data_path, 'r') as f:
        lines = f.readlines()
    eval_data = []
    id_set = []
    for row in lines:
        sample = json.loads(row)
        eval_data.append(sample)
        identifier = sample['after']
        if len(identifier) < 4:
            continue
        if identifier not in id_set:
            id_set.append(identifier)
    print('len of set: ', len(id_set))
    with open('./id_set.json', 'w') as f:
        f.write(json.dumps(id_set))

    # Eval!
    logger.info("***** Running evaluation task IdBench*****")
    model.eval()
    cnt = 0
    dPrime = dict()
    score = dict()
    rng = random.Random(1234)
    res = []
    context_with_ids_with_ebd = []
    for row in tqdm(eval_data):
        rng.shuffle(id_set)
        res_dict = {}
        cnt += 1
        var1 = row['before']
        var2 = row['after']
        var2_set = []
        for identifier in id_set:
            if identifier != var2:
                var2_set.append(identifier)
            if len(var2_set) >= 199:
                break
        context = row['source']
        tmp = [context, [var1, var2]]
        mask_location = row['mask_locations']
        try:
            choice, cos_similarity = compareTwoVar(args, var1=var1, var2=var2, context=context,
                                                   maskLocations=mask_location, tokenizer=tokenizer, model=model,
                                                   encoder=encoder)
        except:
            continue

        id_prob = choice[1] / 100
        res_dict[var2] = id_prob
        # target = (var2, id_prob)
        for identifier in tqdm(var2_set):
            try:
                choice, cos_similarity = compareTwoVar(args, var1=var1, var2=identifier, context=context,
                                                       maskLocations=mask_location, tokenizer=tokenizer, model=model,
                                                       encoder=encoder)
                res_dict[identifier] = choice[1] / 100
            except:
                res_dict[identifier] = 0.0

        sorted_x = sorted(res_dict.items(), key=lambda kv: kv[1], reverse=True)
        tmp.append([sorted_x[i] for i in range(10)])
        # print(sorted_x)
        logger.info('len res dict: ' + str(len(res_dict)))
        assert len(res_dict) == 200
        rank = 201
        for idx, value in enumerate(sorted_x):
            if value[0] == var2:
                rank = idx
                break
        assert rank <= 200
        logger.info('rank: ' + str(rank))
        res.append(rank)
        context_with_ids_with_ebd.append(tmp)
        with open(name_ref_survey_path, 'w') as f:
            f.write(json.dumps(context_with_ids_with_ebd))
        with open(output_path, 'w') as f:
            f.write(json.dumps(res))


