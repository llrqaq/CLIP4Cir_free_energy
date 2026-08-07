from comet_ml import Experiment
import json
import multiprocessing
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean, harmonic_mean, geometric_mean
from typing import List
import clip
import numpy as np
import pandas as pd
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import base_path, squarepad_transform, FashionIQDataset, targetpad_transform, CIRRDataset
from combiner import Combiner
from prob_utils import ProbabilisticHead, free_energy_distance, save_prob_head
from utils import collate_fn, update_train_running_results, set_train_bar_description, save_model, \
    extract_index_features, generate_randomized_fiq_caption, device
from validate import compute_cirr_val_metrics, compute_fiq_val_metrics


def combiner_training_fiq(train_dress_types: List[str], val_dress_types: List[str],
                          projection_dim: int, hidden_dim: int, num_epochs: int, clip_model_name: str,
                          combiner_lr: float, batch_size: int, clip_bs: int, validation_frequency: int,
                          transform: str, save_training: bool, save_best: bool, **kwargs):
    """
    Train the Combiner on FashionIQ dataset keeping frozed the CLIP model
    :param train_dress_types: FashionIQ categories to train on
    :param val_dress_types: FashionIQ categories to validate on
    :param projection_dim: Combiner projection dimension
    :param hidden_dim: Combiner hidden dimension
    :param num_epochs: number of epochs
    :param clip_model_name: CLIP model you want to use: "RN50", "RN101", "RN50x4"...
    :param combiner_lr: Combiner learning rate
    :param batch_size: batch size of the Combiner training
    :param clip_bs: batch size of the CLIP feature extraction
    :param validation_frequency: validation frequency expressed in epoch
    :param transform: preprocess transform you want to use. Should be in ['clip', 'squarepad', 'targetpad']. When
                targetpad is also required to provide `target_ratio` kwarg.
    :param save_training: when True save the weights of the Combiner network
    :param save_best: when True save only the weights of the best Combiner wrt the average_recall metric
    :param kwargs: if you use the `targetpad` transform you should prove `target_ratio` as kwarg. If you want to load a
                fine-tuned version of clip you should provide `clip_model_path` as kwarg. Also supports relevance,
                prob_lr and free_energy_scale for the probabilistic variant.
    """

    training_start = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    combiner_relevance = kwargs.get('relevance', 'dot')
    combiner_rel_tag = combiner_relevance if combiner_relevance in ('dot', 'free_energy', 'cope') else 'dot'
    # Detect Stage-1 CLIP relevance from the clip model path
    clip_path_str = kwargs.get("clip_model_path", "")
    for tag in ('cope', 'free_energy'):
        if tag in clip_path_str:
            clip_rel_tag = tag
            break
    else:
        clip_rel_tag = "dot"
    training_path: Path = Path(
        base_path / f"models/combiner_trained_on_fiq_{clip_model_name}_clip-{clip_rel_tag}_comb-{combiner_rel_tag}_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)

    # Save all the hyperparameters on a file
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)

    clip_model.eval()
    input_dim = clip_model.visual.input_resolution
    feature_dim = clip_model.visual.output_dim

    if transform == "clip":
        preprocess = clip_preprocess
        print('CLIP default preprocess pipeline is used')
    elif transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")

    if kwargs.get("clip_model_path"):
        print('Trying to load the CLIP model')
        clip_model_path = kwargs["clip_model_path"]
        saved_state_dict = torch.load(clip_model_path, map_location=device)
        clip_model.load_state_dict(saved_state_dict["CLIP"])
        print('CLIP model loaded successfully')

    clip_model = clip_model.float()

    idx_to_dress_mapping = {}
    relative_val_datasets = []
    index_features_list = []
    index_names_list = []

    # Define the validation datasets and extract the validation index features for each dress_type
    for idx, dress_type in enumerate(val_dress_types):
        idx_to_dress_mapping[idx] = dress_type
        relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess)
        relative_val_datasets.append(relative_val_dataset)
        classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess)
        index_features_and_names = extract_index_features(classic_val_dataset, clip_model)
        index_features_list.append(index_features_and_names[0])
        index_names_list.append(index_features_and_names[1])

    # Define the combiner and the train dataset
    combiner = Combiner(feature_dim, projection_dim, hidden_dim).to(device, non_blocking=True)

    relevance = kwargs.get('relevance', 'dot')
    use_prob = relevance in ('free_energy', 'cope')
    prob_head = None
    logits_fn = None
    distance_fn = free_energy_distance
    if use_prob:
        from prob_utils import get_prob_distances
        prob_lr = kwargs.get('prob_lr', 1e-4)
        free_energy_scale = kwargs.get('free_energy_scale', 10.0)
        logits_fn, distance_fn = get_prob_distances(relevance)
        print(f'Relevance: {relevance} (scale={free_energy_scale})')
        prob_head = ProbabilisticHead(feature_dim).to(device, non_blocking=True)
    else:
        print('Relevance: dot product (baseline)')

    relative_train_dataset = FashionIQDataset('train', train_dress_types, 'relative', preprocess)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size,
                                       num_workers=4, pin_memory=False, collate_fn=collate_fn,
                                       drop_last=True, shuffle=True)

    # Define the optimizer, the loss and the grad scaler
    param_groups = [{'params': combiner.parameters(), 'lr': combiner_lr}]
    if use_prob:
        param_groups.append({'params': prob_head.parameters(), 'lr': prob_lr})
    optimizer = optim.Adam(param_groups)
    crossentropy_criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    # When save_best == True initialize the best result to zero
    if save_best:
        best_avg_recall = 0

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    # Start with the training loop
    print('Training loop started')
    for epoch in range(num_epochs):
        if torch.cuda.is_available():  # RuntimeError: "slow_conv2d_cpu" not implemented for 'Half'
            clip.model.convert_weights(clip_model)  # Convert CLIP model in fp16 to reduce computation and memory
        with experiment.train():
            train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
            combiner.train()
            train_bar = tqdm(relative_train_loader, ncols=150)
            for idx, (reference_images, target_images, captions) in enumerate(train_bar):  # Load a batch of triplets
                step = len(train_bar) * epoch + idx
                images_in_batch = reference_images.size(0)

                optimizer.zero_grad()

                reference_images = reference_images.to(device, non_blocking=True)
                target_images = target_images.to(device, non_blocking=True)

                # Randomize the training caption in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1 (d) cap2
                flattened_captions: list = np.array(captions).T.flatten().tolist()
                input_captions = generate_randomized_fiq_caption(flattened_captions)
                text_inputs = clip.tokenize(input_captions, truncate=True).to(device, non_blocking=True)

                # Extract the features with CLIP
                with torch.no_grad():
                    reference_images_list = torch.split(reference_images, clip_bs)
                    reference_image_features = torch.vstack(
                        [clip_model.encode_image(mini_batch).float() for mini_batch in reference_images_list])
                    target_images_list = torch.split(target_images, clip_bs)
                    target_image_features = torch.vstack(
                        [clip_model.encode_image(mini_batch).float() for mini_batch in target_images_list])

                    text_inputs_list = torch.split(text_inputs, clip_bs)
                    text_features = torch.vstack(
                        [clip_model.encode_text(mini_batch).float() for mini_batch in text_inputs_list])

                # Compute the logits and the loss
                with torch.cuda.amp.autocast():
                    if use_prob:
                        query_mu = combiner.combine_features(reference_image_features, text_features)
                        target_mu = torch.nn.functional.normalize(target_image_features, dim=-1)
                        query_log_var = prob_head.query_log_var(query_mu)
                        target_log_var = prob_head.target_log_var(target_mu)
                        logits = logits_fn(query_mu, query_log_var, target_mu, target_log_var,
                                           scale=free_energy_scale)
                    else:
                        logits = combiner(reference_image_features, text_features, target_image_features)
                    ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=device)
                    loss = crossentropy_criterion(logits, ground_truth)

                # Backprogate and update the weights
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                experiment.log_metric('step_loss', loss.detach().cpu().item(), step=step)
                update_train_running_results(train_running_results, loss, images_in_batch)
                set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)

            train_epoch_loss = float(
                train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch'])
            experiment.log_metric('epoch_loss', train_epoch_loss, epoch=epoch)

            # Training CSV logging
            training_log_frame = pd.concat(
                [training_log_frame,
                 pd.DataFrame(data={'epoch': epoch, 'train_epoch_loss': train_epoch_loss}, index=[0])])
            training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if epoch % validation_frequency == 0:
            clip_model = clip_model.float()  # In validation we use fp32 CLIP model
            with experiment.validate():
                combiner.eval()
                recalls_at10 = []
                recalls_at50 = []

                # Compute and log validation metrics for each validation dataset (which corresponds to a different
                # FashionIQ category)
                for relative_val_dataset, index_features, index_names, idx in zip(relative_val_datasets,
                                                                                  index_features_list,
                                                                                  index_names_list,
                                                                                  idx_to_dress_mapping):
                    recall_at10, recall_at50 = compute_fiq_val_metrics(relative_val_dataset, clip_model, index_features,
                                                                       index_names, combiner.combine_features,
                                                                       prob_head, distance_fn)
                    recalls_at10.append(recall_at10)
                    recalls_at50.append(recall_at50)

                results_dict = {}
                for i in range(len(recalls_at10)):
                    results_dict[f'{idx_to_dress_mapping[i]}_recall_at10'] = recalls_at10[i]
                    results_dict[f'{idx_to_dress_mapping[i]}_recall_at50'] = recalls_at50[i]
                results_dict.update({
                    f'average_recall_at10': mean(recalls_at10),
                    f'average_recall_at50': mean(recalls_at50),
                    f'average_recall': (mean(recalls_at50) + mean(recalls_at10)) / 2
                })

                print(json.dumps(results_dict))
                experiment.log_metrics(
                    results_dict,
                    epoch=epoch
                )

                # Validation CSV logging
                log_dict = {'epoch': epoch}
                log_dict.update(results_dict)
                validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
                validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

            # Save model
            if save_training:
                models_path = training_path / "saved_models"
                if save_best and results_dict['average_recall'] > best_avg_recall:
                    best_avg_recall = results_dict['average_recall']
                    save_model('combiner', epoch, combiner, training_path)
                    if use_prob:
                        models_path.mkdir(exist_ok=True, parents=True)
                        save_prob_head(models_path / 'prob_head_best.pt', epoch, prob_head)
                elif not save_best:
                    save_model(f'combiner_{epoch}', epoch, combiner, training_path)
                    if use_prob:
                        models_path.mkdir(exist_ok=True, parents=True)
                        save_prob_head(models_path / f'prob_head_{epoch}.pt', epoch, prob_head)


def combiner_training_cirr(projection_dim: int, hidden_dim: int, num_epochs: int, clip_model_name: str,
                           combiner_lr: float, batch_size: int, clip_bs: int, validation_frequency: int, transform: str,
                           save_training: bool, save_best: bool, **kwargs):
    """
    Train the Combiner on CIRR dataset keeping frozen the CLIP model
    :param projection_dim: Combiner projection dimension
    :param hidden_dim: Combiner hidden dimension
    :param num_epochs: number of epochs
    :param clip_model_name: CLIP model you want to use: "RN50", "RN101", "RN50x4"...
    :param combiner_lr: Combiner learning rate
    :param batch_size: batch size of the Combiner training
    :param clip_bs: batch size of the CLIP feature extraction
    :param validation_frequency: validation frequency expressed in epoch
    :param transform: preprocess transform you want to use. Should be in ['clip', 'squarepad', 'targetpad']. When
                targetpad is also required to provide `target_ratio` kwarg.
    :param save_training: when True save the weights of the Combiner network
    :param save_best: when True save only the weights of the best Combiner wrt three different averages of the metrics
    :param kwargs: if you use the `targetpad` transform you should prove `target_ratio` as kwarg. If you want to load a
                fine-tuned version of clip you should provide `clip_model_path` as kwarg.
    """

    training_start = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    combiner_relevance = kwargs.get('relevance', 'dot')
    combiner_rel_tag = combiner_relevance if combiner_relevance in ('dot', 'free_energy', 'cope') else 'dot'
    clip_path_str = kwargs.get("clip_model_path", "")
    for tag in ('cope', 'free_energy'):
        if tag in clip_path_str:
            clip_rel_tag = tag
            break
    else:
        clip_rel_tag = "dot"
    training_path: Path = Path(
        base_path / f"models/combiner_trained_on_cirr_{clip_model_name}_clip-{clip_rel_tag}_comb-{combiner_rel_tag}_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)

    # Save all the hyperparameters on a file
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)

    clip_model.eval()
    input_dim = clip_model.visual.input_resolution
    feature_dim = clip_model.visual.output_dim

    if transform == "clip":
        preprocess = clip_preprocess
        print('CLIP default preprocess pipeline is used')
    elif transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")

    if kwargs.get("clip_model_path"):
        print('Trying to load the fine-tuned CLIP model')
        clip_model_path = kwargs["clip_model_path"]
        state_dict = torch.load(clip_model_path, map_location=device)
        clip_model.load_state_dict(state_dict["CLIP"])
        print('CLIP model loaded successfully')

    clip_model = clip_model.float()

    # Define the validation datasets and extract the validation index features
    relative_val_dataset = CIRRDataset('val', 'relative', preprocess)
    classic_val_dataset = CIRRDataset('val', 'classic', preprocess)
    val_index_features, val_index_names = extract_index_features(classic_val_dataset, clip_model)

    # Define the combiner and the train dataset
    combiner = Combiner(feature_dim, projection_dim, hidden_dim).to(device, non_blocking=True)

    relevance = kwargs.get('relevance', 'dot')
    use_prob = relevance in ('free_energy', 'cope')
    prob_head = None
    logits_fn = None
    distance_fn = free_energy_distance
    if use_prob:
        from prob_utils import get_prob_distances
        prob_lr = kwargs.get('prob_lr', 1e-4)
        free_energy_scale = kwargs.get('free_energy_scale', 10.0)
        logits_fn, distance_fn = get_prob_distances(relevance)
        print(f'Relevance: {relevance} (scale={free_energy_scale})')
        prob_head = ProbabilisticHead(feature_dim).to(device, non_blocking=True)
    else:
        print('Relevance: dot product (baseline)')

    relative_train_dataset = CIRRDataset('train', 'relative', preprocess)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size, num_workers=8,
                                       pin_memory=False, collate_fn=collate_fn, drop_last=True, shuffle=True)

    # Define the optimizer, the loss and the grad scaler
    param_groups = [{'params': combiner.parameters(), 'lr': combiner_lr}]
    if use_prob:
        param_groups.append({'params': prob_head.parameters(), 'lr': prob_lr})
    optimizer = optim.Adam(param_groups)
    crossentropy_criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    # When save_best == True initialize the best results to zero
    if save_best:
        best_harmonic = 0
        best_geometric = 0
        best_arithmetic = 0

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    # Start with the training loop
    print('Training loop started')
    for epoch in range(num_epochs):
        if torch.cuda.is_available():  # RuntimeError: "slow_conv2d_cpu" not implemented for 'Half'
            clip.model.convert_weights(clip_model)  # Convert CLIP model in fp16 to reduce computation and memory
        with experiment.train():
            train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
            combiner.train()
            train_bar = tqdm(relative_train_loader, ncols=150)
            for idx, (reference_images, target_images, captions) in enumerate(train_bar):  # Load a batch of triplets
                images_in_batch = reference_images.size(0)
                step = len(train_bar) * epoch + idx

                optimizer.zero_grad()

                reference_images = reference_images.to(device, non_blocking=True)
                target_images = target_images.to(device, non_blocking=True)
                text_inputs = clip.tokenize(captions, truncate=True).to(device, non_blocking=True)

                # Extract the features with CLIP
                with torch.no_grad():
                    reference_images_list = torch.split(reference_images, clip_bs)
                    reference_features = torch.vstack(
                        [clip_model.encode_image(mini_batch).float() for mini_batch in reference_images_list])
                    target_images_list = torch.split(target_images, clip_bs)
                    target_features = torch.vstack(
                        [clip_model.encode_image(mini_batch).float() for mini_batch in target_images_list])

                    text_inputs_list = torch.split(text_inputs, clip_bs)
                    text_features = torch.vstack(
                        [clip_model.encode_text(mini_batch).float() for mini_batch in text_inputs_list])

                # Compute the logits and loss
                with torch.cuda.amp.autocast():
                    if use_prob:
                        query_mu = combiner.combine_features(reference_features, text_features)
                        target_mu = torch.nn.functional.normalize(target_features, dim=-1)
                        query_log_var = prob_head.query_log_var(query_mu)
                        target_log_var = prob_head.target_log_var(target_mu)
                        logits = logits_fn(query_mu, query_log_var, target_mu, target_log_var,
                                           scale=free_energy_scale)
                    else:
                        logits = combiner(reference_features, text_features, target_features)
                    ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=device)
                    loss = crossentropy_criterion(logits, ground_truth)

                # Backpropagate and update the weights
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                experiment.log_metric('step_loss', loss.detach().cpu().item(), step=step)
                update_train_running_results(train_running_results, loss, images_in_batch)
                set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)

            train_epoch_loss = float(
                train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch'])
            experiment.log_metric('epoch_loss', train_epoch_loss, epoch=epoch)

            # Training CSV logging
            training_log_frame = pd.concat(
                [training_log_frame,
                 pd.DataFrame(data={'epoch': epoch, 'train_epoch_loss': train_epoch_loss}, index=[0])])
            training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if epoch % validation_frequency == 0:
            clip_model = clip_model.float()  # In validation we use fp32 CLIP model
            with experiment.validate():
                combiner.eval()

                # Compute and log validation metrics
                results = compute_cirr_val_metrics(relative_val_dataset, clip_model, val_index_features,
                                                   val_index_names, combiner.combine_features, prob_head, distance_fn)
                group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = results

                results_dict = {
                    'group_recall_at1': group_recall_at1,
                    'group_recall_at2': group_recall_at2,
                    'group_recall_at3': group_recall_at3,
                    'recall_at1': recall_at1,
                    'recall_at5': recall_at5,
                    'recall_at10': recall_at10,
                    'recall_at50': recall_at50,
                    'mean(R@5+R_s@1)': (group_recall_at1 + recall_at5) / 2,
                    'arithmetic_mean': mean(results),
                    'harmonic_mean': harmonic_mean(results),
                    'geometric_mean': geometric_mean(results)
                }

                print(json.dumps(results_dict))
                experiment.log_metrics(
                    results_dict,
                    epoch=epoch
                )

                # Validation CSV logging
                log_dict = {'epoch': epoch}
                log_dict.update(results_dict)
                validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
                validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

                # Save model
                if save_training:
                    models_path = training_path / "saved_models"
                    if save_best and results_dict['arithmetic_mean'] > best_arithmetic:
                        best_arithmetic = results_dict['arithmetic_mean']
                        save_model('combiner_arithmetic', epoch, combiner, training_path)
                        if use_prob:
                            models_path.mkdir(exist_ok=True, parents=True)
                            save_prob_head(models_path / 'prob_head_arithmetic.pt', epoch, prob_head)
                    if save_best and results_dict['harmonic_mean'] > best_harmonic:
                        best_harmonic = results_dict['harmonic_mean']
                        save_model('combiner_harmonic', epoch, combiner, training_path)
                        if use_prob:
                            models_path.mkdir(exist_ok=True, parents=True)
                            save_prob_head(models_path / 'prob_head_harmonic.pt', epoch, prob_head)
                    if save_best and results_dict['geometric_mean'] > best_geometric:
                        best_geometric = results_dict['geometric_mean']
                        save_model('combiner_geometric', epoch, combiner, training_path)
                        if use_prob:
                            models_path.mkdir(exist_ok=True, parents=True)
                            save_prob_head(models_path / 'prob_head_geometric.pt', epoch, prob_head)
                    if not save_best:
                        save_model(f'combiner_{epoch}', epoch, combiner, training_path)
                        if use_prob:
                            models_path.mkdir(exist_ok=True, parents=True)
                            save_prob_head(models_path / f'prob_head_{epoch}.pt', epoch, prob_head)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")
    parser.add_argument("--api-key", type=str, help="api for Comet logging")
    parser.add_argument("--workspace", type=str, help="workspace of Comet logging")
    parser.add_argument("--experiment-name", type=str, help="name of the experiment on Comet")
    parser.add_argument("--projection-dim", default=640 * 4, type=int, help='Combiner projection dim')
    parser.add_argument("--hidden-dim", default=640 * 8, type=int, help="Combiner hidden dim")
    parser.add_argument("--num-epochs", default=300, type=int, help="number training epochs")
    parser.add_argument("--clip-model-name", default="RN50x4", type=str, help="CLIP model to use, e.g 'RN50', 'RN50x4'")
    parser.add_argument("--clip-model-path", type=str, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--combiner-lr", default=2e-5, type=float, help="Combiner learning rate")
    parser.add_argument("--batch-size", default=1024, type=int, help="Batch size of the Combiner training")
    parser.add_argument("--clip-bs", default=32, type=int, help="Batch size during CLIP feature extraction")
    parser.add_argument("--validation-frequency", default=3, type=int, help="Validation frequency expressed in epochs")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    parser.add_argument("--save-training", dest="save_training", action='store_true',
                        help="Whether save the training model")
    parser.add_argument("--save-best", dest="save_best", action='store_true',
                        help="Save only the best model during training")
    parser.add_argument("--fiq-dress-type", default="all", type=str,
                        help="FashionIQ dress type: all, dress, toptee, or shirt")
    parser.add_argument("--relevance", default="dot", type=str, choices=['dot', 'free_energy', 'cope'],
                        help="Relevance measure: 'dot' (baseline), 'free_energy' (KL) or 'cope' (CoPE distance)")
    parser.add_argument("--prob-lr", default=1e-4, type=float,
                        help="Learning rate for the variance heads when --relevance free_energy")
    parser.add_argument("--free-energy-scale", default=10.0, type=float,
                        help="Scale applied to the negative free energy before the cross-entropy loss")

    args = parser.parse_args()
    if args.dataset.lower() not in ['fashioniq', 'cirr']:
        raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ")

    training_hyper_params = {
        "projection_dim": args.projection_dim,
        "hidden_dim": args.hidden_dim,
        "num_epochs": args.num_epochs,
        "clip_model_name": args.clip_model_name,
        "clip_model_path": args.clip_model_path,
        "combiner_lr": args.combiner_lr,
        "batch_size": args.batch_size,
        "clip_bs": args.clip_bs,
        "validation_frequency": args.validation_frequency,
        "transform": args.transform,
        "target_ratio": args.target_ratio,
        "save_training": args.save_training,
        "save_best": args.save_best,
        "relevance": args.relevance,
        "prob_lr": args.prob_lr,
        "free_energy_scale": args.free_energy_scale,
    }

    if args.api_key and args.workspace:
        print("Comet logging ENABLED")
        experiment = Experiment(
            api_key=args.api_key,
            project_name=f"{args.dataset} combiner training",
            workspace=args.workspace,
            disabled=False
        )
        if args.experiment_name:
            experiment.set_name(args.experiment_name)
    else:
        print("Comet loging DISABLED, in order to enable it you need to provide an api key and a workspace")
        experiment = Experiment(
            api_key="",
            project_name="",
            workspace="",
            disabled=True
        )

    experiment.log_code(folder=str(base_path / 'src'))
    experiment.log_parameters(training_hyper_params)

    if args.dataset.lower() == 'cirr':
        combiner_training_cirr(**training_hyper_params)
    elif args.dataset.lower() == 'fashioniq':
        if args.fiq_dress_type == 'all':
            dress_types = ['dress', 'toptee', 'shirt']
        else:
            dress_types = [args.fiq_dress_type]
        training_hyper_params.update(
            {'train_dress_types': dress_types, 'val_dress_types': dress_types})
        combiner_training_fiq(**training_hyper_params)
