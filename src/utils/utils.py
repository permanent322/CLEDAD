import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import os
import random
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import torch.nn.functional as F
from sklearn import metrics
import time



from .metrics import calc_quantile_CRPS_sum, calc_quantile_CRPS, save_roc_curve

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    
def gsutil_cp(src_path: str, dst_path: str):
    exec_result = os.system(f"gsutil cp -R {src_path} {dst_path}")
    if exec_result != 0:
        error_msg = f"gsutil_cp: Failed to copy file from {src_path} to {dst_path}"
        raise OSError(error_msg)
    else:
        print(f"gsutil_cp: copied file from {src_path} to {dst_path}")
    return exec_result, src_path, dst_path


# Function to wait for a file to exist
def wait_for_file(file_path, timeout=60):
    start_time = time.time()
    while not os.path.exists(file_path):
        time.sleep(1)  # Wait for 1 second before checking again
        if time.time() - start_time > timeout:
            raise TimeoutError(f"File {file_path} not found after {timeout} seconds.")
    print(f"File {file_path} found. Continuing script...")
    
def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred

def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    test_loader=None,
    valid_epoch_interval=5,   # 每个多少个训练epoch进行一次验证
    eval_epoch_interval=500,  # 每个多少个训练epoch进行一次评估
    foldername="",
    mode = 'pretraining',
    scaler=0,
    mean_scaler=1,
    nsample=100,
    save_samples = False,
    physionet_classification=False,

    normalize_for_ad=False
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-3)   # 原来是1e-6
    if foldername != "":
        output_path = foldername + "/model.pth"
        loss_path = foldername + "/losses.txt"
    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch,task=mode, normalize_for_ad=normalize_for_ad)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            if foldername != "":
                ## Save Losses in txt File
                with open(loss_path, "a") as file:
                    file.write('avg_epoch_loss: '+ str(avg_loss / batch_no) + ", epoch= "+ str(epoch_no) + "\n")
            lr_scheduler.step()
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0, normalize_for_ad=normalize_for_ad)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                with open(loss_path, "a") as file:
                    file.write('best loss is updated to: '+ str(avg_loss_valid / batch_no) + "at epoch= "+ str(epoch_no) + "\n")
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )
        
        if mode == 'pretraining' and test_loader is not None and (epoch_no + 1) % eval_epoch_interval == 0:
            current_checkpoint = (epoch_no + 1) // eval_epoch_interval
            previous_checkpoint_path = foldername + "checkpoint_"+str(current_checkpoint-1)+"/model.pth"
            checkpoint_folder = foldername + "checkpoint_"+str(current_checkpoint)
            os.makedirs(checkpoint_folder, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_folder+"/model.pth")
            if os.path.exists(previous_checkpoint_path):
                os.remove(previous_checkpoint_path)
                print(f"Checkpoint '{previous_checkpoint_path}' has been deleted.")
            else:
                print(f"No checkpoint found at '{previous_checkpoint_path}'.")
            model.eval()
            
            evaluate(model, test_loader, nsample=nsample, scaler=scaler, mean_scaler=mean_scaler, foldername=checkpoint_folder, save_samples = save_samples, physionet_classification=physionet_classification, normalize_for_ad=normalize_for_ad)

    if foldername != "":
        torch.save(model.state_dict(), output_path)

def finetune(model,
    config,
    train_loader,
    criterion,
    foldername="",
    task = 'classification',
    normalize_for_ad=False):
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"
        loss_path = foldername + "/losses.txt"
    ### Include loss in train_finetuning
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                outputs, loss = model.forward_finetuning(batch=train_batch, criterion=criterion, task=task, normalize_for_ad=normalize_for_ad)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            ## Save Losses in txt File
            with open(loss_path, "a") as file:
                file.write('avg_epoch_loss: '+ str(avg_loss / batch_no) + ", epoch= "+ str(epoch_no) + "\n")

    if foldername != "":
        torch.save(model.state_dict(), output_path)
    



def evaluate(model, test_loader, nsample=20, scaler=1, mean_scaler=0, foldername="", save_samples = False, physionet_classification=False, set_type='Train', normalize_for_ad=False):
    
    loss_path = foldername + "/losses.txt"
    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        all_labels = []
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(batch=test_batch, n_samples=nsample, normalize_for_ad=normalize_for_ad)

                if physionet_classification:
                    samples, c_target, eval_points, observed_points, observed_time, labels = output
                else:
                    samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)
                
                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)
                if physionet_classification:
                    all_labels.extend(labels.tolist())

                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points) 
                ) * scaler

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )
            all_target = torch.cat(all_target, dim=0)
            all_evalpoint = torch.cat(all_evalpoint, dim=0)
            all_observed_point = torch.cat(all_observed_point, dim=0)
            all_observed_time = torch.cat(all_observed_time, dim=0)
            all_generated_samples = torch.cat(all_generated_samples, dim=0)
            
            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )
            CRPS_sum = calc_quantile_CRPS_sum(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )
            print("RMSE:", np.sqrt(mse_total / evalpoints_total))
            print("MAE:", mae_total / evalpoints_total)
            print("CRPS:", CRPS)
            print("CRPS-sum:", CRPS_sum)
            print("MSE:", mse_total/evalpoints_total)

            with open(loss_path, "a") as file:
                file.write("RMSE:"+ str(np.sqrt(mse_total / evalpoints_total)) + "\n")
                file.write("MAE:"+ str(mae_total / evalpoints_total) + "\n")
                file.write("CRPS:"+ str(CRPS) + "\n")
                file.write("CRPS-sum:"+ str(CRPS_sum) + "\n")
                file.write("MSE:"+ str(mse_total/evalpoints_total) + "\n")
                
            print(len(all_labels))
            if save_samples and physionet_classification:
                with open(
                    foldername + f"/generated_outputs_{set_type}_nsample" + str(nsample) + ".pk", "wb"
                ) as f:
                    

                    pickle.dump(
                        [
                            all_generated_samples,
                            all_target,
                            all_evalpoint,
                            all_observed_point,
                            all_observed_time,
                            all_labels,
                            scaler,
                            mean_scaler,
                        ],
                        f,
                    )

            elif save_samples:
                with open(
                    foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
                ) as f:
                    

                    pickle.dump(
                        [
                            all_generated_samples,
                            all_target,
                            all_evalpoint,
                            all_observed_point,
                            all_observed_time,
                            scaler,
                            mean_scaler,
                        ],
                        f,
                    )
            with open(
                foldername + "/result_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        np.sqrt(mse_total / evalpoints_total),
                        mae_total / evalpoints_total,
                        CRPS,
                        CRPS_sum,
                        mse_total / evalpoints_total,
                    ],
                    f,
                )
'''
    evaluate_anomaly : to evaluate the anomaly detection model using the test_loader, find the threshold automatically, and evaluate the model
'''
def evaluate_anomaly(model, train_loader, test_loader, n_samples= 20, num_steps=50, foldername="", anomaly_ratio = 10, save_embeddings = False, normalize_for_ad=False, scaler=1):
    attens_energy = []     # energy of the test data, just reconstruction error
    train_energies = []    # energy of the train data, just reconstruction error
    test_labels = []
    all_correct = 0
    all_total = 0
    all_outputs = []

    with torch.no_grad():
        model.eval()


        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            # test data evaluation
            for batch_no, test_batch in enumerate(it, start=1):
                # outputs: samples  result : MSE between samples and target
                outputs, result = model.evaluate_finetuned_model(batch=test_batch,  n_samples=n_samples, normalize_for_ad=normalize_for_ad)


                attens_energy.append(result)
                test_labels.append(test_batch["label"])

            with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
                for batch_no, batch in enumerate(it, start=1):
                    # reconstruction
                    outputs, result = model.evaluate_finetuned_model(batch, n_samples=n_samples, normalize_for_ad=normalize_for_ad)
                    train_energies.append(result)
                    # print('Output shape', outputs.shape)
                    # print('Score shape', score.shape)
            train_energies = np.concatenate(train_energies, axis=0).reshape(-1)
            train_energy = np.array(train_energies)
                


            # (2) find the threshold
            attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            test_energy = np.array(attens_energy)
            combined_energy = np.concatenate([train_energy, test_energy], axis=0)

            threshold = np.percentile(combined_energy, 100 - anomaly_ratio)


            print("Threshold :", threshold)
            print("attens_energy :", attens_energy.shape)

            # (3) evaluation on the test set
            pred = (test_energy > threshold).astype(int)
            test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
            test_labels = np.array(test_labels)
            gt = test_labels.astype(int)

            print("pred:   ", pred.shape)
            print("gt:     ", gt.shape)

            # (4) detection adjustment
            gt, pred = adjustment(gt, pred)

            pred = np.array(pred)
            gt = np.array(gt)
            print("pred: ", pred.shape)
            print("gt:   ", gt.shape)

            accuracy = accuracy_score(gt, pred)
            precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary', zero_division=1)

            print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score))
            with open(foldername+'results.txt', "a") as file:
                file.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score))


def evaluate_anomaly_vote(model, train_loader, test_loader, n_samples=20, num_steps=50, result_folder="",foldername="", anomaly_ratio=1,
                     save_embeddings=False, normalize_for_ad=False, mean_scaler=0, scaler=1):
    attens_energy = []  # energy of the test data, just reconstruction error
    train_energies = []  # energy of the train data, just reconstruction error
    test_labels = []
    all_correct = 0
    all_total = 0
    all_outputs = []
    epoch_number = 0
    stop_number = -1
    split = 4
    name = ""
    with torch.no_grad():
        model.eval()

        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        all_middle = []

        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            # test data evaluation
            for batch_no, test_batch in enumerate(it, start=1):
                # outputs: samples  result : MSE between samples and target
                samples, result, c_target, eval_points, observed_points, observed_time, middle_result = model.evaluate_finetuned_model(batch=test_batch, n_samples=n_samples,
                                                                 normalize_for_ad=normalize_for_ad)

                attens_energy.append(result)
                test_labels.append(test_batch["label"])

                middle_result = middle_result.permute(0, 1, 3, 2)
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)  # (B,L,K)
                observed_points = observed_points.permute(0, 2, 1)  # (B,L,K)



                samples = c_target * (observed_points - eval_points) + samples.squeeze() * eval_points

                samples_length = samples.shape[1]
                # print(f"shape of samples after merging is {samples.shape}")
                #
                # print(f"shape of middle result before merging is {middle_result.shape}")
                #
                # print(f"shape of eval points is {eval_points.shape}")



                middle_result = middle_result.cpu() * eval_points.unsqueeze(
                    1).cpu() + c_target.unsqueeze(1).cpu() * (observed_points - eval_points).unsqueeze(1).cpu()

                # print("shape of middle result after merging")
                # print(middle_result.shape)
                # 保存头条带
                if batch_no == 1:
                    head = samples[0, 0: samples_length // split, :]
                    # print("shape of head is")
                    # print(head.shape)
                    head_c_target = c_target[0, 0: samples_length // split, :]
                    # print("shape of head c target is")
                    # print(head_c_target.shape)
                    head_observed_points = observed_points[0, 0: samples_length // split, :]
                    # print("shape of observed points is")
                    # print(head_observed_points.shape)
                    head_middle = middle_result[0, :, 0: samples_length // split, :]
                    # print("shape of head middle is")
                    # print(head_middle.shape)  # -> [100,25,25]

                samples = samples[:, samples_length // split: samples_length - samples_length // split, :]
                c_target = c_target[:, samples_length // split: samples_length - samples_length // split, :]
                observed_points = observed_points[:, samples_length // split: samples_length - samples_length // split,
                                  :]

                middle_result = middle_result[:, :, samples_length // split: samples_length - samples_length // split,
                                :]

                eval_points = torch.ones_like(samples)
                samples_median = samples

                # print("shape of samples is")
                # print(samples.shape)
                # print("shape of c target is")
                # print(c_target.shape)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)
                all_middle.append(middle_result)

                mse_current = (
                                      ((samples_median - c_target)) ** 2
                              ) * (scaler ** 2)
                mae_current = (
                                  torch.abs((samples_median - c_target))
                              ) * scaler

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

                # if np.sqrt(mse_total / evalpoints_total) > 20:
                #     residual = ((samples_median.values - c_target)) ** 2 * (scaler ** 2)
                #     print(residual)

            with open(
                    foldername + f"/{epoch_number}-generated_outputs_nsample" + str(
                        n_samples) + f"{name}_stop_number_{stop_number}.pk", "wb"
            ) as f:
                print("open file successfully.........")
                all_target = torch.cat(all_target, dim=0).to("cpu")
                all_evalpoint = torch.cat(all_evalpoint, dim=0).to("cpu")
                all_observed_point = torch.cat(all_observed_point, dim=0).to("cpu")
                all_observed_time = torch.cat(all_observed_time, dim=0).to("cpu")
                all_generated_samples = torch.cat(all_generated_samples, dim=0).to("cpu")
                all_middle = torch.cat(all_middle, dim=0).to("cpu")
                head = head.to("cpu")
                head_c_target = head_c_target.to("cpu")
                head_observed_points = head_observed_points.to("cpu")

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        head,
                        head_c_target,
                        head_observed_points,
                        scaler,
                        mean_scaler,
                        all_middle,
                        head_middle
                    ],
                    f,
                )



            with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
                for batch_no, batch in enumerate(it, start=1):
                    # reconstruction
                    samples, result, c_target, eval_points, observed_points, observed_time, middle_result = model.evaluate_finetuned_model(batch, n_samples=n_samples,
                                                                     normalize_for_ad=normalize_for_ad)
                    train_energies.append(result)
                    # print('Output shape', outputs.shape)
                    # print('Score shape', score.shape)
            train_energies = np.concatenate(train_energies, axis=0).reshape(-1)
            train_energy = np.array(train_energies)

            # (2) find the threshold
            attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            test_energy = np.array(attens_energy)
            combined_energy = np.concatenate([train_energy, test_energy], axis=0)
            print("anomaly_ratio : ", anomaly_ratio)
            threshold = np.percentile(combined_energy, 100 - anomaly_ratio)

            # threshold = 8
            print("Threshold :", threshold)
            print("attens_energy :", attens_energy.shape)

            # (3) evaluation on the test set
            pred = (test_energy > threshold).astype(int)
            test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
            test_labels = np.array(test_labels)
            gt = test_labels.astype(int)

            print("pred:   ", pred.shape)
            print("gt:     ", gt.shape)

            # (4) detection adjustment
            gt, pred = adjustment(gt, pred)

            pred = np.array(pred)
            gt = np.array(gt)
            print("pred: ", pred.shape)
            print("gt:   ", gt.shape)

            np.save(os.path.join(result_folder, "pred.npy"), pred)
            np.save(os.path.join(result_folder, "gt.npy"), gt)

            print("pred and gt have been saved as .npy files in 'folder' directory.")

            accuracy = accuracy_score(gt, pred)
            precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary',
                                                                                  zero_division=1)

            print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score))
            with open(result_folder + 'results.txt', "a") as file:
                file.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                    accuracy, precision,
                    recall, f_score))

            


    
