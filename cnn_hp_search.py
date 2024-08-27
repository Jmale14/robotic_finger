from tensorboard.plugins.hparams import api as hp
import numpy as np
import datetime
import time
import random
import tensorflow as tf
import csv

from train_cnn import run_trial

random.seed(10)
tf.random.set_seed(1234)

folds2Test = 5
use_pca = True
HP_H_UNITS = hp.HParam('h_units', hp.Discrete([100]))
HP_FILTERS = hp.HParam('hp_filters', hp.Discrete([100]))
HP_KERNEL = hp.HParam('hp_kernel', hp.Discrete([3]))
HP_POOL = hp.HParam('hp_pool', hp.Discrete([5]))
HP_EPOCHS = hp.HParam('epochs', hp.Discrete([300]))
HP_BATCH = hp.HParam('batch', hp.Discrete([64]))
HP_LR = hp.HParam('lr', hp.Discrete([0.01]))
HP_L2_LAMBDA = hp.HParam('l2_lambda', hp.Discrete([0.01]))

startTimeStamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

TRIALS = len(HP_H_UNITS.domain.values) * len(HP_POOL.domain.values) *\
    len(HP_KERNEL.domain.values) * len(HP_EPOCHS.domain.values) * \
    len(HP_BATCH.domain.values) * len(HP_FILTERS.domain.values) * \
    len(HP_LR.domain.values) * len(HP_L2_LAMBDA.domain.values)

# Loop through architectures
session_num = 1
ave_duration = 0
dur_list = []
max_acc = 0
max_f1 = 0

for h_units in HP_H_UNITS.domain.values:
    for filters in HP_FILTERS.domain.values:
        for kernel in HP_KERNEL.domain.values:
            for pool in HP_POOL.domain.values:
                for epochs in HP_EPOCHS.domain.values:
                    for batch in HP_BATCH.domain.values:
                        for lr in HP_LR.domain.values:
                            for l2_lambda in HP_L2_LAMBDA.domain.values:
                                hparams = {
                                    "HP_H_UNITS": h_units,
                                    "HP_FILTERS": filters,
                                    "HP_POOL": pool,
                                    "HP_KERNEL": kernel,
                                    "HP_EPOCHS": epochs,
                                    "HP_BATCH": batch,
                                    "HP_LR": lr,
                                    "HP_L2_LAMBDA": l2_lambda,
                                }
                                
                                run_name = "run-%d" % session_num
                                print(f'--- Starting trial {session_num}/{TRIALS}: {run_name}')
                                print({h: hparams[h] for h in hparams})
                                start_t = time.time()

                                results = run_trial(hparams, folds2Test, use_pca)

                                hist = {'loss': [],
                                        'accuracy': [],
                                        'f1_score': [],
                                        'val_loss': [],
                                        'val_accuracy': [],
                                        'val_f1_score': []}
                                for metric in ['loss', 'val_loss', 'accuracy', 'val_accuracy', 'f1_score', 'val_f1_score']:
                                    for fold in range(folds2Test):
                                        hist[metric].append([f'{metric}_{fold+1}_trial{session_num}']+ results["hist"][fold].history[metric])
                                    
                                    hist[metric].append([f'average_trial{session_num}'] + list(np.mean([history.history[metric] for history in results["hist"]], axis=0)))
                                    hist[metric].append([f'std_trial{session_num}'] + list(np.std([history.history[metric] for history in results["hist"]], axis=0)))
                                    hist[metric].append([])
                                
                                hparam_hist = []
                                if session_num == 1:
                                    hparam_hist = [["trial"] + [hprm for hprm in hparams.keys()]]
                                hparam_hist.append([session_num] + [hprm for hprm in hparams.values()])

                                try:
                                    duration = time.time() - start_t
                                    dur_list.append(duration)
                                    ave_duration = np.mean(dur_list) / 60

                                    maxf1new = max(list(np.mean([history.history['val_f1_score'] for history in results["hist"]], axis=0)))
                                    if maxf1new > max_f1:
                                        max_f1 = maxf1new
                                    maxaccnew = max(list(np.mean([history.history['val_accuracy'] for history in results["hist"]], axis=0)))
                                    if maxaccnew > max_acc:
                                        max_acc = maxaccnew

                                    print(f'Completed in {round(duration)}s, estimated time remaining: '
                                            f'{((TRIALS - session_num) * ave_duration):.2f}min, '
                                            f'{((TRIALS - session_num) * ave_duration / 60):.2f}hr, '
                                            f'accuracy: {maxaccnew:.2f}, '
                                            f'f1: {maxf1new:.2f}, '
                                            f'max accuracy: {max_acc:.2f}, '
                                            f'max f1: {max_f1:.2f}')
                                except Exception as e:
                                    print(e)

                                session_num += 1


                                for m in ['loss', 'accuracy', 'f1_score']:
                                    name = f"train_hist_{m}_PCA{use_pca}_CNN_" + startTimeStamp
                                    with open(f'{name}.csv', 'a') as out:
                                        for row in hist[m]:
                                            for col in row:
                                                out.write('{0},'.format(col))
                                            out.write('\n')
                                        
                                        for row in hist["val_"+m]:
                                            for col in row:
                                                out.write('{0},'.format(col))
                                            out.write('\n')

                                name = f"trial_details_PCA{use_pca}_CNN_" + startTimeStamp
                                with open(f'{name}.csv', 'a') as out:
                                    write = csv.writer(out)
                                    write.writerows(hparam_hist)
                                
                                # name = f"lr_details_PCA{use_pca}_CNN_" + startTimeStamp
                                # with open(f'{name}.csv', 'a') as out:
                                #     write = csv.writer(out)
                                #     write.writerows(results['lr_hist'])
                                
                                del results, hist

print("Done")
