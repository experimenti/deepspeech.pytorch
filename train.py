import typing 
import json
import os
import time
import torch
import torch.nn as nn
from model import DeepSpeech 
from data.utils import reduce_tensor
from decoder import GreedyDecoder
from data.data_loader import SpectrogramDataset, BucketingSampler, AudioDataLoader
from torch.nn import CTCLoss as TorchCTCLoss

from typing import Dict
from tqdm import tqdm

labels = ["_", "'", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
          "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", " "]


supported_rnns: Dict[str, nn.RNNBase] =  {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU
}

def getOptions():

    options = { 'rnn': {}, 'train': {}, 'data': {} ,'spectrogram':{},'dataLoader':{},'optimizer':{}}

    options['rnn']['rnn_type'] = supported_rnns['gru']
    options['rnn']['labels'] = labels 

    options['train']['loss_function'] = 'pytorch' 
    options['train']['epochs'] = 20 
    options['train']['max_norm'] = 400 
    options['train']['model_path'] = 'models/deepspeech_final.pth' 
    options['train']['models_folder'] = 'models' 
    options['train']['learning_anneal'] = 1.1 

    audio_conf = dict(sample_rate=16000,
                      window_size=.02,
                      window_stride=.01,
                      window='hamming',
                      noise_dir=None,
                      noise_prob=.04,
                      noise_levels=(0.0, 0.5))

    options['audio_conf'] = audio_conf

    options['data']['train_manifest'] = 'data/train_manifest.csv'
    options['data']['val_manifest'] = 'data/val_manifest.csv'
    options['data']['audio_conf'] = audio_conf
    options['data']['augment'] = False 
    options['data']['batch_size'] = 16 
    options['data']['labels'] = labels 

    #TODO: options['spectrogram'] 
    options['spectrogram']['normalize']=True 

    options['dataLoader']['num_workers']=4

    options['optimizer']['lr']=3e-4
    options['optimizer']['momentum']=0.9

    return options


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def display_training_progress():

    opts = dict(title='training progress', ylabel='', xlabel='Epoch',
                legend=['Loss', 'WER', 'CER'])

def define_rnn(rnn_options, audio_conf):

    rnn = DeepSpeech(rnn_hidden_size=800,
                       nb_layers=5,
                       labels=rnn_options['labels'],
                       rnn_type=rnn_options['rnn_type'],
                       audio_conf=audio_conf,
                       bidirectional=True)

    parameters = rnn.parameters()

    return (rnn, parameters)


def load_test_train_data(audio_conf, train_manifest, val_manifest, labels, batch_size=20, augment=False,normalize=True,num_workers=4):


    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=train_manifest,
                                       labels=labels, normalize=normalize, augment=augment)

    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=val_manifest,
                                      labels=labels, normalize=normalize, augment=False)

    train_sampler = BucketingSampler(train_dataset, batch_size=batch_size)

    train_loader = AudioDataLoader(train_dataset,
                                   num_workers=num_workers, batch_sampler=train_sampler)
    test_loader = AudioDataLoader(test_dataset, batch_size=batch_size,
                                  num_workers=num_workers)


    return (train_loader, test_loader, train_sampler)

def train(rnn, rnn_parameters, train_loader, train_sampler, epochs=20, loss_function='pytorch', model_path='models/deepspeech_final.pth', models_folder='models', cuda=True, max_norm=400, lr=None, learning_anneal=1.1, momentum=None):

    avg_loss, start_epoch, start_iter = 0, 0, 0

    criterion = TorchCTCLoss(reduction='sum')

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    decoder = GreedyDecoder(labels)

    loss_results, cer_results, wer_results = torch.Tensor(epochs), torch.Tensor(epochs), torch.Tensor(epochs)

    parameters = rnn.parameters()

    optimizer = torch.optim.SGD(parameters, lr=lr,
                                momentum=momentum, nesterov=True)

    if (start_epoch != 0): 
        print("Shuffling batches for the following epochs")
        train_sampler.shuffle(start_epoch)

    if cuda:
        rnn.cuda()

    for epoch in range(epochs):

        #Flag the rnn that we are in training mode
        rnn.train()

        best_wer = None

        start_epoch_time = time.time()
        end = time.time()

        for i, (data) in enumerate(train_loader,  start=start_iter):
            if i == len(train_sampler):
                break

            inputs, targets, input_percentages, target_sizes = data
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

            data_time.update(time.time() - end)
          
            if cuda:
                inputs = inputs.cuda()

            out, output_sizes = rnn(inputs, input_sizes)
            out = out.transpose(0, 1)  

            #Softmax prior to loss calc for pytorch native ctc
            out = out.log_softmax(2).requires_grad_()

            loss = criterion(out, targets, output_sizes, target_sizes)
            loss = loss / inputs.size(0)

            inf = float("inf")
            loss_value = loss.item()

            if loss_value == inf or loss_value == -inf:
                print("WARNING: received an inf loss, setting loss value to 0")
                loss_value = 0

            avg_loss += loss_value
            losses.update(loss_value, inputs.size(0))

            # compute gradient
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(rnn.parameters(), 400)
            # SGD step
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format((epoch + 1), (i + 1), len(train_sampler), batch_time=batch_time, data_time=data_time, loss=losses))

            del loss
            del out

        avg_loss /= len(train_sampler)

        epoch_time = time.time() - start_epoch_time
        print('Training Summary Epoch: [{0}]\t'
              'Time taken (s): {epoch_time:.0f}\t'
              'Average Loss {loss:.3f}\t'.format(epoch + 1, epoch_time=epoch_time, loss=avg_loss))

        start_iter = 0  # Reset start iteration for next epoch
        total_cer, total_wer = 0, 0

        rnn.eval()

        with torch.no_grad():
            for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
                inputs, targets, input_percentages, target_sizes = data
                input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

                # unflatten targets
                split_targets = []
                offset = 0
                for size in target_sizes:
                    split_targets.append(targets[offset:offset + size])
                    offset += size

                if cuda:
                    inputs = inputs.cuda()

                out, output_sizes = rnn(inputs, input_sizes)

                decoded_output, _ = decoder.decode(out.data, output_sizes)
                target_strings = decoder.convert_to_strings(split_targets)
                wer, cer = 0, 0
                for x in range(len(target_strings)):
                    transcript, reference = decoded_output[x][0], target_strings[x][0]
                    wer += decoder.wer(transcript, reference) / float(len(reference.split()))
                    cer += decoder.cer(transcript, reference) / float(len(reference))
                total_cer += cer
                total_wer += wer
                del out
            wer = total_wer / len(test_loader.dataset)
            cer = total_cer / len(test_loader.dataset)
            wer *= 100
            cer *= 100
            loss_results[epoch] = avg_loss
            wer_results[epoch] = wer
            cer_results[epoch] = cer
            print('Validation Summary Epoch: [{0}]\t'
                    'Average WER {wer:.3f}\t'
                    'Average CER {cer:.3f}\t'.format(epoch + 1, wer=wer, cer=cer))

            file_path = '%s/deepspeech_%d.pth' % (models_folder, epoch + 1)
            torch.save(DeepSpeech.serialize(rnn, optimizer=optimizer, epoch=epoch, loss_results=loss_results,
                                            wer_results=wer_results, cer_results=cer_results),
                            file_path)
            # anneal lr
            optim_state = optimizer.state_dict()
            # TODO Param anneal rate
            optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / learning_anneal 
            optimizer.load_state_dict(optim_state)
            print('Learning rate annealed to: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))

        if (best_wer is None or best_wer > wer):
            # TODO: rnns path
            print("Found better validated rnn, saving to %s" % model_path)
            torch.save(DeepSpeech.serialize(rnn, optimizer=optimizer, epoch=epoch, loss_results=loss_results,
                                            wer_results=wer_results, cer_results=cer_results), model_path)
            best_wer = wer

            avg_loss = 0
        
        print("Shuffling batches...")
        train_sampler.shuffle(epoch)

if __name__ == '__main__':

    options = getOptions()

    torch.manual_seed(123456)
    torch.cuda.manual_seed_all(123456)

    options['train']['cuda'] = torch.cuda.is_available() 

    rnn, parameters = define_rnn(options['rnn'], options['audio_conf'])

    if(torch.cuda.is_available()):
        rnn.cuda()

    print(rnn)

    train_loader, test_loader, train_sampler = load_test_train_data(**options['data'],**options['spectrogram'],**options['dataLoader']) 
    train(rnn, parameters, train_loader, train_sampler, **options['train'],**options['optimizer'])
