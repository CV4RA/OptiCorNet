import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
from os.path import join
from os import remove
import h5py
from math import ceil

def train(opt, model, encoder_dim, device, dataset, criterion, optimizer,
          train_set, whole_train_set, whole_training_data_loader, epoch, writer):
    epoch_loss = 0
    startIter = 1

    if opt.cacheRefreshRate > 0:
        subsetN = ceil(len(train_set) / opt.cacheRefreshRate)
        subsetIdx = np.array_split(np.arange(len(train_set)), subsetN)
    else:
        subsetN = 1
        subsetIdx = [np.arange(len(train_set))]

    nBatches = (len(train_set) + opt.batchSize - 1) // opt.batchSize

    for subIter in range(subsetN):
        print('====> Building Cache')
        model.eval()
        with h5py.File(train_set.cache, mode='w') as h5:
            pool_size = encoder_dim
            if opt.pooling.lower() == 'dsdnet':
                pool_size = opt.outDims
            h5feat = h5.create_dataset("features", [len(whole_train_set), pool_size], dtype=np.float32)
            with torch.no_grad():
                for iteration, (input, indices) in tqdm(enumerate(whole_training_data_loader, 1),
                                                        total=len(whole_training_data_loader)-1, leave=False):
                    image_encoding = input.float().to(device)
                    seq_encoding = model.pool(image_encoding)
                    h5feat[indices.detach().numpy(), :] = seq_encoding.detach().cpu().numpy()
                    del input, image_encoding, seq_encoding

        sub_train_set = Subset(dataset=train_set, indices=subsetIdx[subIter])

        training_data_loader = DataLoader(dataset=sub_train_set,
                                          num_workers=opt.threads,
                                          batch_size=opt.batchSize, shuffle=True,
                                          collate_fn=dataset.collate_fn,
                                          pin_memory=not opt.nocuda)

        print('Allocated:', torch.cuda.memory_allocated())
        print('Cached:', torch.cuda.memory_reserved())

        model.train()
        for iteration, (query, positives, negatives1, negatives2,
                        negCounts1, negCounts2, indices) in tqdm(enumerate(training_data_loader, startIter),
                                                                 total=len(training_data_loader), leave=False):

            if query is None:
                continue
            # Build Quadruplet
            loss = 0
            B = query.shape[0]
            nNeg1 = torch.sum(negCounts1)
            nNeg2 = torch.sum(negCounts2)
            totalNeg = nNeg1 + nNeg2

            input = torch.cat([query, positives, negatives1, negatives2]).float().to(device)
            seq_encoding = model.pool(input)

            seqQ, seqP, seqN1, seqN2 = torch.split(
                seq_encoding,
                [B, B, nNeg1, nNeg2]
            )

            optimizer.zero_grad()

            # computing each query 
            for i, (negCount1, negCount2) in enumerate(zip(negCounts1, negCounts2)):
                for n in range(negCount1):
                    negIx = (torch.sum(negCounts1[:i]) + n).item()
                    loss += criterion(seqQ[i:i+1], seqP[i:i+1], seqN1[negIx:negIx+1])
                for n in range(negCount2):
                    negIx2 = (torch.sum(negCounts2[:i]) + n).item()
                    loss += criterion(seqQ[i:i+1], seqP[i:i+1], seqN2[negIx2:negIx2+1])

            loss /= totalNeg.float().to(device)
            loss.backward()
            optimizer.step()

            del input, seq_encoding, seqQ, seqP, seqN1, seqN2
            del query, positives, negatives1, negatives2

            batch_loss = loss.item()
            epoch_loss += batch_loss

            if iteration % 50 == 0 or nBatches <= 10:
                print(f"==> Epoch[{epoch}]({iteration}/{nBatches}): Loss: {batch_loss:.4f}", flush=True)
                writer.add_scalar('Train/Loss', batch_loss,
                                  ((epoch-1) * nBatches) + iteration)
                writer.add_scalar('Train/TotalNeg', totalNeg,
                                  ((epoch-1) * nBatches) + iteration)
                print('Allocated:', torch.cuda.memory_allocated())
                print('Cached:', torch.cuda.memory_reserved())

        startIter += len(training_data_loader)
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        remove(train_set.cache)

    avg_loss = epoch_loss / nBatches
    print(f"===> Epoch {epoch} Complete: Avg. Loss: {avg_loss:.4f}", flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)
