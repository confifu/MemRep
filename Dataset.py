from torch.utils.data import Dataset, ConcatDataset
import pandas as pd
from random import randint as ri
import random
import cv2
import os
import glob
import numpy as np
from PIL import Image
from torchvision import transforms
import torch


def getRandomTransformParameter(high, mid, low, length = 64):
    retarr = []
    midpos = ri(length//4, length//2)
    highpos = ri(length//2, 3*length//4)
    
    retarr = list(np.linspace(start=low, stop=mid, num=midpos))
    retarr.extend(list(np.linspace(start=mid, stop=high, num=highpos-midpos)))
    retarr.extend(list(np.linspace(start=high, stop=mid, num=length - highpos)))
    
    retarr = np.array(retarr)
    retarr = retarr[::random.choice([-1, 1])]
    return retarr

def transform(frames, randomize = False, length=64):
    scaleParams = getRandomTransformParameter(0.9, 0.75, 0.5, length)
    zRotateParams = getRandomTransformParameter(45, 0, -45, length)
    
    h, w, _ = frames[0].shape
    horizTransParam = (h/4)*getRandomTransformParameter(0.4, 0.0, -0.4, length)
    verticalTransParam = (w/4)*getRandomTransformParameter(0.4, 0.0, -0.4, length)
    
    newFrames = []
    for i, frame in enumerate(frames):
        
        img = Image.fromarray(frame)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        ])
        frame = preprocess(img).unsqueeze(0)

        if randomize:
            frame = transforms.functional.affine(frame,
                                                zRotateParams[i],
                                                [horizTransParam[i], verticalTransParam[i]],
                                                scaleParams[i],
                                                [0.0, 0.0])
        newFrames.append(frame)
    return newFrames


'''
- Creates a combined dataset from countix, synthetic and blender datasets.
- Returns frames which are multiples of "framePerVid".
- First frame is always blank (zeros)
'''
class stackedDataset(Dataset):

    def __init__(self,
                dfPath = None,
                countixDir = None,
                countixPrefix = None,
                synthDir = None,
                synthSize = 1000,
                blenderVidDir = None,
                blenderAnnotDir = None,
                frame_h = 182,
                frame_w = 182,
                framePerVid=64):

        super().__init__()

         #=====================Countix Setup==========================
        self.df = None
        if countixDir is not None and countixPrefix is not None:
            df = pd.read_csv(dfPath)
            self.path_prefix = countixDir + '/' + countixPrefix

            files_present = []
            for i in range(0, len(df)):
                path_to_video = self.path_prefix + str(i) + '.mp4'
                #print(path_to_video)
                if os.path.exists(path_to_video):
                    files_present.append(i)

            self.df = df.iloc[files_present]

        #=====================Blender Setup==========================
        self.annotPrefix = blenderAnnotDir
        self.vidPrefix = blenderVidDir
        if blenderVidDir is None:
            self.blenderVideos = []
        else:
            self.blenderVideos =  list(glob.glob(blenderVidDir + '/*.mkv'))

        #=====================Synthetic Setup========================
        self.synthVidPath = synthDir
        self.synthSize = synthSize

        #=====================General Setup==========================
        self.framePerVid = framePerVid
        self.frame_h = frame_h
        self.frame_w = frame_w

    def getFrames(self, path = None, cap = None):
        if cap is None:
            cap = cv2.VideoCapture(path)

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False:
                break
            frame = cv2.resize(frame , (self.frame_h, self.frame_w), interpolation = cv2.INTER_AREA)
            frames.append(frame)
        cap.release()
        return frames       

    def padFrames(self, frames, padStart=0, padEnd=0, blankStart = True):
        '''
        returns 1 + padStart + frames + padEnd
        
        blankStart : Starts with a zero frame
        padStart, padEnd : pad random frames from a random video
        don't use blankStart = True and padStart >0
        
        '''

        blankFrame = np.zeros((self.frame_h, self.frame_w, 3), dtype = "uint8")

        if blankStart:
            retFrames = [blankFrame]
            assert padStart >= 1, str(padStart)
            padStart = padStart - 1
        else:
            retFrames = []

        randpath = random.choice(glob.glob(self.synthVidPath))
        randFrames = self.getFrames(path = randpath)

        newRandFrames = []
        for i in range(1, padStart + padEnd + 1):
            newRandFrames.append(randFrames[i * len(randFrames)//(padStart + padEnd)  - 1])

        #each of the events has 1/3 probability
        if not ri(0, 2):
            retFrames += [frames[0] for i in range(padStart)] + frames + [frames[-1] for i in range(padEnd)]
        elif ri(0, 1):
            retFrames += [blankFrame for i in range(padStart)] + frames + [blankFrame for i in range(padEnd)]
        else :
            retFrames += newRandFrames[:padStart] + frames + newRandFrames[padStart:]

        assert blankStart + padStart + len(frames) + padEnd == len(retFrames), str(blankStart + padStart + len(frames) + padEnd) + " " + str(len(retFrames))
        return retFrames


    def fillWithSynthVid(self):

        while True:
            path = random.choice(glob.glob(self.synthVidPath))
            assert os.path.exists(path), "No file with this pattern exist" + self.synthPath

            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total > 1:
                break
        
        totalFramesReq = self.numFramePerVid * self.framePerVid
        mirror = ri(0,1)
        clipDur = ri(15, min([30, total, (totalFramesReq-1)//(mirror + 1)]))
        halfPeriod = int(clipDur * (ri(1, 15)/15))
        period = halfPeriod * (mirror + 1)
        count = ri(max(1, ((totalFramesReq-1 - halfPeriod*mirror)//period) - 6),
                  max(1, (totalFramesReq-1 - halfPeriod*mirror)//period ))

        noRepDur = max(0, min(total - clipDur, totalFramesReq - 1 - period*count - halfPeriod*mirror, (period*count * ri(1, 10))//10))
        
        #if total - clipDur - noRepDur < 0:
        #    print(f"{total=} {totalFrames=} {mirror=} {clipDur=} {halfPeriod=} {period=} {count=} {noRepDur=}")
        

        startFrame = ri(0, total - (clipDur + noRepDur))
        cap.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
        
        frames = self.getFrames(cap = cap)
        frames = frames[:noRepDur+clipDur]

        begNoRepDur = ri(0, noRepDur)
        endNoRepDur = noRepDur - begNoRepDur

        finalFrames = [np.zeros((self.frame_h, self.frame_w, 3), dtype = "uint8")]
        periodLength = [0]
        
        finalFrames.extend(frames[:begNoRepDur])
        periodLength.extend([0 for i in range(begNoRepDur)])
        assert len(periodLength) == len(finalFrames), str(len(periodLength)) + " "+str(len(finalFrames))

        clipFrames = frames[begNoRepDur : begNoRepDur + clipDur]
        repFrames = []
        for i in range(1, halfPeriod + 1):
            repFrames.append(clipFrames[i * len(clipFrames)//halfPeriod  - 1])
        assert len(repFrames) == halfPeriod,  str(halfPeriod) + " "+str(len(repFrames))


        if mirror:
            repFrames.extend(repFrames[::-1])
        assert len(repFrames) == period,  str(period) + " "+str(len(repFrames))


        if count == 1:
            period = 0
    
        for i in range(count):
            finalFrames.extend(repFrames)
            periodLength.extend([period for i in range(len(repFrames))])
        assert len(periodLength) == len(finalFrames), str(len(periodLength)) + " "+str(len(finalFrames))

        if mirror and endNoRepDur != 0:
            finalFrames.extend(repFrames[:halfPeriod])
            periodLength.extend([0 for i in range(halfPeriod)])
        assert len(periodLength) == len(finalFrames), str(len(periodLength)) + " "+str(len(finalFrames))
        
        finalFrames.extend(frames[begNoRepDur + clipDur:])
        periodLength.extend([0 for i in range(len(frames[begNoRepDur + clipDur:]))])
        assert len(periodLength) == len(finalFrames), str(len(periodLength)) + " "+str(len(finalFrames))

        for i in range(totalFramesReq - len(finalFrames)):
            periodLength.append(0)

        finalFrames = self.padFrames(finalFrames, blankStart=False, padEnd=totalFramesReq-len(finalFrames))

        assert len(periodLength) == len(finalFrames), str(len(periodLength)) + " "+str(len(finalFrames))
        assert len(periodLength) == totalFramesReq, str(len(periodLength)) + " " + str(totalFramesReq)

        labelStack = periodLength
        frameStack = transform(finalFrames, randomize=True, length=len(finalFrames))

        assert len(labelStack) == len(frameStack), str(len(labelStack)) + " "+str(len(frameStack))
        assert len(labelStack) == totalFramesReq, str(len(labelStack)) + " "+str(totalFramesReq)
        return frameStack, labelStack

    def fillWithCountixVid(self, vidPath, count):

        frames = self.getFrames(path = vidPath)
        
        totalFrames = self.framePerVid *self.numFramePerVid
        output_len = min(len(frames), int((ri(7, 10)/10) * totalFrames), totalFrames - 2)
        
        newFrames = []
        for i in range(1, output_len + 1):
            newFrames.append(frames[i * len(frames)//output_len  - 1])

        assert len(newFrames) == output_len

        a = ri(1, totalFrames - output_len)
        b = totalFrames - output_len - a
        
        finalFrames = self.padFrames(frames = newFrames, padStart = a, padEnd = b, blankStart=True)

        periodLength = []
        period = output_len/count
        if period < 2:
            period = 0
        periodLength += [0 for i in range(a)]\
                        +[period for i in range(len(newFrames))]\
                        +[0 for i in range(b)]

        labelStack = periodLength
        frameStack = transform(finalFrames, length=len(finalFrames))

        assert len(labelStack) == len(frameStack), str(len(labelStack)) + " "+str(len(frameStack))
        assert len(labelStack) == totalFrames, str(len(labelStack)) + " "+str(totalFrames)
        return frameStack, labelStack

    def fillWithBlenderVid(self, vidPath, annotPath):

        finalFrames = self.getFrames(vidPath)[1:]

        labels = glob.glob(annotPath + '/*')
        periodLength = list(np.load(labels[0]))[1:]

        assert len(finalFrames)==len(periodLength)

        totalFrames = self.numFramePerVid * self.framePerVid
        assert totalFrames >= len(finalFrames) + 1        

        a = ri(1, totalFrames - len(finalFrames))
        b = totalFrames - len(finalFrames) - a

        finalFrames = self.padFrames(finalFrames, padStart = a, padEnd = b, blankStart=True)

  
        periodLength = [0 for i in range(a)]\
                        +periodLength\
                        +[0 for i in range(b)]

        labelStack = periodLength
        frameStack = transform(finalFrames, length=len(finalFrames))

        assert len(labelStack) == len(frameStack), str(len(labelStack)) + " "+str(len(frameStack))
        assert len(labelStack) == totalFrames, str(len(labelStack)) + " "+str(totalFrames)
        return frameStack, labelStack

    def fill(self, index):
        '''
        index [0, len(blenderVideos) => Blender dataset
        index [len(blenderVideos), len(df)) => Countix
        index [len(df), ....) => Synthetic
        '''
        isSynth = index >= len(self.blenderVideos) +len(self.df.index)
        isBlender = index <len(self.blenderVideos)

        if isSynth and self.synthVidPath is not None:
            #=============synth
            tries = 10
            for i in range(tries):
                try:
                    self.numFramePerVid = ri(1, 300//self.framePerVid)
                    return self.fillWithSynthVid()
                except ValueError as e:
                    print ('Value Error, can be handled')
                except Exception as e:
                    print("Other error.")
                    print(e)
                    break

        elif isBlender:
            vidPath = self.blenderVideos[index]
            annotPath = self.annotPrefix + '/' + vidPath[len(self.vidPrefix)+1 : -4]

            cap = cv2.VideoCapture(vidPath)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.numFramePerVid = 1 + (total//self.framePerVid)
            cap.release()

            return self.fillWithBlenderVid(vidPath, annotPath)

        else:
            #=============countix
            dfi = self.df.iloc[[index - len(self.blenderVideos)]]
            vidPath = self.path_prefix + str(dfi.index.item()) +'.mp4'
            df = dfi.reset_index()
            count = df.loc[0, 'count']
            cap = cv2.VideoCapture(vidPath)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            #print(total, self.count)
            if total/count < 2:
                self.numFramePerVid = 1
            else :
                lowerLimit = max(1, 1 + (5*count)//self.framePerVid)
                upperLimit = max(1, 1 + (total // self.framePerVid))
                self.numFramePerVid = ri(min(upperLimit, lowerLimit), upperLimit)
            cap.release()
            return self.fillWithCountixVid(vidPath, count)

    def __getitem__(self, index):
        X, y = self.fill(index)
        X = torch.cat(X)
        y = torch.FloatTensor(y).unsqueeze(-1)
        return X, y

    def __len__(self):
        return self.synthSize + len(self.blenderVideos) +len(self.df.index)
