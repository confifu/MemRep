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

from threading import Thread


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


'''Takes a vidPath, and creates a random length minidataset from it with fixed frames per vid.'''
class stackedDataset(Dataset):

    def __init__(self,
                vidPath = None,
                annotPath = None,
                df = None,
                synthVidPath = None,
                framePerVid = 32):

        super().__init__()
        self.vidPath = vidPath
        self.annotPath = annotPath
        self.synthVidPath = synthVidPath
        self.framePerVid = framePerVid

        self.frame_h = 182
        self.frame_w = 182
        self.frameStack = []
        self.labelStack = []

        if vidPath is None:
            self.length = ri(1, 5)
        elif self.annotPath is None:
            df = df.reset_index()
            self.count = df.loc[0, 'count']

            cap = cv2.VideoCapture(self.vidPath)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            #print(total, self.count)
            if total/self.count < 2:
                self.length = 1
            else :
                lowerLimit = max(1, 1 + (5*self.count)//self.framePerVid)
                upperLimit = max(1, 1 + (total // self.framePerVid))
                
                self.length = ri(min(upperLimit, lowerLimit), upperLimit)
            cap.release()
        else:
            cap = cv2.VideoCapture(self.vidPath)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.length = 1 + (total//self.framePerVid)            
            cap.release()

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

    def fillWithSynthVid(self):
        assert len(self.labelStack) == len(self.frameStack), str(len(self.labelStack)) + " "+str(len(self.frameStack))
        assert len(self.labelStack) == 0, str(len(self.labelStack))

        while True:
            path = random.choice(glob.glob(self.synthVidPath))
            assert os.path.exists(path), "No file with this pattern exist" + self.synthPath

            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total > 1:
                break
        
        totalFrames = self.length * self.framePerVid
        mirror = ri(0,1)
        clipDur = ri(15, min([30, total, (totalFrames-1)//(mirror + 1)]))
        halfPeriod = int(clipDur * (ri(1, 15)/15))
        period = halfPeriod * (mirror + 1)
        count = ri(max(1, ((totalFrames-1 - halfPeriod*mirror)//period) - 6),
                  max(1, (totalFrames-1 - halfPeriod*mirror)//period ))

        noRepDur = max(0, min(total - clipDur, totalFrames - 1 - period*count - halfPeriod*mirror, (period*count * ri(1, 10))//10))
        
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

        for i in range(totalFrames - len(finalFrames)):
            finalFrames.append(np.zeros((self.frame_h, self.frame_w, 3), dtype = "uint8"))
            periodLength.append(0)
        assert len(periodLength) == len(finalFrames), str(len(periodLength)) + " "+str(len(finalFrames))
        
        assert len(periodLength) == totalFrames, str(len(periodLength)) + " " + str(totalFrames)

        self.labelStack = periodLength
        self.frameStack = transform(finalFrames, randomize=True, length=len(finalFrames))

        assert len(self.labelStack) == len(self.frameStack), str(len(self.labelStack)) + " "+str(len(self.frame_stack))
        assert len(self.labelStack) == totalFrames, str(len(self.labelStack)) + " "+str(totalFrames)

    def fillWithCountixVid(self):

        frames = self.getFrames(path = self.vidPath)
        
        totalFrames = self.framePerVid *self.length
        output_len = min(len(frames), int((ri(7, 10)/10) * totalFrames), totalFrames - 2)
        
        newFrames = []
        for i in range(1, output_len + 1):
            newFrames.append(frames[i * len(frames)//output_len  - 1])

        assert len(newFrames) == output_len

        a = ri(0, totalFrames - output_len-1)
        b = totalFrames - output_len - a -1
        
        randpath = random.choice(glob.glob(self.synthVidPath))
        randFrames = self.getFrames(path = randpath)
        
        newRandFrames = []
        for i in range(1, a + b + 1):
            newRandFrames.append(randFrames[i * len(randFrames)//(a+b)  - 1])

        finalFrames = [np.zeros((self.frame_h, self.frame_w, 3), dtype = "uint8")]
        periodLength = [0]
        if ri(0, 1):
            finalFrames += [newFrames[0] for i in range(a)] + newFrames + [newFrames[-1] for i in range(b)]
           
        else:
            finalFrames += newRandFrames[:a] + newFrames + newRandFrames[a:]

        period = output_len/self.count
        if period < 2:
            period = 0
        periodLength += [0 for i in range(a)]\
                        +[period for i in range(len(newFrames))]\
                        +[0 for i in range(b)]

        self.labelStack = periodLength
        self.frameStack = transform(finalFrames, length=len(finalFrames))

        assert len(self.labelStack) == len(self.frameStack), str(len(self.labelStack)) + " "+str(len(self.frame_stack))
        assert len(self.labelStack) == totalFrames, str(len(self.labelStack)) + " "+str(totalFrames)

    def fillWithBlenderVid(self):

        finalFrames = self.getFrames(self.vidPath)
        finalFrames[0] = np.zeros((self.frame_h, self.frame_w, 3), dtype = "uint8")

        labels = glob.glob(self.annotPath + '/*')
        periodLength = list(np.load(labels[0]))
        periodLength[0] = 0
        for i in range(len(periodLength)):
            if periodLength[i] >= 32:
                periodLength[i] = 0
        
        totalFrames = self.length * self.framePerVid
        for i in range(totalFrames - len(finalFrames)):
            finalFrames.append(np.zeros((self.frame_h, self.frame_w, 3), dtype = "uint8"))
            periodLength.append(0)

        self.labelStack = periodLength
        self.frameStack = transform(finalFrames, length=len(finalFrames))

        assert len(self.labelStack) == len(self.frameStack), str(len(self.labelStack)) + " "+str(len(self.frame_stack))
        assert len(self.labelStack) == totalFrames, str(len(self.labelStack)) + " "+str(totalFrames)

    def fill(self):
        if self.vidPath is None:
            self.fillWithSynthVid()
        elif self.annotPath is None:
            self.fillWithCountixVid()
        else:
            self.fillWithBlenderVid()
    
    def __getitem__(self, index):
        if index == 0:
            self.fill()
        
        X = torch.cat( self.frameStack[0: self.framePerVid] )
        y = torch.FloatTensor( self.labelStack[0: self.framePerVid] ).unsqueeze(-1)

        del self.frameStack[0: self.framePerVid]
        del self.labelStack[0: self.framePerVid]

        if index == self.length - 1:
            assert len(self.frameStack) == 0
            assert len(self.labelStack) == 0

        return X, y

    def __len__(self):
        return self.length


def getCombinedDataset(dfPath = None,
                       countixDir = None,
                       countixPrefix = None,
                       synthDir = None,
                       blenderVidDir = None,
                       blenderAnnotDir = None,
                       frame_per_vid=64):


    miniDatasetList = []

    #countix
    if countixDir is not None and countixPrefix is not None:
        df = pd.read_csv(dfPath)
        path_prefix = countixDir + '/' + countixPrefix
        
        files_present = []
        for i in range(0, len(df)):
            path_to_video = path_prefix + str(i) + '.mp4'
            #print(path_to_video)
            if os.path.exists(path_to_video):
                files_present.append(i)

        df = df.iloc[files_present]

        for i in range(len(df)):
            dfi = df.iloc[[i]]
            vidPath = path_prefix + str(dfi.index.item()) +'.mp4'
            miniDatasetList.append(stackedDataset(vidPath = vidPath,
                                                df = dfi,
                                                synthVidPath = synthDir,
                                                framePerVid= frame_per_vid))
        
    #Blender
    if blenderVidDir is not None and blenderAnnotDir is not None:
        vidPrefix = blenderVidDir
        annotPrefix = blenderAnnotDir
        videos =  list(glob.glob(vidPrefix + '/*.mkv'))

        for i in range(len(videos)):
            vidPath = videos[i]
            annotPath = annotPrefix + '/' + vidPath[len(vidPrefix)+1 : -4]
        
            miniDatasetList.append(stackedDataset(vidPath = vidPath,
                                                annotPath = annotPath,
                                                synthVidPath = synthDir,
                                                framePerVid= frame_per_vid))
    
    #Synth
    for i in range(0, 4000):
        miniDatasetList.append(stackedDataset(synthVidPath = synthDir,
                                              framePerVid= frame_per_vid))
    
    random.shuffle(miniDatasetList)
    megaDataset = ConcatDataset(miniDatasetList)
    
    index = 0
    indices = []
    for i in range(len(miniDatasetList)):
        indices.append(index)
        index += len(miniDatasetList[i])
    
    #return miniDatasetList
    return megaDataset