import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist

class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.maxDisappeared = maxDisappeared
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.widthheight = {}
        self.classes = {}
    def register(self, centroid, wh, class_info):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.widthheight[self.nextObjectID] = wh
        self.classes[self.nextObjectID] = class_info
        self.nextObjectID += 1
    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.widthheight[objectID]
        if objectID in self.classes:
            del self.classes[objectID]
    def update(self, rects, class_infos):
        if not rects:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.maxDisappeared:
                    self.deregister(oid)
            return self.objects
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        whs = []
        for i, (x1, y1, x2, y2) in enumerate(rects):
            cX = (x1 + x2)//2
            cY = (y1 + y2)//2
            inputCentroids[i] = (cX, cY)
            whs.append((x2-x1, y2-y1))
        if not self.objects:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i], whs[i], class_infos[i])
        else:
            oIDs = list(self.objects)
            oCentroids = list(self.objects.values())
            D = dist.cdist(np.array(oCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedR = set(); usedC = set()
            for r,c in zip(rows, cols):
                if r in usedR or c in usedC:
                    continue
                oid = oIDs[r]
                self.objects[oid] = inputCentroids[c]
                self.widthheight[oid] = whs[c]
                self.classes[oid] = class_infos[c]
                self.disappeared[oid] = 0
                usedR.add(r); usedC.add(c)
            for r in set(range(D.shape[0]))-usedR:
                oid = oIDs[r]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.maxDisappeared:
                    self.deregister(oid)
            for c in set(range(D.shape[1]))-usedC:
                self.register(inputCentroids[c], whs[c], class_infos[c])
        return self.objects 