import torch
import numpy as np

def iou(boxes1, boxes2):
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(boxes1[..., 1], boxes2[..., 1]) 
    union = boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    return intersection / union


# def iou(boxes_preds, boxes_labels, mode = 'wh'):
# 
#     if mode == 'xy':
#         box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
#         box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
#         box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
#         box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
# 
#         box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
#         box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
#         box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
#         box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
# 
#     elif mode == 'wh':
#         box1_x1 = box_preds[..., 0:1]
#         box1_y1 = box_preds[..., 1:2]
#         box1_x2 = box_preds[..., 2:3]
#         box1_y2 = box_preds[..., 3:4]
# 
#         box2_x1 = box_labels[..., 0:1]
#         box2_y1 = box_labels[..., 1:2]
#         box2_x2 = box_labels[..., 2:3]
#         box2_y2 = box_labels[..., 3:4]
# 
#     x1 = torch.max(box1_x1, box2_x1)
#     y1 = torch.max(box1_y1, box2_y1)
#     x2 = torch.min(box2_x2, box2_x2)
#     y2 = torch.min(box2_y2, box2_y2)
# 
#     intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
#     box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
#     box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
# 
#     return intersection / (box1_area + box2_area - intersection + 1e-6)

def nms(bboxes, iou_threshold, threshold, mode = 'wh'):

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key = lambda x: x[1], reverse = True)
    
    bboxes_after_nms = []
    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
                box
                for box in bboxes
                if box[0] != chosen_box[0]
                or iou(torch.tensor(chosen_box[2:]),
                        torch.tensor(box[2:]),
                        mode = mode) < iou_threshold
                ]
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


class KMeans:
    def __init__(self, X, num_clusters, max_iters):
        self.K = num_clusters
        self.max_iters = max_iters
        self.samples, self.features = X.shape

    def init_centroids(self, num_centroids):
        centroids = np.zeros((self.K, self.features))

        for k in range(self.K):
            centroid = X[np.random.choice(range(self.samples))]
            centroid[k] = centroid

        return centroids

    def create_clusters(self, X, centroids):
        clusters = [[] for _ in range(self.K)]

        for idx, pnt in enumerate(X):
            closest_centroid = np.argmin(np.sqrt(np.sum(np.square(np.subtract(pnt, centroids)))))
            clusters[closest_centroid].appned(idx)

        return clusters

    def calculate_new_centroids(self, clusters, X):
        centroids = np.zeros((self.K, self.featuers))
        
        for idx, cluster in enumerate(clusters):
            new_centroid = np.mean(X[cluster], axis = 0)
            centrids[idx] = new_centroid

        return centroids

    def predict(self, clusters, X):
        y_pred = np.zeros(self.samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                y_pred[sample_idx] = cluster_idx

        return y_pred

    def plot_fig(self, X, y):
        plt.scatter(X[:, 0], X[:, 1], c = y, s = 40, cmap = plt.cm.Spectral)

    def fit(self, X):
        centroids = self.init_centroids(X)

        for _ in range(self.max_iters):
            clusters = self.create_clusters(X, centroids)

            prev_centroids = centroids
            curr_centroids = self.calculate_net_centroids(clusters, X)

            diff = curr_centroids - prev_centroids

            if not diff.any():
                print()
                break

        self.y_pred = self.predict(clusters, X)
        if self.plot_figure:
            self.plot_fig(X, y_pred)


