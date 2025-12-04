import torch


def pairwise_distances(x, y):
    # x: (N, 3), y: (M, 3)
    # returns (N, M)
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    return torch.sqrt((diff ** 2).sum(-1) + 1e-8)


def precision_recall_f1(pred, gt, tau=0.03):
    """
    pred, gt: (N, 3) point clouds, torch tensors
    """
    dists = pairwise_distances(pred, gt)

    # precision: % of pred points that have a GT match within tau
    min_pred_gt = torch.min(dists, dim=1)[0]
    precision = (min_pred_gt < tau).float().mean()

    # recall: % of GT points that have a pred match within tau
    min_gt_pred = torch.min(dists, dim=0)[0]
    recall = (min_gt_pred < tau).float().mean()

    # f1
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = torch.tensor(0.0)

    return precision.item(), recall.item(), f1.item()
