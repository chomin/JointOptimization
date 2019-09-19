import torch.nn.functional as F
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)


def cross_entropy(output, target):
    return F.cross_entropy(output, target)


def mycriterion(outputs, soft_targets):
    # We introduce a prior probability distribution p, which is a distribution of classes among all training data.
    
    USE_CUDA = torch.cuda.is_available()
    p = torch.ones(10).cuda() / 10 if USE_CUDA else torch.ones(10) / 10

    probs = F.softmax(outputs, dim=1)
    avg_probs = torch.mean(probs, dim=0)

    L_c = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * soft_targets, dim=1))
    L_p = -torch.sum(torch.log(avg_probs) * p)
    L_e = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * probs, dim=1))

    loss = L_c + args.alpha * L_p + args.beta * L_e
    return probs, loss