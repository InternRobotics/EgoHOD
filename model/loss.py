import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from ipdb import set_trace

def gather_hand_feature(hand_box,l_valid,r_valid,left_data,right_data,        
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False):
    if gather_with_grad:
        all_hand_box = torch.cat(torch.distributed.nn.all_gather(hand_box), dim=0)
        all_l_valid = torch.cat(torch.distributed.nn.all_gather(l_valid), dim=0)
    else:
        gathered_hand_box = [torch.zeros_like(hand_box) for _ in range(world_size)]
        gathered_l_valid = [torch.zeros_like(l_valid) for _ in range(world_size)]
        gathered_r_valid = [torch.zeros_like(r_valid) for _ in range(world_size)]
        gathered_left_data = [torch.zeros_like(left_data) for _ in range(world_size)]
        gathered_right_data = [torch.zeros_like(right_data) for _ in range(world_size)]
        dist.all_gather(gathered_hand_box, hand_box)
        dist.all_gather(gathered_l_valid, l_valid)
        dist.all_gather(gathered_r_valid, r_valid)
        dist.all_gather(gathered_left_data, left_data)
        dist.all_gather(gathered_right_data, right_data)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_hand_box[rank] = hand_box
            gathered_l_valid[rank] = l_valid
            gathered_r_valid[rank] = r_valid
            gathered_left_data[rank] = left_data
            gathered_right_data[rank] = right_data
        all_hand_box = torch.cat(gathered_hand_box, dim=0)
        all_l_valid = torch.cat(gathered_l_valid, dim=0)
        all_r_valid = torch.cat(gathered_r_valid, dim=0)
        all_left_data = torch.cat(gathered_left_data, dim=0)
        all_right_data = torch.cat(gathered_right_data, dim=0)
    return all_hand_box,all_l_valid,all_r_valid,all_left_data,all_right_data

def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features

def loss_boxes(outputs, targets, num_boxes):

    loss_bbox = F.l1_loss(outputs, targets, reduction='none')

    losses = F.l1_loss(outputs, targets, reduction='none').sum() / num_boxes

    # loss_giou = 1 - torch.diag(generalized_box_iou(
    #     box_cxcywh_to_xyxy(outputs),
    #     box_cxcywh_to_xyxy(targets)))
    # losses['loss_giou'] = loss_giou.sum() / num_boxes
    return losses

class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale,hand_box=None,l_valid=None,r_valid=None,
                left_data=None,right_data=None):
        device = image_features.device

        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            if hand_box is not None:
                all_hand_box,all_l_valid,all_r_valid,all_left_data,all_right_data = gather_hand_feature(
                    hand_box,l_valid,r_valid,left_data,right_data,self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
                
                
                all_left_hand = all_hand_box[all_l_valid>0][:,:4,:]
                all_left_hand = rearrange(all_left_hand,'b t c->b (t c)')
                all_left_target = all_left_data[all_l_valid>0]
                all_right_hand = all_hand_box[all_r_valid>0][:,4:,:] #B,4,4
                all_right_hand = rearrange(all_right_hand,'b t c->b (t c)')
                all_right_target = all_right_data[all_r_valid>0] #[B,16]   
                loss2 = loss_boxes(all_right_hand,all_right_target,all_right_hand.shape[0]*4)
                loss3 = loss_boxes(all_left_hand,all_left_target,all_left_hand.shape[0]*4)
                loss1 = (loss2 + loss3) / 2
            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        vlp_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2

        with torch.no_grad():
            pred = torch.argmax(logits_per_image, dim=-1)
            correct = pred.eq(labels).sum()
            acc = 100 * correct / logits_per_image.size(0)
        
        if hand_box is not None:
            total_loss = vlp_loss + loss1
            return {'vlp_loss': vlp_loss, 'clip_acc': acc,'box_loss':loss1,'loss':vlp_loss + loss1}
        else:
            return {'clip_acc': acc,'loss':vlp_loss}

class Multiview_Cliploss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        self.clip_loss = ClipLoss(local_loss=self.local_loss,
            gather_with_grad=self.gather_with_grad, cache_labels=self.cache_labels,
            rank=self.rank, world_size=self.world_size, use_horovod=self.use_horovod)

        self.multiview_loss = 'global'
        assert self.multiview_loss in ['global']
        
    def forward(self, image_features, text_features, logit_scale, ego_features, exo_features, multiview_logit_scale):
        vt_loss = self.clip_loss(image_features, text_features, logit_scale)
        # set_trace()
        if self.multiview_loss == 'global':
            ego_features = ego_features.mean(1)
            exo_features = exo_features.mean(1)
            vv_loss = self.clip_loss(ego_features, exo_features, multiview_logit_scale)
        else:
            vv_loss = {}

        print(vt_loss, vv_loss)
        set_trace()
        loss_dict = {
            'loss': vt_loss['loss'] + vv_loss['loss'],
            'v2t': vt_loss['loss'], 'vt_acc': vt_loss['clip_acc'],
            'v2v': vv_loss['loss'], 'vv_acc': vv_loss['clip_acc'],
        }   
        return loss_dict


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


class MaxMarginRankingLoss(nn.Module):

    def __init__(
        self,
        margin=0.2,
        fix_norm=True,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__()
        self.fix_norm = fix_norm
        self.margin = margin
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

    def forward(self, image_features, text_features, weight=None):
        # TODO: try gather_from_all in
        # https://github.com/facebookresearch/LaViLa/blob/main/lavila/models/distributed_utils.py
        # all_image_features = gather_from_all(image_features)
        # all_text_features = gather_from_all(text_features)
        all_image_features, all_text_features = gather_features(
            image_features, text_features,
            self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)


        x = sim_matrix(all_text_features, all_image_features)

        n = x.size()[0]

        x1 = torch.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(n, n)
        x1 = x1.contiguous().view(-1, 1)
        x1 = torch.cat((x1, x1), 0)

        x2 = x.view(-1, 1)
        x3 = x.transpose(0, 1).contiguous().view(-1, 1)

        x2 = torch.cat((x2, x3), 0)
        max_margin = F.relu(self.margin - (x1 - x2))

        if self.fix_norm:
            # remove the elements from the diagonal
            keep = torch.ones(x.shape) - torch.eye(x.shape[0])  # 128 x 128
            keep1 = keep.view(-1, 1)
            keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
            keep_idx = torch.nonzero(torch.cat((keep1, keep2), 0).flatten()).flatten()
            if x1.is_cuda:
                keep_idx = keep_idx.cuda()
            x1_ = torch.index_select(x1, dim=0, index=keep_idx)
            x2_ = torch.index_select(x2, dim=0, index=keep_idx)
            max_margin = F.relu(self.margin - (x1_ - x2_))

        return {
            'loss': max_margin.mean()
        }



class CaptionLoss(nn.Module):
    def __init__(self, pad_id=0, tokenizer=None):
        super().__init__()
        self.pad_id = pad_id
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id

    def forward(self, outputs):
        logits = outputs['text_tokens_logits']
        labels = outputs['labels']
        # loss = F.cross_entropy(logits, labels, ignore_index=self.pad_id)
        loss = F.cross_entropy(logits, labels, ignore_index=self.pad_id, reduction='none')

        # compute accuracy
        with torch.no_grad():
            correct = 0.
            total = 0.
            ppls = []
            for i in range(logits.size(0)):
                pred = torch.argmax(logits[i], dim=0)
                nopad = labels[i].ne(self.pad_id)
                correct += (pred.eq(labels[i]) & nopad).sum()
                total += nopad.sum()
                ppl = torch.exp(loss[i].sum() / nopad.sum())
                ppls.append(ppl)
                # TODO: for debug only
                # sep_pos = labels[i].tolist().index(self.tokenizer.tokenizer.sep_token_id)
                # if self.tokenizer is not None:
                #     print('{} {} {}'.format(
                #         i, self.tokenizer.tokenizer.convert_ids_to_tokens(pred[:sep_pos]),
                #         self.tokenizer.tokenizer.convert_ids_to_tokens(labels[i, :sep_pos]),
                #     ))
            acc = 100 * correct / (total + 1e-8)
        return {'loss': loss.mean(), 'caption_loss': loss.mean(), 'caption_acc': acc, 'ppl': torch.tensor(ppls).mean()}

