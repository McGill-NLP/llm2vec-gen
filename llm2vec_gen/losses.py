import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn import MSELoss
from peft.peft_model import PeftModel
from transformers import AutoModel
from transformers.utils import logging

logger = logging.get_logger(__name__)


def all_gather(tensor: torch.Tensor, with_grad: bool = False) -> torch.Tensor:
    """
    Gathers a tensor from each distributed rank into a list. Always retains gradients for the local rank's tensor,
    and optionally retains gradients for the gathered tensors if `with_grad` is True.

    Args:
        tensor (torch.Tensor): The tensor to gather from each rank.
        with_grad (bool, optional): If True, the local rank's tensor retains its gradients. Defaults to False.

    Returns:
        torch.Tensor: A tensor containing the gathered tensors from all ranks, concatenated along the first dimension.
        If torch.distributed is not available or not initialized, returns the original tensor.
    """

    if dist.is_available() and dist.is_initialized():
        if with_grad:
            gathered_tensors = torch.distributed.nn.all_gather(tensor)
        else:
            world_size = dist.get_world_size()
            gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]

            # Perform all_gather.
            dist.all_gather(gathered_tensors, tensor)

            # Replace local rank's tensor with the original (retaining gradients).
            local_rank = dist.get_rank()
            gathered_tensors[local_rank] = tensor
        return torch.cat(gathered_tensors, dim=0)

    # Warn once about uninitialized or single-GPU usage.
    warning = (
        "Trying to gather while torch.distributed is not available or has not been initialized, "
        "returning the original (local) tensor. This is expected if you are "
        "only using one GPU; consider not using gathering to remove this warning."
    )
    logger.warning_once(warning)
    return tensor


def compute_margin_loss(positives, negatives, margin_threshold=None, gather_across_devices=False):
    batch_size = positives.shape[0]
    margin_squared = margin_threshold ** 2
    offset = 0

    def pairwise_sqdist(x, y):
        # returns (xN, yN) squared euclidean distances
        x2 = (x * x).sum(dim=1, keepdim=True)          # (xN, 1)
        y2 = (y * y).sum(dim=1, keepdim=True).t()      # (1, yN)
        d2 = x2 + y2 - 2.0 * (x @ y.t())
        return d2.clamp_min_(0.0)

    if gather_across_devices:
        all_positives = all_gather(positives, with_grad=True)
        all_negatives = all_gather(negatives, with_grad=True)

        rank = torch.distributed.get_rank()
        offset = rank * batch_size
    
        candidates = torch.cat([all_positives, all_negatives], dim=0)
    else:
        candidates = torch.cat([positives, negatives], dim=0)
    
    d2 = pairwise_sqdist(positives, candidates)  # (B, 2B)

    mask = torch.ones((batch_size, len(candidates)), dtype=torch.bool, device=positives.device)
    idx = torch.arange(batch_size, device=positives.device)
    pos_idx = torch.arange(offset, offset + batch_size, device=positives.device)
    mask[idx, pos_idx] = False

    neg_d2 = d2[mask].view(batch_size, len(candidates) - 1)
    loss = torch.nn.functional.relu(margin_squared - neg_d2)
    return loss.mean()


def compute_contrastive_loss(queries, positives, negatives, scale=1.0, function="dot_product", gather_across_devices=False, hardness_weighting_alpha=0.0):
    def normalize(x):
        return x / (x.norm(dim=-1, keepdim=True) + 1e-9)

    batch_size = queries.size(0)
    offset = 0

    if gather_across_devices:
        positives = all_gather(positives, with_grad=True)
        negatives = all_gather(negatives, with_grad=True)

        rank = torch.distributed.get_rank()
        offset = rank * batch_size
        
    candidates = torch.cat([positives, negatives], dim=0)  # (B+N, D)

    if function == "dot_product":
        logits = torch.matmul(queries, candidates.t()) * scale
    elif function == "cosine":
        logits = torch.matmul(normalize(queries), normalize(candidates).t()) * scale
    else:
        raise ValueError(f"Unknown contrastive function: {function}")

    if hardness_weighting_alpha == 0.0:
        labels = torch.arange(offset, offset + batch_size, device=positives.device)
        loss = F.cross_entropy(logits, labels)
        return loss

    # hardness weighting enabled
    all_pos_size = positives.size(0)
    idx = torch.arange(batch_size, device=positives.device)
    pos_idx = torch.arange(offset, offset + batch_size, device=positives.device)
    hard_neg_idx = torch.arange(offset + all_pos_size, offset + all_pos_size + batch_size, device=positives.device)
    with torch.no_grad():
        hard_neg_sim = logits[idx, hard_neg_idx] / scale # raw similarity
        w = torch.exp(hardness_weighting_alpha * hard_neg_sim.detach()) # (B, N)

    exp_logits = torch.exp(logits)
    w_mask = torch.ones_like(exp_logits)
    w_mask[idx, hard_neg_idx] = w
    exp_logits = exp_logits * w_mask
    sum_exp_logits = torch.sum(exp_logits, dim=-1)
    loss = -torch.log(exp_logits[idx, pos_idx] / (sum_exp_logits + 1e-9))
    
    return loss.mean()


def get_teacher_embeddings(input_ids, attention_mask, teacher_model, student_encoder_states=None):
    
    assert isinstance(teacher_model, PeftModel) or isinstance(teacher_model, AutoModel) or teacher_model.config.model_type == "xlm-roberta"
    teacher_model.to(student_encoder_states.device if student_encoder_states is not None else 'cuda')
    
    if not hasattr(teacher_model, "encode"):
        if teacher_model.config.model_type == "xlm-roberta":
            model_output = teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            teacher_encoder_states = model_output[0][:, 0]
        else:
            outputs = teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            attention_mask = attention_mask.unsqueeze(-1)
            embeddings = outputs.last_hidden_state * attention_mask
            sum_embeddings = torch.sum(embeddings, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            teacher_encoder_states = sum_embeddings / sum_mask
    else:
        teacher_encoder_states = teacher_model.encode(input_ids, attention_mask)
    teacher_encoder_states = teacher_encoder_states.to(student_encoder_states.dtype if student_encoder_states is not None else torch.float32)
    
    return teacher_encoder_states


def compute_mse_loss(q, d, mse_loss_weight=0.0):
    mse_loss, total = torch.tensor(0.0, device=q.device), torch.tensor(0.0, device=q.device)
    if mse_loss_weight > 0:
        loss_fnc = MSELoss(reduction="mean")
        mse_loss = loss_fnc(q, d)
        total += mse_loss_weight * mse_loss
    
    return total, mse_loss