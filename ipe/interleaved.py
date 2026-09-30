"""Attention, position, and target-index mechanics shared by IEPE and SPO."""
import torch


def attention_mask(attention_mask_2d, starts, ends, dtype, *, mask_reflection,
                   mode="full", k=64, p=0.5, include_bos=False, prefix_keep=None):
    batch_size, length = attention_mask_2d.shape
    device = attention_mask_2d.device
    minimum = torch.finfo(dtype).min
    causal = torch.triu(torch.full((length, length), minimum, device=device, dtype=dtype), diagonal=1)
    mask = causal[None, None].expand(batch_size, 1, -1, -1).clone()
    # Preserve IEPE's padding arithmetic and dtype behavior.
    mask = mask + (1.0-attention_mask_2d.float())[:, None, None, :] * minimum
    for i in range(batch_size):
        start, end = int(starts[i]), int(ends[i])
        if start < 0 or end < 0:
            continue
        if mask_reflection and end+1 < length:
            mask[i, 0, end+1:, start:end+1] = minimum
        if prefix_keep is not None:
            blocked = (~prefix_keep[i][:start]).nonzero(as_tuple=True)[0]
            mask[i, 0, start:end+1, blocked] = minimum
        elif mode == "last_k":
            cutoff = max(0, start-k)
            if cutoff > int(include_bos):
                mask[i, 0, start:end+1, int(include_bos):cutoff] = minimum
        elif mode == "random_p":
            first = int(include_bos)
            candidates = start-first
            if candidates > 0:
                keep = max(1, int(candidates*p))
                blocked = torch.randperm(candidates, device=device)[keep:]+first
                mask[i, 0, start:end+1, blocked] = minimum
    return mask


def position_ids(attention, starts, ends):
    batch_size, length = attention.shape
    positions = torch.arange(length, device=attention.device)
    result = positions[None].expand(batch_size, -1).clone()
    if starts is not None and ends is not None:
        valid = (starts >= 0) & (ends >= starts)
        suffix = valid[:, None] & (positions[None] > ends[:, None])
        result -= suffix.long() * (ends-starts+1)[:, None]
    return result.masked_fill(~attention.bool(), 0)


def prediction_layout(attention, starts, ends):
    """Return predictor indices and story/reflection masks in shifted coordinates."""
    batch_size, length = attention.shape
    positions = torch.arange(length-1, device=attention.device)
    predictors = positions[None].expand(batch_size, -1).clone()
    story = attention[:, 1:].bool().clone()
    reflection = torch.zeros_like(story)
    if starts is not None and ends is not None:
        for i in range(batch_size):
            start, end = int(starts[i]), int(ends[i])
            if start < 0 or end < 0:
                continue
            targets = positions+1
            reflection[i] = (targets > start) & (targets < end) & story[i]
            story[i] &= (targets < start) | (targets > end)
            if end+1 < length and attention[i, end+1]:
                if start == 0:
                    raise ValueError("Interleaved suffix prediction requires a pre-reflection token")
                predictors[i, end] = start-1
    return predictors, story, reflection
