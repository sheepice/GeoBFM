import hashlib

def frozen_digest(actor):
    digest = hashlib.sha256()
    for name, value in sorted(actor.named_parameters()):
        if not value.requires_grad:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
