import torch
import torch.nn as nn


class NTXEntLossAEFoster(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXEntLossAEFoster, self).__init__()
        self.temperature = temperature
        self.cossim = nn.CosineSimilarity(dim=-1)
        self.xent = nn.CrossEntropyLoss()
        self.alignment = None
        self.uniformity = None

    def forward(self, batch_a, batch_b):
        sim11 = (
            self.cossim(batch_a.unsqueeze(-2), batch_a.unsqueeze(-3)) / self.temperature
        )
        sim22 = (
            self.cossim(batch_b.unsqueeze(-2), batch_b.unsqueeze(-3)) / self.temperature
        )
        sim12 = (
            self.cossim(batch_a.unsqueeze(-2), batch_b.unsqueeze(-3)) / self.temperature
        )

        d = sim12.shape[-1]
        sim11[..., range(d), range(d)] = float("-inf")
        sim22[..., range(d), range(d)] = float("-inf")

        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        raw_scores = torch.cat([raw_scores1, raw_scores2], dim=-2)
        targets = torch.arange(2 * d, dtype=torch.long, device=raw_scores.device)

        return self.xent(raw_scores, targets)


class NTXEntLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXEntLoss, self).__init__()
        self.temperature = temperature
        self.alignment = None
        self.uniformity = None

    def forward(self, batch_a, batch_b):
        # It's much faster to do this than to broadcast cosine_similarity
        normalised_batch_a = nn.functional.normalize(batch_a)
        normalised_batch_b = nn.functional.normalize(batch_b)

        cosim_aa = normalised_batch_a @ normalised_batch_a.T / self.temperature
        cosim_bb = normalised_batch_b @ normalised_batch_b.T / self.temperature
        cosim_ab = normalised_batch_a @ normalised_batch_b.T / self.temperature

        tempered_alignment = cosim_ab.diagonal().mean()

        # correct for the temperature adjusgment in the stored alignment metric
        # higher values indicate more alignment
        self.alignment = self.temperature * tempered_alignment

        # exclude self inner products from the log-sum-exp
        aa_mask = torch.eye(cosim_aa.size(0), dtype=torch.bool, device=cosim_aa.device)
        bb_mask = torch.eye(cosim_bb.size(0), dtype=torch.bool, device=cosim_bb.device)
        cosim_aa.masked_fill_(aa_mask, float("-inf"))
        cosim_bb.masked_fill_(bb_mask, float("-inf"))

        logsumexp_1 = torch.cat((cosim_ab, cosim_bb), dim=-1).logsumexp(-1).mean()
        logsumexp_2 = torch.cat((cosim_aa, cosim_ab.T), dim=-1).logsumexp(-1).mean()

        raw_uniformity = logsumexp_1 + logsumexp_2

        uniformity_batch_normalisation = torch.tensor(
            2 * cosim_ab.size(-1) + cosim_aa.size(-1) + cosim_bb.size(-1),
            device=raw_uniformity.device,
        )
        # higher values indicate more uniformity
        self.uniformity = -raw_uniformity / 2

        return -(tempered_alignment - raw_uniformity / 2)


class AlignmentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(AlignmentLoss, self).__init__()
        self.temperature = temperature

        self.alignment = None

    def forward(self, batch_a, batch_b):

        self.alignment = nn.functional.cosine_similarity(batch_a, batch_b).mean()

        return -self.alignment / self.temperature