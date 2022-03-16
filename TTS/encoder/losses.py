import torch
import torch.nn.functional as F
from torch import nn


# adapted from https://github.com/cvqluu/GE2E-Loss
class GE2ELoss(nn.Module):
    def __init__(self, init_w=10.0, init_b=-5.0, loss_method="softmax"):
        """
        Implementation of the Generalized End-to-End loss defined in https://arxiv.org/abs/1710.10467 [1]
        Accepts an input of size (N, M, D)
            where N is the number of speakers in the batch,
            M is the number of utterances per speaker,
            and D is the dimensionality of the embedding vector (e.g. d-vector)
        Args:
            - init_w (float): defines the initial value of w in Equation (5) of [1]
            - init_b (float): definies the initial value of b in Equation (5) of [1]
        """
        super().__init__()
        # pylint: disable=E1102
        self.w = nn.Parameter(torch.tensor(init_w))
        # pylint: disable=E1102
        self.b = nn.Parameter(torch.tensor(init_b))
        self.loss_method = loss_method

        print(" > Initialized Generalized End-to-End loss")

        assert self.loss_method in ["softmax", "contrast"]

        if self.loss_method == "softmax":
            self.embed_loss = self.embed_loss_softmax
        if self.loss_method == "contrast":
            self.embed_loss = self.embed_loss_contrast

    # pylint: disable=R0201
    def calc_new_centroids(self, dvecs, centroids, spkr, utt):
        """
        Calculates the new centroids excluding the reference utterance
        """
        excl = torch.cat((dvecs[spkr, :utt], dvecs[spkr, utt + 1 :]))
        excl = torch.mean(excl, 0)
        new_centroids = []
        for i, centroid in enumerate(centroids):
            if i == spkr:
                new_centroids.append(excl)
            else:
                new_centroids.append(centroid)
        return torch.stack(new_centroids)

    def calc_cosine_sim(self, dvecs, centroids):
        """
        Make the cosine similarity matrix with dims (N,M,N)
        """
        cos_sim_matrix = []
        for spkr_idx, speaker in enumerate(dvecs):
            cs_row = []
            for utt_idx, utterance in enumerate(speaker):
                new_centroids = self.calc_new_centroids(dvecs, centroids, spkr_idx, utt_idx)
                # vector based cosine similarity for speed
                cs_row.append(
                    torch.clamp(
                        torch.mm(
                            utterance.unsqueeze(1).transpose(0, 1),
                            new_centroids.transpose(0, 1),
                        )
                        / (torch.norm(utterance) * torch.norm(new_centroids, dim=1)),
                        1e-6,
                    )
                )
            cs_row = torch.cat(cs_row, dim=0)
            cos_sim_matrix.append(cs_row)
        return torch.stack(cos_sim_matrix)

    # pylint: disable=R0201
    def embed_loss_softmax(self, dvecs, cos_sim_matrix):
        """
        Calculates the loss on each embedding $L(e_{ji})$ by taking softmax
        """
        N, M, _ = dvecs.shape
        L = []
        for j in range(N):
            L_row = []
            for i in range(M):
                L_row.append(-F.log_softmax(cos_sim_matrix[j, i], 0)[j])
            L_row = torch.stack(L_row)
            L.append(L_row)
        return torch.stack(L)

    # pylint: disable=R0201
    def embed_loss_contrast(self, dvecs, cos_sim_matrix):
        """
        Calculates the loss on each embedding $L(e_{ji})$ by contrast loss with closest centroid
        """
        N, M, _ = dvecs.shape
        L = []
        for j in range(N):
            L_row = []
            for i in range(M):
                centroids_sigmoids = torch.sigmoid(cos_sim_matrix[j, i])
                excl_centroids_sigmoids = torch.cat((centroids_sigmoids[:j], centroids_sigmoids[j + 1 :]))
                L_row.append(1.0 - torch.sigmoid(cos_sim_matrix[j, i, j]) + torch.max(excl_centroids_sigmoids))
            L_row = torch.stack(L_row)
            L.append(L_row)
        return torch.stack(L)

    def forward(self, x, _label=None):
        """
        Calculates the GE2E loss for an input of dimensions (num_speakers, num_utts_per_speaker, dvec_feats)
        """

        assert x.size()[1] >= 2

        centroids = torch.mean(x, 1)
        cos_sim_matrix = self.calc_cosine_sim(x, centroids)
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = self.w * cos_sim_matrix + self.b
        L = self.embed_loss(x, cos_sim_matrix)
        return L.mean()


# adapted from https://github.com/clovaai/voxceleb_trainer/blob/master/loss/angleproto.py
class AngleProtoLoss(nn.Module):
    """
    Implementation of the Angular Prototypical loss defined in https://arxiv.org/abs/2003.11982
        Accepts an input of size (N, M, D)
            where N is the number of speakers in the batch,
            M is the number of utterances per speaker,
            and D is the dimensionality of the embedding vector
        Args:
            - init_w (float): defines the initial value of w
            - init_b (float): definies the initial value of b
    """

    def __init__(self, init_w=10.0, init_b=-5.0):
        super().__init__()
        # pylint: disable=E1102
        self.w = nn.Parameter(torch.tensor(init_w))
        # pylint: disable=E1102
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion = torch.nn.CrossEntropyLoss()

        print(" > Initialized Angular Prototypical loss")

    def forward(self, x, _label=None):
        """
        Calculates the AngleProto loss for an input of dimensions (num_speakers, num_utts_per_speaker, dvec_feats)
        """

        assert x.size()[1] >= 2

        out_anchor = torch.mean(x[:, 1:, :], 1)
        out_positive = x[:, 0, :]
        num_speakers = out_anchor.size()[0]

        cos_sim_matrix = F.cosine_similarity(
            out_positive.unsqueeze(-1).expand(-1, -1, num_speakers),
            out_anchor.unsqueeze(-1).expand(-1, -1, num_speakers).transpose(0, 2),
        )
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        label = torch.arange(num_speakers).to(cos_sim_matrix.device)
        L = self.criterion(cos_sim_matrix, label)
        return L


class SoftmaxLoss(nn.Module):
    """
    Implementation of the Softmax loss as defined in https://arxiv.org/abs/2003.11982
        Args:
            - embedding_dim (float): speaker embedding dim
            - n_speakers (float): number of speakers
    """

    def __init__(self, embedding_dim, n_speakers):
        super().__init__()

        self.criterion = torch.nn.CrossEntropyLoss()
        self.fc = nn.Linear(embedding_dim, n_speakers)

        print("Initialised Softmax Loss")

    def forward(self, x, label=None):
        # reshape for compatibility
        x = x.reshape(-1, x.size()[-1])
        label = label.reshape(-1)

        x = self.fc(x)
        L = self.criterion(x, label)

        return L

    def inference(self, embedding):
        x = self.fc(embedding)
        activations = torch.nn.functional.softmax(x, dim=1).squeeze(0)
        class_id = torch.argmax(activations)
        return class_id


class SoftmaxAngleProtoLoss(nn.Module):
    """
    Implementation of the Softmax AnglePrototypical loss as defined in https://arxiv.org/abs/2009.14153
        Args:
            - embedding_dim (float): speaker embedding dim
            - n_speakers (float): number of speakers
            - init_w (float): defines the initial value of w
            - init_b (float): definies the initial value of b
    """

    def __init__(self, embedding_dim, n_speakers, init_w=10.0, init_b=-5.0):
        super().__init__()

        self.softmax = SoftmaxLoss(embedding_dim, n_speakers)
        self.angleproto = AngleProtoLoss(init_w, init_b)

        print("Initialised SoftmaxAnglePrototypical Loss")

    def forward(self, x, label=None):
        """
        Calculates the SoftmaxAnglePrototypical loss for an input of dimensions (num_speakers, num_utts_per_speaker, dvec_feats)
        """

        Lp = self.angleproto(x)

        Ls = self.softmax(x, label)

        return Ls + Lp
