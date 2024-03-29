import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from losses.utils import cos_sim
import torch.nn.functional as F

class ListNetDistillLoss(nn.Module):
    """
    This loss expects as input a batch consisting of sentence pairs (a_1, p_1), (a_2, p_2)..., (a_n, p_n)
    where we assume that (a_i, p_i) are a positive pair and (a_i, p_j) for i!=j a negative pair.

    For each a_i, it uses all other p_j as negative samples, i.e., for a_i, we have 1 positive example (p_i) and
    n-1 negative examples (p_j). It then minimizes the negative log-likehood for softmax normalized scores.

    This loss function works great to train embeddings for retrieval setups where you have positive pairs (e.g. (query, relevant_doc))
    as it will sample in each batch n-1 negative docs randomly.

    The performance usually increases with increasing batch sizes.

    For more information, see: https://arxiv.org/pdf/1705.00652.pdf
    (Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4)

    You can also provide one or multiple hard negatives per anchor-positive pair by structering the data like this:
    (a_1, p_1, n_1), (a_2, p_2, n_2)

    Here, n_1 is a hard negative for (a_1, p_1). The loss will use for the pair (a_i, p_i) all p_j (j!=i) and all n_j as negatives.

    Example::

        from sentence_transformers import SentenceTransformer, losses, InputExample
        from torch.utils.data import DataLoader

        model = SentenceTransformer('distilbert-base-uncased')
        train_examples = [InputExample(texts=['Anchor 1', 'Positive 1']),
            InputExample(texts=['Anchor 2', 'Positive 2'])]
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
        train_loss = losses.MultipleNegativesRankingLoss(model=model)
    """

    def __init__(
        self,
        similarity_fct=cos_sim,
        temperature=1.0
    ):
        """
        :param model: SentenceTransformer model
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        """
        super(ListNetDistillLoss, self).__init__()
        self.similarity_fct = similarity_fct
        self.temperature = temperature

    def _compute_scores(self, sentence_embeddings: Iterable[Tensor]):
        reps = sentence_embeddings
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])

        scores = self.similarity_fct(embeddings_a, embeddings_b)
        return scores

    def forward(
        self,
        student_sentence_embeddings: Iterable[Tensor],
        teacher_sentence_embeddings: Iterable[Tensor],
    ):

        teacher_scores = self._compute_scores(teacher_sentence_embeddings)
        student_scores = self._compute_scores(student_sentence_embeddings)
        teacher_scores_normalized = F.softmax(teacher_scores / self.temperature, dim=1)
        student_scores_normalized = F.softmax(student_scores / self.temperature, dim=1)
        student_scores_normalized_log = torch.log(student_scores_normalized + 1e-6)
        loss = torch.mean(-torch.sum(teacher_scores_normalized * student_scores_normalized_log, dim=1))
        return loss

    def get_config_dict(self):
        return {"similarity_fct": self.similarity_fct.__name__}


def _test():
    
    loss_fn = ListNetDistillLoss()
    teacher_embeddings = [torch.Tensor([[1,1],[2,2]]), torch.Tensor([[1,1],[2,2]]), torch.Tensor([[1,1],[2,2]])]
    student_embeddings = [torch.Tensor([[1,1],[2,2]]), torch.Tensor([[1,1],[2,2]]), torch.Tensor([[1,1],[2,2]])]
    
    loss = loss_fn(teacher_embeddings, student_embeddings)
    
if __name__ == "__main__":
    _test()