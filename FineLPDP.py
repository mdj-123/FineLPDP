
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class FineLPDPModel(nn.Module):


    def __init__(self, line_dim, line_gru_dim, line_num_layers, dropout, device, num_classes=2):
        super().__init__()

        self.line_encoder = BiGRULineEncoder(
            line_dim, line_gru_dim, line_num_layers, dropout, device
        )

        self.file_fc = nn.Linear(line_gru_dim * 2, 1)
        self.line_fc = nn.Linear(line_gru_dim * 2, 1)


        self.prototypes = nn.Parameter(torch.randn(num_classes, line_gru_dim * 2))

    def forward(self, code_tensor, return_line_features=False):
        sent_lengths = [len(code) for code in code_tensor]
        code_tensor = pad_sequence(code_tensor, batch_first=True)
        file_features, line_output, _ = self.line_encoder(code_tensor, sent_lengths)
        file_results = self.file_fc(file_features)
        line_scores = self.line_fc(line_output).squeeze(-1)

        if return_line_features:

            return file_results, line_scores, sent_lengths, file_features, line_output
        else:

            return file_results, line_scores, sent_lengths, file_features


class BiGRULineEncoder(nn.Module):


    def __init__(self, input_dim, hidden_dim, num_layers, dropout, device):
        super().__init__()
        self.device = device

        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, code_tensor, sent_lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            code_tensor, sent_lengths, batch_first=True, enforce_sorted=False
        )
        output, _ = self.gru(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        line_features = self.dropout(output)

        mask = torch.arange(line_features.size(1), device=self.device).unsqueeze(0) < \
               torch.tensor(sent_lengths, device=self.device).unsqueeze(1)

        pooled = (line_features * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

        return pooled, line_features, sent_lengths
