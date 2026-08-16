'Internal utilities for the iSCALE research workflow.'
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_add, scatter_mean

from config import DEFAULT_CONTACT_THRESHOLDS

from mamba.mamba_ssm.ops.triton.ssd_combined_with_state import mamba_chunk_scan_combined


class BlockContactPredictor(nn.Module):
    'Implementation of BlockContactPredictor.'

    def __init__(self, d_inner, num_classes=7, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes


        self.contact_distribution = nn.Sequential(
            nn.Linear(d_inner, d_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_inner, num_classes)
        )


        self.contact_intensity = nn.Sequential(
            nn.Linear(d_inner, d_inner // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_inner // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, block_states):
        'Forward.'
        contact_logits = self.contact_distribution(block_states)
        intensity = self.contact_intensity(block_states)
        return contact_logits, intensity


class StructureAwareSSDBlock(nn.Module):
    'Implementation of StructureAwareSSDBlock.'

    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            headdim=16,
            ngroups=1,
            chunk_size=32,
            num_contact_classes=7,
            dropout=0.1,
            use_contact_pred=True
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.headdim = headdim
        self.chunk_size = chunk_size
        self.nheads = self.d_inner // headdim
        self.ngroups = ngroups
        self.use_contact_pred = use_contact_pred


        assert self.d_inner % self.headdim == 0, "d_inner must be divisible by headdim"
        assert self.nheads % ngroups == 0, "nheads must be divisible by ngroups"


        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj)


        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
        )


        A_log = torch.randn(self.nheads) - 1.0
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.dt_bias = nn.Parameter(torch.zeros(self.nheads))
        self.dt_bias._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True


        self.norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)


        if use_contact_pred:
            self.contact_predictor = BlockContactPredictor(
                d_inner=self.d_inner,
                num_classes=num_contact_classes,
                dropout=dropout
            )

    def forward(self, x, mask=None):
        'Forward.'
        batch, seq_len, _ = x.shape


        if mask is not None:
            x = x * mask.unsqueeze(-1)


        zxbcdt = self.in_proj(x)


        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )


        xBC = xBC.transpose(1, 2)
        xBC = self.conv1d(xBC)[:, :, :seq_len]
        xBC = xBC.transpose(1, 2)
        xBC = F.silu(xBC)


        x_ssm, B, C = torch.split(
            xBC,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1
        )


        x_ssm = rearrange(x_ssm, "b l (h p) -> b l h p", p=self.headdim)
        B = rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)


        A = -torch.exp(self.A_log)


        out, final_states, states = mamba_chunk_scan_combined(
            x_ssm, dt, A, B, C,
            chunk_size=self.chunk_size,
            D=self.D,
            dt_bias=self.dt_bias,
            dt_softplus=True,
            return_final_states=True,
            return_states=True
        )


        out = rearrange(out, "b l h p -> b l (h p)")
        out = out * F.silu(z)


        out = self.norm(out)
        out = self.out_proj(out)


        contact_logits = None
        contact_intensities = None

        if self.use_contact_pred:

            chunk_states = states.mean(dim=3).view(batch, -1, self.d_inner)
            flat_states = chunk_states.reshape(-1, self.d_inner)
            contact_logits, contact_intensities = self.contact_predictor(flat_states)

        return out, contact_logits, contact_intensities, states


class BidirectionalSSDLayer(nn.Module):
    'Implementation of BidirectionalSSDLayer.'

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, headdim=16,
                 chunk_size=32, num_contact_classes=7, dropout=0.1, use_contact_pred=True):
        super().__init__()
        assert (d_model * expand) % headdim == 0, "d_model * expand must be divisible by headdim"


        self.forward_ssd = StructureAwareSSDBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            chunk_size=chunk_size,
            num_contact_classes=num_contact_classes,
            dropout=dropout,
            use_contact_pred=use_contact_pred
        )


        self.backward_ssd = StructureAwareSSDBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            chunk_size=chunk_size,
            num_contact_classes=num_contact_classes,
            dropout=dropout,
            use_contact_pred=use_contact_pred
        )


        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        'Forward.'

        forward_out, fwd_contacts, fwd_intensities, fwd_states = self.forward_ssd(x, mask)


        contacts = []
        intensities = []


        if fwd_contacts is not None:
            contacts.append(fwd_contacts)
        if fwd_intensities is not None:
            intensities.append(fwd_intensities)


        x_reversed = torch.flip(x, dims=[1])
        if mask is not None:
            mask_reversed = torch.flip(mask, dims=[1])
        else:
            mask_reversed = None

        backward_out, bwd_contacts, bwd_intensities, bwd_states = self.backward_ssd(x_reversed, mask_reversed)
        backward_out = torch.flip(backward_out, dims=[1])


        if bwd_contacts is not None:
            contacts.append(bwd_contacts)
        if bwd_intensities is not None:
            intensities.append(bwd_intensities)


        combined = torch.cat([forward_out, backward_out], dim=-1)
        gate = self.gate(combined)
        output = gate * forward_out + (1 - gate) * backward_out

        return output, contacts, intensities


class SSD_RNA_Interaction(nn.Module):
    'Implementation of SSD RNA Interaction.'

    def __init__(
            self,
            protein_channels,
            rna_channels,
            hidden_channels,
            out_channels=1,
            num_layers=3,
            d_state=16,
            d_conv=4,
            expand=2,
            headdim=16,
            chunk_size=32,
            num_contact_classes=7,
            contact_thresholds=None,
            dropout=0.1,
            aux_weight=0.1,
            **kwargs
    ):
        super().__init__()


        # if contact_thresholds is None:
        #     contact_thresholds = [8.0, 10.0, 15.0, 20.0, 30.0, 50.0]


        if contact_thresholds is None:
            contact_thresholds = DEFAULT_CONTACT_THRESHOLDS

        num_contact_classes = len(contact_thresholds) + 1

        self.hidden_dim = hidden_channels
        self.chunk_size = chunk_size
        self.num_contact_classes = num_contact_classes
        self.contact_thresholds = contact_thresholds
        self.aux_weight = aux_weight
        self.gmb_args = {
            'd_state': d_state,
            'd_conv': d_conv,
            'expand': expand,
            'headdim': headdim,
            'chunk_size': chunk_size
        }


        self.protein_type_embedding = nn.Parameter(torch.randn(1, 1, hidden_channels))
        self.rna_type_embedding = nn.Parameter(torch.randn(1, 1, hidden_channels))



        self.protein_encoder = nn.Sequential(
            nn.Linear(protein_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU()
        )



        self.rna_encoder = nn.Sequential(
            nn.Linear(rna_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU()
        )


        self.rna_ssd_layers = nn.ModuleList([
            BidirectionalSSDLayer(
                d_model=hidden_channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
                chunk_size=chunk_size,
                num_contact_classes=num_contact_classes,
                dropout=dropout,
                use_contact_pred=False
            ) for _ in range(num_layers)
        ])



        self.protein_ssd_layers = nn.ModuleList([
            BidirectionalSSDLayer(
                d_model=hidden_channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
                chunk_size=chunk_size,
                num_contact_classes=num_contact_classes,
                dropout=dropout,
                use_contact_pred=True
            ) for _ in range(num_layers)
        ])



        self.interaction_layer = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
            nn.LayerNorm(hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels)
        )


        self.ddg_predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.LayerNorm(hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, wild_data, mutant_data, rna_data=None):
        'Forward.'

        outputs = {}


        wild_x, wild_mask = to_dense_batch(wild_data.x, wild_data.batch)
        mutant_x, mutant_mask = to_dense_batch(mutant_data.x, mutant_data.batch)


        batch_size, wild_len, _ = wild_x.shape
        _, mutant_len, _ = mutant_x.shape


        wild_features = self.protein_encoder(wild_x) + self.protein_type_embedding
        mutant_features = self.protein_encoder(mutant_x) + self.protein_type_embedding


        wild_features = wild_features * wild_mask.unsqueeze(-1)
        mutant_features = mutant_features * mutant_mask.unsqueeze(-1)



        forward_combined = torch.cat([wild_features, mutant_features], dim=1)
        forward_combined_mask = torch.cat([wild_mask, mutant_mask], dim=1)

        backward_combined = torch.cat([mutant_features, wild_features], dim=1)
        backward_combined_mask = torch.cat([mutant_mask, wild_mask], dim=1)


        forward_features = forward_combined
        backward_features = backward_combined

        all_forward_contacts = []
        all_forward_intensities = []
        all_backward_contacts = []
        all_backward_intensities = []

        for layer in self.protein_ssd_layers:

            forward_features, contacts_f, intensities_f = layer(forward_features, forward_combined_mask)
            forward_features = forward_features * forward_combined_mask.unsqueeze(-1)

            if contacts_f:
                all_forward_contacts.extend(contacts_f)
            if intensities_f:
                all_forward_intensities.extend(intensities_f)


            backward_features, contacts_b, intensities_b = layer(backward_features, backward_combined_mask)
            backward_features = backward_features * backward_combined_mask.unsqueeze(-1)

            if contacts_b:
                all_backward_contacts.extend(contacts_b)
            if intensities_b:
                all_backward_intensities.extend(intensities_b)


        mutant_forward_features = forward_features[:, wild_len:, :]
        mutant_forward_mask = forward_combined_mask[:, wild_len:]

        wild_backward_features = backward_features[:, mutant_len:, :]
        wild_backward_mask = backward_combined_mask[:, mutant_len:]



        mutant_forward_pooled = self._safe_global_pooling(
            mutant_forward_features,
            mutant_forward_mask,
            mutant_data.batch
        )

        wild_backward_pooled = self._safe_global_pooling(
            wild_backward_features,
            wild_backward_mask,
            wild_data.batch
        )


        diff_features = mutant_forward_pooled - wild_backward_pooled
        ddg_diff = self.ddg_predictor(diff_features).squeeze(-1)


        outputs['ddg'] = ddg_diff
        outputs['forward_contacts'] = all_forward_contacts
        outputs['forward_intensities'] = all_forward_intensities
        outputs['backward_contacts'] = all_backward_contacts
        outputs['backward_intensities'] = all_backward_intensities


        if rna_data is not None:

            rna_x, rna_mask = to_dense_batch(rna_data.x, rna_data.batch)
            rna_features = self.rna_encoder(rna_x) + self.rna_type_embedding
            rna_features = rna_features * rna_mask.unsqueeze(-1)


            wild_encoded = wild_features
            all_wild_contacts = []
            all_wild_intensities = []

            for layer in self.protein_ssd_layers:
                wild_encoded, contacts, intensities = layer(wild_encoded, wild_mask)
                all_wild_contacts.extend(contacts if contacts else [])
                all_wild_intensities.extend(intensities if intensities else [])


            rna_encoded = rna_features
            for layer in self.rna_ssd_layers:
                rna_encoded, _, _ = layer(rna_encoded, rna_mask)


            outputs['wild_contacts'] = all_wild_contacts
            outputs['wild_intensities'] = all_wild_intensities

        return outputs

    def _safe_global_pooling(self, features, mask, batch_indices):
        'Safe global pooling.'

        sum_features = torch.sum(features * mask.unsqueeze(-1), dim=1)
        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = sum_features / mask_sum


        batch_size = batch_indices.max().item() + 1
        if pooled.size(0) != batch_size:

            result = torch.zeros(batch_size, self.hidden_dim, device=features.device)
            for b in range(pooled.size(0)):
                if b < batch_size:
                    result[b] = pooled[b]
            pooled = result

        return pooled

    def compute_loss(self, wild_data, mutant_data, rna_data, ddg_target):
        'Compute loss.'

        outputs = self.forward(wild_data, mutant_data, rna_data)


        ddg_loss = F.mse_loss(outputs['ddg'], ddg_target)
        loss_dict = {'ddg_loss': ddg_loss}


        aux_loss = 0

        if hasattr(wild_data, 'block_contact_dist') and hasattr(wild_data, 'block_contact_int'):
            try:

                gt_dist = wild_data.block_contact_dist.to(ddg_loss.device)
                gt_int = wild_data.block_contact_int.to(ddg_loss.device)


                contact_loss = 0
                intensity_loss = 0
                valid_predictions = 0


                contact_source = (outputs.get('wild_contacts') or
                                  outputs.get('forward_contacts') or [])
                intensity_source = (outputs.get('wild_intensities') or
                                   outputs.get('forward_intensities') or [])


                for contacts, intensities in zip(contact_source, intensity_source):

                    if contacts.size(-1) != gt_dist.size(1):
                        continue


                    min_blocks = min(contacts.size(0), gt_dist.size(0))


                    dist_loss = F.kl_div(
                        F.log_softmax(contacts[:min_blocks], dim=-1),
                        gt_dist[:min_blocks],
                        reduction='batchmean'
                    )


                    int_loss = F.binary_cross_entropy(
                        intensities[:min_blocks],
                        gt_int[:min_blocks],
                        reduction='mean'
                    )


                    contact_loss += dist_loss
                    intensity_loss += int_loss
                    valid_predictions += 1


                if valid_predictions > 0:
                    contact_loss /= valid_predictions
                    intensity_loss /= valid_predictions


                    aux_loss = contact_loss + intensity_loss


                    loss_dict['contact_loss'] = contact_loss
                    loss_dict['intensity_loss'] = intensity_loss

            except Exception as e:
                print(f"Could not compute the auxiliary loss: {e}")


        total_loss = ddg_loss + self.aux_weight * aux_loss
        loss_dict['total_loss'] = total_loss
        loss_dict['aux_loss'] = aux_loss

        return total_loss, loss_dict


class SSD_RNA_Ablation(SSD_RNA_Interaction):
    'Implementation of SSD RNA Ablation.'

    def __init__(
            self,
            protein_channels,
            rna_channels,
            hidden_channels,
            out_channels=1,
            num_layers=3,
            d_state=16,
            d_conv=4,
            expand=2,
            headdim=16,
            chunk_size=32,
            num_contact_classes=7,
            contact_thresholds=None,
            dropout=0.1,
            aux_weight=0.1,
            **kwargs
    ):

        super().__init__(
            protein_channels=protein_channels,
            rna_channels=rna_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            chunk_size=chunk_size,
            num_contact_classes=num_contact_classes,
            contact_thresholds=contact_thresholds,
            dropout=dropout,
            aux_weight=aux_weight,
            **kwargs
        )
        print("Initializing the RNA-ablation variant; RNA inputs are ignored.")

    def forward(self, wild_data, mutant_data, rna_data=None):
        'Forward.'

        outputs = {}


        wild_x, wild_mask = to_dense_batch(wild_data.x, wild_data.batch)
        mutant_x, mutant_mask = to_dense_batch(mutant_data.x, mutant_data.batch)


        batch_size, wild_len, _ = wild_x.shape
        _, mutant_len, _ = mutant_x.shape


        wild_features = self.protein_encoder(wild_x) + self.protein_type_embedding
        mutant_features = self.protein_encoder(mutant_x) + self.protein_type_embedding


        wild_features = wild_features * wild_mask.unsqueeze(-1)
        mutant_features = mutant_features * mutant_mask.unsqueeze(-1)



        forward_combined = torch.cat([wild_features, mutant_features], dim=1)
        forward_combined_mask = torch.cat([wild_mask, mutant_mask], dim=1)

        backward_combined = torch.cat([mutant_features, wild_features], dim=1)
        backward_combined_mask = torch.cat([mutant_mask, wild_mask], dim=1)


        forward_features = forward_combined
        backward_features = backward_combined

        all_forward_contacts = []
        all_forward_intensities = []
        all_backward_contacts = []
        all_backward_intensities = []

        for layer in self.protein_ssd_layers:

            forward_features, contacts_f, intensities_f = layer(forward_features, forward_combined_mask)
            forward_features = forward_features * forward_combined_mask.unsqueeze(-1)

            if contacts_f:
                all_forward_contacts.extend(contacts_f)
            if intensities_f:
                all_forward_intensities.extend(intensities_f)


            backward_features, contacts_b, intensities_b = layer(backward_features, backward_combined_mask)
            backward_features = backward_features * backward_combined_mask.unsqueeze(-1)

            if contacts_b:
                all_backward_contacts.extend(contacts_b)
            if intensities_b:
                all_backward_intensities.extend(intensities_b)


        mutant_forward_features = forward_features[:, wild_len:, :]
        mutant_forward_mask = forward_combined_mask[:, wild_len:]

        wild_backward_features = backward_features[:, mutant_len:, :]
        wild_backward_mask = backward_combined_mask[:, mutant_len:]


        mutant_forward_pooled = self._safe_global_pooling(
            mutant_forward_features,
            mutant_forward_mask,
            mutant_data.batch
        )

        wild_backward_pooled = self._safe_global_pooling(
            wild_backward_features,
            wild_backward_mask,
            wild_data.batch
        )


        diff_features = mutant_forward_pooled - wild_backward_pooled
        ddg_diff = self.ddg_predictor(diff_features).squeeze(-1)


        outputs['ddg'] = ddg_diff
        outputs['wild_contacts'] = all_forward_contacts
        outputs['wild_intensities'] = all_forward_intensities
        outputs['forward_contacts'] = all_forward_contacts
        outputs['forward_intensities'] = all_forward_intensities
        outputs['backward_contacts'] = all_backward_contacts
        outputs['backward_intensities'] = all_backward_intensities

        return outputs


class ISCALE(SSD_RNA_Interaction):
    """iSCALE model used for the manuscript experiments."""

    def __init__(
            self,
            protein_channels,
            rna_channels,
            hidden_channels,
            out_channels=1,
            num_layers=3,
            d_state=32,  # State dimension used by the manuscript model
            d_conv=4,
            expand=2,
            headdim=16,
            chunk_size=32,
            num_contact_classes=7,
            contact_thresholds=None,
            dropout=0.1,
            aux_weight=0.2,  # Auxiliary-task weight used by the manuscript model
            **kwargs
    ):

        super().__init__(
            protein_channels=protein_channels,
            rna_channels=rna_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            chunk_size=chunk_size,
            num_contact_classes=num_contact_classes,
            contact_thresholds=contact_thresholds,
            dropout=dropout,
            aux_weight=aux_weight,
            **kwargs
        )
        self.chunk_size = chunk_size

    def forward(self, wild_data, mutant_data, rna_data=None):
        'Forward.'

        outputs = {}


        wild_x, wild_mask = to_dense_batch(wild_data.x, wild_data.batch)
        mutant_x, mutant_mask = to_dense_batch(mutant_data.x, mutant_data.batch)


        batch_size, wild_len, _ = wild_x.shape
        _, mutant_len, _ = mutant_x.shape


        wild_features = self.protein_encoder(wild_x) + self.protein_type_embedding
        mutant_features = self.protein_encoder(mutant_x) + self.protein_type_embedding


        wild_features = wild_features * wild_mask.unsqueeze(-1)
        mutant_features = mutant_features * mutant_mask.unsqueeze(-1)



        wild_padded_len = ((wild_len + self.chunk_size - 1) // self.chunk_size) * self.chunk_size
        wild_padding = wild_padded_len - wild_len

        if wild_padding > 0:
            padding_features = torch.zeros(batch_size, wild_padding, self.hidden_dim,
                                           device=wild_features.device)
            padding_mask = torch.zeros(batch_size, wild_padding,
                                       device=wild_mask.device)

            wild_features_padded = torch.cat([wild_features, padding_features], dim=1)
            wild_mask_padded = torch.cat([wild_mask, padding_mask], dim=1)
        else:
            wild_features_padded = wild_features
            wild_mask_padded = wild_mask


        mutant_padded_len = ((mutant_len + self.chunk_size - 1) // self.chunk_size) * self.chunk_size
        mutant_padding = mutant_padded_len - mutant_len

        if mutant_padding > 0:
            padding_features = torch.zeros(batch_size, mutant_padding, self.hidden_dim,
                                           device=mutant_features.device)
            padding_mask = torch.zeros(batch_size, mutant_padding,
                                       device=mutant_mask.device)

            mutant_features_padded = torch.cat([mutant_features, padding_features], dim=1)
            mutant_mask_padded = torch.cat([mutant_mask, padding_mask], dim=1)
        else:
            mutant_features_padded = mutant_features
            mutant_mask_padded = mutant_mask


        forward_combined = torch.cat([wild_features_padded, mutant_features_padded], dim=1)
        forward_combined_mask = torch.cat([wild_mask_padded, mutant_mask_padded], dim=1)

        backward_combined = torch.cat([mutant_features_padded, wild_features_padded], dim=1)
        backward_combined_mask = torch.cat([mutant_mask_padded, wild_mask_padded], dim=1)


        wild_padded_len = wild_features_padded.size(1)
        mutant_padded_len = mutant_features_padded.size(1)


        forward_features = forward_combined
        backward_features = backward_combined

        all_forward_contacts = []
        all_forward_intensities = []
        all_backward_contacts = []
        all_backward_intensities = []



        wild_only_features = wild_features_padded
        wild_only_mask = wild_mask_padded
        wild_only_contacts = []
        wild_only_intensities = []


        for layer in self.protein_ssd_layers:

            forward_features, contacts_f, intensities_f = layer(forward_features, forward_combined_mask)
            forward_features = forward_features * forward_combined_mask.unsqueeze(-1)

            if contacts_f:
                all_forward_contacts.extend(contacts_f)
            if intensities_f:
                all_forward_intensities.extend(intensities_f)


            backward_features, contacts_b, intensities_b = layer(backward_features, backward_combined_mask)
            backward_features = backward_features * backward_combined_mask.unsqueeze(-1)

            if contacts_b:
                all_backward_contacts.extend(contacts_b)
            if intensities_b:
                all_backward_intensities.extend(intensities_b)


            wild_only_features, contacts_w, intensities_w = layer(wild_only_features, wild_only_mask)
            wild_only_features = wild_only_features * wild_only_mask.unsqueeze(-1)

            if contacts_w:
                wild_only_contacts.extend(contacts_w)
            if intensities_w:
                wild_only_intensities.extend(intensities_w)


        mutant_forward_features = forward_features[:, wild_padded_len:wild_padded_len + mutant_len, :]
        mutant_forward_mask = forward_combined_mask[:, wild_padded_len:wild_padded_len + mutant_len]

        wild_backward_features = backward_features[:, mutant_padded_len:mutant_padded_len + wild_len, :]
        wild_backward_mask = backward_combined_mask[:, mutant_padded_len:mutant_padded_len + wild_len]


        mutant_forward_pooled = self._safe_global_pooling(
            mutant_forward_features,
            mutant_forward_mask,
            mutant_data.batch
        )

        wild_backward_pooled = self._safe_global_pooling(
            wild_backward_features,
            wild_backward_mask,
            wild_data.batch
        )


        diff_features = mutant_forward_pooled - wild_backward_pooled
        ddg_diff = self.ddg_predictor(diff_features).squeeze(-1)


        outputs['ddg'] = ddg_diff

        outputs['wild_contacts'] = wild_only_contacts
        outputs['wild_intensities'] = wild_only_intensities

        outputs['forward_contacts'] = all_forward_contacts
        outputs['forward_intensities'] = all_forward_intensities
        outputs['backward_contacts'] = all_backward_contacts
        outputs['backward_intensities'] = all_backward_intensities

        return outputs

    def compute_loss(self, wild_data, mutant_data, rna_data, ddg_target):
        'Compute loss.'

        outputs = self.forward(wild_data, mutant_data, rna_data)


        ddg_loss = F.mse_loss(outputs['ddg'], ddg_target)
        loss_dict = {'ddg_loss': ddg_loss}


        aux_loss = 0

        if hasattr(wild_data, 'block_contact_dist') and hasattr(wild_data, 'block_contact_int'):
            try:

                gt_dist = wild_data.block_contact_dist.to(ddg_loss.device)
                gt_int = wild_data.block_contact_int.to(ddg_loss.device)


                contact_loss = 0
                intensity_loss = 0
                valid_predictions = 0


                contact_source = outputs.get('wild_contacts', [])
                intensity_source = outputs.get('wild_intensities', [])


                for contacts, intensities in zip(contact_source, intensity_source):

                    if contacts.size(-1) != gt_dist.size(1):
                        print(f"Contact prediction size mismatch: {contacts.size()} vs {gt_dist.size()}")
                        continue


                    min_blocks = min(contacts.size(0), gt_dist.size(0))


                    dist_loss = F.kl_div(
                        F.log_softmax(contacts[:min_blocks], dim=-1),
                        gt_dist[:min_blocks],
                        reduction='batchmean'
                    )


                    int_loss = F.binary_cross_entropy(
                        intensities[:min_blocks],
                        gt_int[:min_blocks],
                        reduction='mean'
                    )


                    contact_loss += dist_loss
                    intensity_loss += int_loss
                    valid_predictions += 1


                if valid_predictions > 0:
                    contact_loss /= valid_predictions
                    intensity_loss /= valid_predictions


                    aux_loss = contact_loss + intensity_loss


                    loss_dict['contact_loss'] = contact_loss
                    loss_dict['intensity_loss'] = intensity_loss

            except Exception as e:
                print(f"Could not compute the auxiliary loss: {e}")
                import traceback
                traceback.print_exc()


        total_loss = ddg_loss + self.aux_weight * aux_loss
        loss_dict['total_loss'] = total_loss
        loss_dict['aux_loss'] = aux_loss

        return total_loss, loss_dict


# Backward-compatible alias for checkpoints and scripts created before the
# public model name was finalized as iSCALE.
DualSSD = ISCALE
