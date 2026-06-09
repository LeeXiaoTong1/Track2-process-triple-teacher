from pathlib import Path
import re
import shutil

p = Path('model.py')
bak = Path('model.py.bak_before_stable_cross_all95')
if not bak.exists():
    shutil.copy2(p, bak)
    print(f'[backup] {p} -> {bak}')

s = p.read_text(encoding='utf-8')

new_class = r'''class CrossStreamFusionBlock(nn.Module):
    """
    Stable gated semantic-forensic cross interaction.

    This replaces the original bidirectional MultiheadAttention block.
    It keeps --ufm_layers 1 meaningful while avoiding the MHA backward NaN
    observed in UFM Track2 training. The block starts close to identity,
    so strong baseline-like Singing cues are less likely to be overwritten.
    """
    def __init__(self, dim=512, num_heads=8, dropout=0.1):
        super(CrossStreamFusionBlock, self).__init__()
        self.dim = dim

        self.sem_norm = nn.LayerNorm(dim)
        self.for_norm = nn.LayerNorm(dim)

        self.sem_ctx = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.for_ctx = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        self.for_to_sem_delta = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        self.for_to_sem_gate = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim)
        )

        self.sem_to_for_delta = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        self.sem_to_for_gate = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim)
        )

        # Very small learnable residual strengths. Upper bound = 0.02.
        self.raw_cross_scale = nn.Parameter(torch.tensor(-4.0))
        self.raw_ffn_scale = nn.Parameter(torch.tensor(-4.0))

        self.sem_ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )
        self.for_ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )

        self._stable_init()

    def _stable_init(self):
        # Delta/FFN branches start at exact zero: initial block = identity.
        nn.init.zeros_(self.for_to_sem_delta[-1].weight)
        nn.init.zeros_(self.for_to_sem_delta[-1].bias)
        nn.init.zeros_(self.sem_to_for_delta[-1].weight)
        nn.init.zeros_(self.sem_to_for_delta[-1].bias)
        nn.init.zeros_(self.sem_ffn[-1].weight)
        nn.init.zeros_(self.sem_ffn[-1].bias)
        nn.init.zeros_(self.for_ffn[-1].weight)
        nn.init.zeros_(self.for_ffn[-1].bias)

        # Gates start very small.
        nn.init.zeros_(self.for_to_sem_gate[-1].weight)
        nn.init.constant_(self.for_to_sem_gate[-1].bias, -3.0)
        nn.init.zeros_(self.sem_to_for_gate[-1].weight)
        nn.init.constant_(self.sem_to_for_gate[-1].bias, -3.0)

    def _pool(self, x):
        mean_pool = x.mean(dim=1)
        max_pool = x.max(dim=1).values
        return torch.cat([mean_pool, max_pool], dim=-1)

    def _finite_or_zero(self, x):
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(self, semantic_tokens, forensic_tokens):
        sem = self.sem_norm(torch.clamp(semantic_tokens, -20.0, 20.0))
        forg = self.for_norm(torch.clamp(forensic_tokens, -20.0, 20.0))

        sem_c = self.sem_ctx(self._pool(sem)).unsqueeze(1).expand(-1, sem.size(1), -1)
        for_c = self.for_ctx(self._pool(forg)).unsqueeze(1).expand(-1, forg.size(1), -1)

        sem_in = torch.cat([sem, for_c], dim=-1)
        for_in = torch.cat([forg, sem_c], dim=-1)

        sem_delta = torch.clamp(self._finite_or_zero(self.for_to_sem_delta(sem_in)), -5.0, 5.0)
        for_delta = torch.clamp(self._finite_or_zero(self.sem_to_for_delta(for_in)), -5.0, 5.0)

        sem_gate = torch.sigmoid(self.for_to_sem_gate(sem_in))
        for_gate = torch.sigmoid(self.sem_to_for_gate(for_in))

        cross_alpha = 0.02 * torch.sigmoid(self.raw_cross_scale)
        semantic_tokens = semantic_tokens + cross_alpha * sem_gate * sem_delta
        forensic_tokens = forensic_tokens + cross_alpha * for_gate * for_delta

        sem_ffn = torch.clamp(self._finite_or_zero(self.sem_ffn(semantic_tokens)), -5.0, 5.0)
        for_ffn = torch.clamp(self._finite_or_zero(self.for_ffn(forensic_tokens)), -5.0, 5.0)

        ffn_alpha = 0.02 * torch.sigmoid(self.raw_ffn_scale)
        semantic_tokens = semantic_tokens + ffn_alpha * sem_ffn
        forensic_tokens = forensic_tokens + ffn_alpha * for_ffn

        semantic_tokens = self._finite_or_zero(semantic_tokens)
        forensic_tokens = self._finite_or_zero(forensic_tokens)
        return semantic_tokens, forensic_tokens

'''

pattern = r'(?ms)^class CrossStreamFusionBlock\(nn\.Module\):.*?(?=^class BiCrossStreamTransformer\(nn\.Module\):)'
s2, n = re.subn(pattern, new_class + '\n', s, count=1)
if n != 1:
    raise RuntimeError(f'Failed to replace CrossStreamFusionBlock. matched={n}')

p.write_text(s2, encoding='utf-8')
print('[done] model.py CrossStreamFusionBlock replaced with stable gated all95 version')
