"""PQ cache top-K masker implementation."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask
from sparse_attention_hub.sparse_attention.utils.mask_attention_utils import (
    get_masked_attention_output,
    get_true_attention_output,
)

from ..base import TopKMasker, TopKMaskerConfig


@dataclass
class PQCacheConfig(TopKMaskerConfig):
    """Configuration for PQCache masker."""

    pq_sub_dim: int
    pq_bits: int


@MaskerRegistry.register(PQCacheConfig)
class PQCache(TopKMasker):
    """PQ cache-based top-K masker."""

    def __init__(self, config: PQCacheConfig) -> None:
        """Initialize PQ cache masker with configuration."""
        super().__init__(config)
        self.heavy_size = config.heavy_size
        self.pq_sub_dim = config.pq_sub_dim
        self.pq_bits = config.pq_bits


    def add_mask(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
        scaling: float,
        dropout: float,
        sparse_meta_data: Dict[Any, Any],
        previous_mask: Mask,
        **kwargs: Dict[str, Any],
    ) -> Mask:
        
        
        # get module from kwargs
        module = kwargs.get('module')
        
        # handle head dimension mismatch  stubborn erorr
        if queries.shape[1] != keys.shape[1]:
            ##repeat key/value heads to match query heads
            repeat_factor = queries.shape[1] // keys.shape[1]
            keys = keys.repeat_interleave(repeat_factor, dim=1)
            values = values.repeat_interleave(repeat_factor, dim=1)
    
        #build PQ structures during decoding phase when module is available?
        

        L = queries.shape[2] #length of input token sequence

        LocalKVCache = [] #set empty
        #decoding phase
        is_decoding: bool = (queries.shape[2] == 1) and ("PQ" in sparse_meta_data) and ("KVCache" in sparse_meta_data)
        if is_decoding:
            PQ = sparse_meta_data["PQ"]  # list of (centroids, codes)
            KVCache = sparse_meta_data["KVCache"]  # list of (K, V))

            L_pq: int = len(PQ)
            Q = queries[:, :, -1, :]  # last-step query (B,H,D)
            last_topk_idx: Optional[torch.Tensor] = None

            for i in range(0, L_pq):
                if LocalKVCache:
                    EvictK, EvictV = LocalKVCache.pop(0)
                    _ = self._pq_encode(EvictK, PQ[i][0])

                K = keys[:, :, -1, :]
                V = values[:, :, -1, :]
                LocalKVCache.append((K, V))

                Centroids = PQ[i][0]
                Codes_i = PQ[i][1].to(queries.device)

                TopKIdx = self._pq_search(Q, Centroids, Codes_i, self.heavy_size)
                TopkK, TopkV = self._sync_fetch_kv(KVCache, TopKIdx)
                last_topk_idx = TopKIdx

                # built AllKV = InitKV + TopkKV + LocalKV
                def _stack_kv(kv_list: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
                    if not kv_list:
                        B, H, D = keys.shape[0], keys.shape[1], keys.shape[-1]
                        empty = keys.new_zeros((B, H, 0, D))
                        return empty, empty
                    K_parts = [k.unsqueeze(2) if k.dim() == 3 else k for (k, _) in kv_list]
                    V_parts = [v.unsqueeze(2) if v.dim() == 3 else v for (_, v) in kv_list]
                    return torch.cat(K_parts, dim=2), torch.cat(V_parts, dim=2)
 
                InitKV = sparse_meta_data.get("InitKV", [])
                MidKV = sparse_meta_data.get("MidKV", [])
                K_init, V_init = _stack_kv(InitKV)
                K_mid, V_mid = _stack_kv(MidKV)

                # accumulated in localkv this step
                K_local, V_local = _stack_kv(LocalKVCache)

                # ensure TopkKV has a sequence dimension
                if TopkK.dim() == 3:
                    TopkK = TopkK.unsqueeze(2)
                    TopkV = TopkV.unsqueeze(2)

                K_all = torch.cat([K_init, K_mid, TopkK, K_local], dim=2)
                V_all = torch.cat([V_init, V_mid, TopkV, V_local], dim=2)

                # x = AttnFFN(Q, AllKV) using dense attention utility
                Q_last = queries[:, :, -1:, :]
                
                # handle none module case - create mask based on PQ search results
                if module is None:
                    # create mask that selects the top-K positions found by PQ search
                    if last_topk_idx is not None:
                        B, H, _k = last_topk_idx.shape
                        keys_len: int = keys.shape[2]
                        last_pos = (keys_len - 1)
                        last_idx = torch.full((B, H, 1), last_pos, dtype=last_topk_idx.dtype, device=last_topk_idx.device)
                        row_wise_idx = torch.cat([last_topk_idx.clamp_max(keys_len - 1), last_idx], dim=-1)
                        #add sequence dimension to make it 4D: (B, H, 1, k+1)
                        row_wise_idx = row_wise_idx.unsqueeze(2)  #B, H, 1, k+1)
                        data = torch.ones_like(row_wise_idx, dtype=queries.dtype)
                        new_mask = Mask.create_from_row_wise_idx(
                            shape=(B, H, 1, keys_len),
                            row_wise_idx=row_wise_idx,
                            data=data,
                            type="index",
                            dtype=queries.dtype,
                        )
                        merged = previous_mask.merge_mask(new_mask, inplace=False)
                        return merged
                    else:
                        return previous_mask
                
                result = get_true_attention_output(
                    module=module,
                    queries=Q_last,
                    keys=K_all,
                    values=V_all,
                    attention_mask=attention_mask,
                    scaling=scaling,
                    dropout=dropout,
                )
                x, _ = result

                #token = classifier(x) pseduo
                classifier_cfg: Dict[str, Any] = sparse_meta_data.get("classifier", {})
                W_vocab: Optional[torch.Tensor] = classifier_cfg.get("weight")
                b_vocab: Optional[torch.Tensor] = classifier_cfg.get("bias")
                if W_vocab is not None:
                    token = self._classifier(x, W_vocab, b_vocab)
                    sparse_meta_data["token"] = token

            # build a new sparse mask selecting TopK positions (union approximated by last shard)
            if last_topk_idx is not None:
                B, H, _k = last_topk_idx.shape
                keys_len: int = keys.shape[2]
                last_pos = (keys_len - 1)
                last_idx = torch.full((B, H, 1), last_pos, dtype=last_topk_idx.dtype, device=last_topk_idx.device)
                row_wise_idx = torch.cat([last_topk_idx.clamp_max(keys_len - 1), last_idx], dim=-1)
                data = torch.ones_like(row_wise_idx, dtype=queries.dtype)
                new_mask = Mask.create_from_row_wise_idx(
                    shape=(B, H, 1, keys_len),
                    row_wise_idx=row_wise_idx,
                    data=data,
                    type="index",
                    dtype=queries.dtype,
                )
                merged = previous_mask.merge_mask(new_mask, inplace=False)
                return merged


        #decoding



        #prefilling first
        ProcComm = []
        ProcPQ = []
        KVCache = []
        PQ = []

        # prefilling computation on GPU
        for i in range(0, L-1):
            #we are already given q,k,v in the add_mask
            #we want to get the query at the ith position
            q = queries[:, :, i, :]
            k = keys[:, :, i, :]
            v = values[:, :, i, :]

            #asynchronously offload from GPU to CPU, add to ProcComm for monitoring
            # Keep on GPU for now to avoid device mismatch issues
            ProcComm.append((k.detach(), v.detach()))
            
            # Store K,V for PQ construction
        


        #launch pq construction on same device as input
        for i in range(0, L-1):
            k, v = ProcComm[i]
            KVCache.append((k, v))
            # enqueue PQ construction task for this k
            ProcPQ.append(
                self._AsyncPQConstruct(
                    k, L, self.pq_sub_dim, self.pq_bits
                )
            )
            
        
        #wait for pq construction in the next decoding phase
        for i in range(0, L-1):
            centroids, codes = ProcPQ[i].sync()
            PQ.append((centroids, codes))
        
        #store results for the decoding phase and return the mask
        sparse_meta_data["PQ"] = PQ
        sparse_meta_data["KVCache"] = KVCache
        
        #need to retunr  the previous mask for prefilling phase
        return previous_mask

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "PQCache":
        """Create PQCache instance from configuration."""
        if not isinstance(config, PQCacheConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)



    def _classifier(
        self,
        X: torch.Tensor,
        W_vocab: torch.Tensor,
        b_vocab: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """ classifier: matmul + softmax + argmax.

        Returns token ids (B,) without probabilities and handles shapes minimally.
        """
        if X.dim() == 4:
            X = X[:, :, -1, :].mean(dim=1)  # (B, D)
        elif X.dim() == 3:
            X = X[:, -1, :]  # (B, D)

        logits: torch.Tensor = X @ W_vocab.t()
        if b_vocab is not None:
            logits = logits + b_vocab

        token: torch.Tensor = torch.softmax(logits, dim=-1, dtype=torch.float32).argmax(dim=-1)
        return token

    
    class _Sync:
        def __init__(self, k: torch.Tensor, v: torch.Tensor) -> None:
            self._k = k
            self._v = v
        def sync(self) -> tuple[torch.Tensor, torch.Tensor]:
            return self._k, self._v

    class _AsyncPQConstruct:
        def __init__(
            self,
            K: torch.Tensor,
            L: int,
            pq_sub_dim: int,
            pq_bits: int,
        ) -> None:
            self._K = K
            self._L = L
            self._pq_sub_dim = pq_sub_dim
            self._pq_bits = pq_bits

        def sync(self) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Construct a simple PQ: return (centroids, codes).

            centroids: (H, num_subspaces, 2**pq_bits, pq_sub_dim)
            codes: (B, H, S, num_subspaces) with integer codes per subspace
            """
            K = self._K  # expected shape (B, H, D)
            if K.dim() == 4:
                # if shape (B, H, S, D), reduce S by last step
                K = K[:, :, -1, :]

            B, H, D = K.shape
            assert D % self._pq_sub_dim == 0, "D must be divisible by pq_sub_dim"
            num_sub = D // self._pq_sub_dim
            num_centroids = 1 << self._pq_bits

            #reshape into subspaces like (B, H, num_sub, pq_sub_dim)
            K_sub = K.view(B, H, num_sub, self._pq_sub_dim)

            #simple uniform binning centroids per subspace/head based on our data range
            k_min = K_sub.amin(dim=0, keepdim=True)  #1, H, num_sub, pq_sub_dim)
            k_max = K_sub.amax(dim=0, keepdim=True)  #(1, H, num_sub, pq_sub_dim)
            # build centroids as evenly spaced points between min and max
            t = torch.linspace(0, 1, steps=num_centroids, device=K.device, dtype=K.dtype)
            # (1, 1, num_centroids, 1) for proper broadcasting with (H, num_sub, pq_sub_dim)
            t = t.view(1, 1, num_centroids, 1)
            # broadcast with : (H, num_sub, num_centroids, pq_sub_dim) format below
            k_min_squeezed = k_min.squeeze(0)  
            k_max_squeezed = k_max.squeeze(0)  
            centroids = k_min_squeezed.unsqueeze(2) + (k_max_squeezed - k_min_squeezed).unsqueeze(2) * t
            # rearrange to (H, num_sub, num_centroids, pq_sub_dim)

            #assign codes by nearest centroid (L2) per (B,H,num_sub)
            #expand K_sub to (B,H,num_sub,1,pq_sub_dim)
            # centroids is structure H, num_sub, num_centroids, pq_sub_dim)
            #K_sub is (B, H, num_sub, pq_sub_dim)
            diff = K_sub.unsqueeze(3) - centroids.unsqueeze(0)
            dists = (diff * diff).sum(dim=-1)  #(B,H,num_sub,num_centroids)
            codes = dists.argmin(dim=-1)  #(B,H,num_sub)

            return centroids, codes

    class _AsyncCPU2GPU:
        def __init__(self, tensor_cpu: torch.Tensor, device: torch.device) -> None:
            self._tensor_cpu = tensor_cpu
            self._device = device

        def sync(self) -> torch.Tensor:
            return self._tensor_cpu.to(self._device, non_blocking=True)

    def _pq_encode(self, k: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        if k.dim() == 2:
            k = k.unsqueeze(0)
        elif k.dim() == 4:  # need to handle (B,H,S,D) case
            k = k[:, :, -1, :]  # use the last timestep
        B, H, D = k.shape
        num_sub: int = centroids.shape[1]
        d_sub: int = centroids.shape[3]
        assert D % d_sub == 0 and (D // d_sub) == num_sub, "D must be divisible by d_sub and the result must be equal to num_sub"
        k_sub = k.view(B, H, num_sub, d_sub)  # changed K to k
        #nearest centroid per subspace
        #diff: (B, H, num_sub, d_sub)
        
        diff = k_sub.unsqueeze(3) - centroids.unsqueeze(0)  # Changed K to k
        dists = (diff * diff).sum(dim=-1)
        codes = dists.argmin(dim=-1)
        return codes

        
    def _pq_search(
        self,
        q: torch.Tensor,                 ##(B,H,D) last-step query
        centroids: torch.Tensor,         #(H, num_sub, C, d_sub)
        codes: torch.Tensor,             # cached: (Bc,H,num_sub) codes per cached position
        topk: int,
    ) -> torch.Tensor:
        #make q into subspaces
        B, H, D = q.shape
        num_sub, d_sub = centroids.shape[1], centroids.shape[3]
        q_sub = q.view(B, H, num_sub, d_sub)

        # closest centroid ids for q
        diff = q_sub.unsqueeze(3) - centroids.unsqueeze(0)        # (B,H,num_sub,C,d_sub)
        dists = (diff * diff).sum(dim=-1)                         # (B,H,num_sub,C)
        q_codes = dists.argmin(dim=-1)                            # (B,H,num_sub)

        # measure score cache positions by subspace code matches
        Bc = codes.shape[0]
        q_codes_exp = q_codes.unsqueeze(2).expand(B, H, Bc, num_sub)
        codes_exp = codes.unsqueeze(0).permute(0, 2, 1, 3).expand(B, H, Bc, num_sub)
        matches = (q_codes_exp == codes_exp).sum(dim=-1)          # (B,H,Bc)

        k = min(topk, Bc)
        _, idx = matches.topk(k, dim=-1)                          # (B,H,k)
        return idx

    def _sync_fetch_kv(
        self,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]],  # lenght=Bc, each (B,H,D)
        indices: torch.Tensor,                              # (B,H,k)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, H, k = indices.shape
        D = kv_cache[0][0].shape[-1]
        
        # stack the cache along position dim
        K_stack = torch.stack([kv_cache[i][0] for i in range(len(kv_cache))], dim=2)  # (B,H,Bc,D)
        V_stack = torch.stack([kv_cache[i][1] for i in range(len(kv_cache))], dim=2)  # (B,H,Bc,D)
        gather_idx = indices.unsqueeze(-1).expand(B, H, k, D)
        K_sel = K_stack.gather(dim=2, index=gather_idx)  # (B,H,k,D)
        V_sel = V_stack.gather(dim=2, index=gather_idx)  # (B,H,k,D)
        return K_sel, V_sel

        #git config





    

        
