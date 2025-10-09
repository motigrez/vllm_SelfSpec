# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import numpy as np
from numba import get_num_threads, jit, njit, prange, set_num_threads

from vllm.config import VllmConfig

from vllm.logger import init_logger
logger = init_logger(__name__)

class NgramProposer:

    def __init__(self, vllm_config: VllmConfig):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.prompt_lookup_min is not None
        assert vllm_config.speculative_config.prompt_lookup_max is not None

        # Minimum length of the n-gram to match.
        self.min_n = vllm_config.speculative_config.prompt_lookup_min
        # Maximum length of the n-gram to match.
        self.max_n = vllm_config.speculative_config.prompt_lookup_max
        # Number of tokens follow the match. If there are less than k
        # tokens follow the match, we will return the maximum amount of
        # tokens until the end.
        self.k = vllm_config.speculative_config.num_speculative_tokens
        # Maximum length of the model.
        self.max_model_len = vllm_config.model_config.max_model_len

        # Pre-allocate buffers for numba batch propose.
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.valid_ngram_draft = np.zeros((max_num_seqs, self.k),
                                          dtype=np.int32)
        self.valid_ngram_num_drafts = np.zeros((max_num_seqs), dtype=np.int32)
        # Track the longest context observed per prompt group.
        self.group_longest_sequences: dict[int,
                                           tuple[str, np.ndarray]] = {}

        # Threshold of total number of tokens in the batch to enable
        # multi-threading in numba batch propose.
        self.num_tokens_threshold = 8192
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        cpu_count = os.cpu_count()
        # Max number of threads for numba parallel processing.
        if cpu_count:
            # Divide by 2 to use physical cores
            # and not logical cores (hyper-threading).
            # Cap the number of threads to 8 to avoid using too many threads
            # since other components like frontend (incl tokenization)
            # and Structured Outputs also use multiple threads.
            # TODO(ekagra-ranjan): bump up the cap from 1 to 8
            # when TP parallelization for ngram is implemented.
            self.num_numba_thread_available = min(1, (cpu_count // 2))
            # Divide by tp_size to ensure each tensor parallel rank
            # has some threads since all ranks will run this.
            self.num_numba_thread_available //= tp_size
        else:
            self.num_numba_thread_available = 1

        # Trigger Numba JIT compilation for N-gram proposer.
        # This usually takes less than 1 second.
        self.propose([[]] * 1024, [""] * 1024, np.zeros(1024, dtype=np.int32),
                     np.zeros((1024, self.max_model_len), dtype=np.int32),
                     set(), [])

    def _update_group_longest_sequences(self, req_ids: list[str],
                                        prompt_group_ids: list[int],
                                        num_tokens_no_spec: np.ndarray,
                                        token_ids_cpu: np.ndarray) -> None:
        for idx, gid in enumerate(prompt_group_ids):
            if gid is None:
                continue
            seq_len = int(num_tokens_no_spec[idx])
            if seq_len <= 0:
                continue
            tokens = token_ids_cpu[idx, :seq_len]
            cached = self.group_longest_sequences.get(gid)
            if cached is None or seq_len > cached[1].shape[0]:
                self.group_longest_sequences[gid] = (
                    req_ids[idx], tokens.copy())

    def _prepare_group_context_data(
        self,
        req_ids: list[str],
        prompt_group_ids: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        num_requests = len(prompt_group_ids)
        group_slot_ids = np.full(num_requests, -1, dtype=np.int32)
        should_concat = np.zeros(num_requests, dtype=np.bool_)
        if not prompt_group_ids:
            empty_int = np.empty(0, dtype=np.int32)
            empty_bool = np.empty(0, dtype=np.bool_)
            return (group_slot_ids, empty_int, empty_int, empty_int,
                    empty_bool)

        unique_group_ids: list[int] = []
        seen_groups: set[int] = set()
        for gid in prompt_group_ids:
            if gid is None:   # ← new
                continue
            if gid in seen_groups:
                continue
            seen_groups.add(gid)
            unique_group_ids.append(gid)

        group_slot_map: dict[int, int] = {}
        offsets: list[int] = []
        lengths: list[int] = []
        context_segments: list[np.ndarray] = []
        total_length = 0
        for slot, gid in enumerate(unique_group_ids):
            group_slot_map[gid] = slot
            entry = self.group_longest_sequences.get(gid)
            tokens = (entry[1] if entry is not None else
                      np.empty(0, dtype=np.int32))
            tokens = np.asarray(tokens, dtype=np.int32)
            offsets.append(total_length)
            length = int(tokens.shape[0])
            lengths.append(length)
            if length > 0:
                context_segments.append(tokens)
            total_length += length

        if total_length > 0 and context_segments:
            group_context_tokens = np.empty(total_length, dtype=np.int32)
            cursor = 0
            for segment in context_segments:
                seg_len = segment.shape[0]
                if seg_len == 0:
                    continue
                group_context_tokens[cursor:cursor + seg_len] = segment
                cursor += seg_len
        else:
            group_context_tokens = np.empty(0, dtype=np.int32)

        group_context_offsets = (np.array(offsets, dtype=np.int32)
                                 if offsets else np.empty(0, dtype=np.int32))
        group_context_lengths = (np.array(lengths, dtype=np.int32)
                                 if lengths else np.empty(0, dtype=np.int32))

        for idx, gid in enumerate(prompt_group_ids):
            slot = group_slot_map.get(gid, -1)
            group_slot_ids[idx] = slot
            if slot == -1:
                continue
            entry = self.group_longest_sequences.get(gid)
            if entry is None:
                continue
            owner_req_id, tokens = entry
            if tokens.shape[0] > 0 and owner_req_id != req_ids[idx]:
                should_concat[idx] = True

        return (group_slot_ids, group_context_tokens, group_context_offsets,
                group_context_lengths, should_concat)

    def batch_propose(
        self,
        num_requests: int,
        valid_ngram_requests: list,
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
        prompt_group_ids: list[int],
        req_ids: list[str],
    ) -> list[list[int]]:
        """Batch version of ngram proposer using numba for acceleration.
        
        Args:
            valid_ngram_requests: 
                Set of indices of requests that need ngram proposals.
            num_tokens_no_spec: 
                Numpy array of shape (batch_size,) representing the number 
                of tokens without speculative tokens for each request.
            token_ids_cpu: 
                Numpy array of shape (batch_size, max_model_len) 
                representing the token IDs for each request.
            req_ids:
                List of request identifiers aligned with the batch.

        Returns:
            list[list[int]]: 
                A list where each element is a list of proposed 
                token IDs for the corresponding request.
        """
        draft_token_ids: list[list[int]] = []

        # Only run batch propose if there are requests needing ngram proposals.
        # avoid calling numba function with empty list which causes error
        # ValueError: cannot compute fingerprint of empty list
        if num_ngram_requests := len(valid_ngram_requests):
            original_num_numba_threads = get_num_threads()
            # Ensure we use at least one thread.
            # If total tokens is small, using multiple threads
            # may slow down due to overhead.
            total_tokens = np.sum(num_tokens_no_spec)
            if total_tokens >= self.num_tokens_threshold:
                final_num_threads = max(
                    1, min(self.num_numba_thread_available,
                           num_ngram_requests))
                set_num_threads(final_num_threads)
            else:
                set_num_threads(1)

            for req_idx in valid_ngram_requests:
                self.valid_ngram_num_drafts[req_idx] = 0

            if prompt_group_ids:
                (group_slot_ids, group_context_tokens, group_context_offsets,
                 group_context_lengths,
                 should_concat) = self._prepare_group_context_data(
                     req_ids, prompt_group_ids)
            else:
                group_slot_ids = np.full(num_requests, -1, dtype=np.int32)
                group_context_tokens = np.empty(0, dtype=np.int32)
                group_context_offsets = np.empty(0, dtype=np.int32)
                group_context_lengths = np.empty(0, dtype=np.int32)
                should_concat = np.zeros(num_requests, dtype=np.bool_)

            batch_propose_numba(valid_ngram_requests, num_tokens_no_spec,
                                token_ids_cpu, self.min_n, self.max_n,
                                self.max_model_len, self.k,
                                self.valid_ngram_draft,
                                self.valid_ngram_num_drafts,
                                prompt_group_ids,
                                group_slot_ids,
                                group_context_tokens,
                                group_context_offsets,
                                group_context_lengths,
                                should_concat)

            # Restore original number of threads.
            set_num_threads(original_num_numba_threads)

        for i in range(num_requests):
            if i in valid_ngram_requests and \
                self.valid_ngram_num_drafts[i] > 0:
                draft_token_ids.append(self.valid_ngram_draft[
                    i, :self.valid_ngram_num_drafts[i]].tolist())
            else:
                draft_token_ids.append([])

        return draft_token_ids

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        req_ids: list[str],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
        spec_decode_unsupported_reqs: set,
        prompt_group_ids: list[int],
    ) -> list[list[int]]:

        if prompt_group_ids:
            assert len(prompt_group_ids) == len(req_ids)
            self._update_group_longest_sequences(req_ids, prompt_group_ids,
                                                 num_tokens_no_spec,
                                                 token_ids_cpu)
            # logger.info(f"Length of req_ids: {len(req_ids)}, prompt_group_ids: {len(prompt_group_ids)}")
            # logger.info(f"Shape of num_tokens_no_spec: {num_tokens_no_spec.shape}, token_ids_cpu: {token_ids_cpu.shape}")
            # exit()
            # assert len(req_ids) == num_tokens_no_spec.shape[0]
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # ngram_running_info = [
            #     (req_ids[i], prompt_group_ids[i])
            #     for i in range(len(req_ids))
            # ]
            # logger.info(f"[NgramProposer] Currently running {len(ngram_running_info)} requests:")
            # for rid, gid in ngram_running_info:
            #     logger.info(f"  - request_id={rid}, prompt_group_id={gid}")
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # find which requests need ngram proposals
        valid_ngram_requests = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            num_sampled_ids = len(sampled_ids)
            if not num_sampled_ids:
                # Skip speculative decoding.
                continue

            # Skip requests that require sampling parameters that are not
            # supported with speculative decoding.
            req_id = req_ids[i]
            if req_id in spec_decode_unsupported_reqs:
                continue

            num_tokens = num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                # Skip requests that have already reached the max model length.
                continue

            valid_ngram_requests.append(i)

        draft_token_ids = self.batch_propose(
            len(sampled_token_ids),
            valid_ngram_requests,
            num_tokens_no_spec,
            token_ids_cpu,
            prompt_group_ids,
            req_ids,
        )

        return draft_token_ids

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass


@njit(parallel=True)
def batch_propose_numba(valid_ngram_requests: list,
                        num_tokens_no_spec: np.ndarray,
                        token_ids_cpu: np.ndarray, min_n: int, max_n: int,
                        max_model_len: int, k: int,
                        valid_ngram_draft: np.ndarray,
                        valid_ngram_num_drafts: np.ndarray,
                        prompt_group_ids: list,
                        group_slot_ids: np.ndarray,
                        group_context_tokens: np.ndarray,
                        group_context_offsets: np.ndarray,
                        group_context_lengths: np.ndarray,
                        should_concat: np.ndarray):

    for i in prange(len(valid_ngram_requests)):
        idx = valid_ngram_requests[i]
        num_tokens = num_tokens_no_spec[idx]
        context_token_ids = token_ids_cpu[idx, :num_tokens]

        if len(prompt_group_ids) > 0:
            slot = group_slot_ids[idx]
            if (slot >= 0 and slot < group_context_lengths.shape[0]):
                base_len = group_context_lengths[slot]
                if base_len > 0:
                    offset = group_context_offsets[slot]
                    source_tokens = group_context_tokens[offset:offset +
                                                         base_len]
                    if should_concat[idx]:
                        total_len = base_len + num_tokens
                        combined = np.empty(total_len, dtype=np.int32)
                        combined[:base_len] = source_tokens
                        combined[base_len:] = context_token_ids
                        context_token_ids = combined
                    else:
                        context_token_ids = source_tokens

        drafter_output = _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=context_token_ids,
            min_ngram=min_n,
            max_ngram=max_n,
            max_model_len=max_model_len,
            k=k)

        valid_ngram_num_drafts[idx] = drafter_output.shape[0]
        if len(drafter_output):
            valid_ngram_draft[idx, :drafter_output.shape[0]] = drafter_output


@jit(nopython=True)
def _find_longest_matched_ngram_and_propose_tokens(origin_tokens: np.ndarray,
                                                   min_ngram: int,
                                                   max_ngram: int,
                                                   max_model_len: int,
                                                   k: int) -> np.ndarray:
    """
    Find the longest n-gram which matches the suffix of the given tokens
    whose length is within [min_ngram, max_ngram] (inclusive).

    If found, we will extract k right after the matched ngram.
    """
    # Do not generate draft tokens is context is shorter than minimum n-gram
    total_token = origin_tokens.shape[0]
    if total_token < min_ngram:
        return np.empty((0, ), dtype=origin_tokens.dtype)

    # Do not generate draft tokens beyond the max model length.
    k = min(k, max_model_len - total_token)
    if k <= 0:
        return np.empty((0, ), dtype=origin_tokens.dtype)

    # Flip tokens, and the goal become to find longest ngram
    # on the rightmost position which matches the prefix with
    # length [min_n, max_n] (inclusive).
    tokens = origin_tokens[::-1]

    # Longest prefix (not including itself) which is a suffix of
    # the current position.
    #   lps[i] = max{v, where tokens[0:v] == tokens[i+1-v:i+1]}
    #
    # As ngram is capped by max_ngram to save memory, we only need to
    # store lps for the first max_ngram prefix.
    lps = np.zeros(max_ngram, dtype=np.int32)

    longest_ngram = 0
    position = 0

    # lps[0] always equal to 0, we start with index 1
    prev_lps = 0
    i = 1
    while i < total_token:
        # tokens[:prev_lps] is the longest prefix as a suffix of tokens[:i]
        if tokens[prev_lps] == tokens[i]:
            # Token match: tokens[:prev_lps+1] is the longest prefix as
            # a suffix of tokens[:i+1]
            prev_lps += 1
            # Check if we found a longer valid ngram.
            #
            # Update position when longest_ngram matched prev_lps,
            # as we want to get the target n-gram of the earliest position
            # in the original tokens (i.e.
            # latest position in the reversed tokens)
            if prev_lps >= longest_ngram:
                longest_ngram = prev_lps
                position = i
            if i < max_ngram:
                # Store LPS for the first max_ngram prefix
                lps[i] = prev_lps
            if prev_lps == max_ngram:
                # When prev_lps reached max_ngram, update prev_lps
                # to lps[max_ngram-1] to avoid matching ngram
                # longer than max_ngram
                prev_lps = lps[max_ngram - 1]
            i += 1
        elif prev_lps != 0:
            # Token mismatch: try the second longest prefix
            # among all suffix of tokens[:i],
            # which is the longest prefix of tokens[:prev_lps]
            prev_lps = lps[prev_lps - 1]
        else:
            # Token mismatch, and no more prefix (except empty string)
            # as a suffix of tokens[:i]
            i += 1

    if longest_ngram < min_ngram:
        # No valid ngram is found
        return np.empty((0, ), dtype=origin_tokens.dtype)

    # Flip the position back, so in origin_tokens,
    # origin_tokens[total_token-1-position:total_token-1-position+longest_ngram]
    # is the matched ngram, so we should start drafting tokens from
    # total_token-1-position+longest_ngram
    start_position = total_token - 1 - position + longest_ngram
    k = min(k, total_token - start_position)
    return origin_tokens[start_position:start_position + k]
