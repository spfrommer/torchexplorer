from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, List
import torch
import numpy as np
from torch import Tensor


HistogramCounts = List[int]


@dataclass
class HistogramParams:
    bins: int = 10
    sample_n: int = 100
    reject_outlier_proportion: float = 0
    time_unit: str = 'step'


class IncrementalHistogram:
    """A time-indexed incremental histogram. The histogram is automatically rebinned
    when new data goes outside the current bounds of the histogram."""

    def __init__(self, params: HistogramParams):
        self.params = params
        self.min: Optional[float] = None
        self.max: Optional[float] = None

        self.history_times: list[int] = [] # e.g., the step
        self.history_bins: list[HistogramCounts] = []
        self.bin_counts: HistogramCounts = [0 for _ in range(self.params.bins)]

    def update(self, tensor: Tensor):
        """Update the histogram with new data. If the new data is outside the current
        bounds of the histogram, the histogram automatically rebins."""

        tensor = tensor.flatten().float()

        if self.params.sample_n < tensor.shape[0]:
            indices = torch.randint(tensor.shape[0], (self.params.sample_n,))
            tensor = tensor[indices]

        if self.params.reject_outlier_proportion > 0:
            reject_n = int(tensor.shape[0] * self.params.reject_outlier_proportion)
            if reject_n > 0:
                center = tensor.median()
                tensor = tensor[torch.argsort(torch.abs(tensor - center))[:-reject_n]]


        t_min, t_max = tensor.min(), tensor.max()
        if self.min is None:
            self.min = (t_min + 1.5 * (t_min - t_max - 0.0001)).item()
            self.max = (t_min + 1.5 * (t_max - t_min + 0.0001)).item()

        while self.min >= t_min:
            self._rebin('min')
        while self.max <= t_max:
            self._rebin('max')

        assert (self.min is not None) and (self.max is not None)
        hist = torch.histc(
            tensor.to('cpu'), bins=self.params.bins, min=self.min, max=self.max
        )
        self.bin_counts = [
            self.bin_counts[i] + int(cnt) for i, cnt in enumerate(hist.tolist())
        ]

    def push_history(self, time: int):
        """Push the current bin counts to the history and reset the bin counts."""
        self.history_bins.append(self.bin_counts)
        self.history_times.append(time)
        self.bin_counts = [0 for _ in range(self.params.bins)]

    def subsample_history(
            self, subsample_n=10
        ) -> tuple[list[HistogramCounts], list[int]]:
        """Subsample the history of bin counts to a list of length subsample_n.
        Will always include the first and last bin counts."""
        if len(self.history_bins) <= subsample_n:
            return self.history_bins, self.history_times
        else:
            indices = np.round(np.linspace(
                0, len(self.history_bins) - 1, subsample_n
            )).astype(int)
            history_bins_array = np.array(self.history_bins)
            history_times_sel = [self.history_times[i] for i in indices]
            
            history_bins_sel = []
            # Average up the histograms between the selected indices
            indices = np.append(indices, indices[-1] + 1)
            for i, i_next in zip(indices[:-1], indices[1:]):
                mean_bin_counts = history_bins_array[i:i_next].mean(axis=0)
                history_bins_sel.append(np.ceil(mean_bin_counts).astype(int).tolist())
            return history_bins_sel, history_times_sel

    def _bin_width(self):
        return (self.max - self.min) / self.params.bins

    def _rebin(self, double_bound: Literal['min', 'max']):
        """When new data goes outside the max/min bounds of the histogram, we must
        double the appropriate bound and rebin the histogram. We double so that there's
        no ambiguity about how to split up the old data into the new bins."""
        assert (self.min is not None) and (self.max is not None)

        def rebin(bin_counts):
            added_bins = [0 for _ in range(self.params.bins // 2)]
            # Sum subsequent pairs of bin counts
            grouped_bins = [
                sum(bin_counts[i:i+2]) for i in range(0, len(bin_counts), 2)
            ]
            if double_bound == 'min':
                return added_bins + grouped_bins
            elif double_bound == 'max':
                return grouped_bins + added_bins

        if double_bound == 'min':
            self.min = self.max + 2 * (self.min - self.max)
        elif double_bound == 'max':
            self.max = self.min + 2 * (self.max - self.min)
        else:
            raise ValueError(f'Invalid double_bound: {double_bound}')

        self.bin_counts = rebin(self.bin_counts)
        self.history_bins = [rebin(bin_counts) for bin_counts in self.history_bins]

    def __str__(self) -> str:
        return f'IncrementalHistogram(history={len(self.history_bins)})'
