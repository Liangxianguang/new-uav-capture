"""Original point-weighted predictive metric formula with scalar memory only."""
import collections
import numpy as np


class PredictionMetrics:
    def __init__(self):
        self.groups = collections.defaultdict(lambda: {'calls': 0, 'target_points': 0, 'anchor_points': 0,
            'response_points': 0, 'ade_sum': 0., 'reference_sum': 0., 'response_sum': 0., 'kl_sum': 0., 'kl_calls': 0})
        self.calls = 0; self.anchor_zero = True

    def add(self, row, values, output, indices, kl=None):
        indices = np.asarray(indices, dtype=np.int64)
        if (indices.ndim != 1 or not len(indices) or len(set(indices.tolist())) != len(indices) or
            indices.min() < 0 or indices.max() >= len(values['proposed'])):
            raise ValueError('Unique declared predictive candidate population required')
        valid = values['valid'][indices]
        equal = (values['proposed'][indices] == values['anchor'][None]).all((1, 2, 3))
        paired = valid & values['anchor_valid'][None] & ~equal[:, None]
        predicted = output['prediction'][indices]; response = output['response'][indices]
        if predicted.shape != values['target'][indices].shape or not np.isfinite(predicted).all() or not np.isfinite(response).all():
            raise ValueError('Complete finite predictions required')
        item = self.groups[row['group']]; item['calls'] += 1; self.calls += 1
        item['target_points'] += int(valid.sum()); item['anchor_points'] += int(values['anchor_valid'].sum())
        item['response_points'] += int(paired.sum())
        item['ade_sum'] += float(np.linalg.norm(predicted-values['target'][indices], axis=-1)[valid].sum())
        # Preserve the original V10 diagnostic arithmetic, not a new M metric.
        reference = values['backbone']+output['motion'][0]
        item['reference_sum'] += float(np.linalg.norm(reference-values['anchor_target'], axis=-1)[values['anchor_valid']].sum())
        item['response_sum'] += float(np.linalg.norm(response-(values['target'][indices]-values['anchor_target'][None]), axis=-1)[paired].sum())
        self.anchor_zero &= bool((response[equal] == 0).all())
        if kl is not None:
            if not np.isfinite(kl): raise ValueError('Finite native BASE ranking KL required')
            item['kl_sum'] += float(kl); item['kl_calls'] += 1

    def result(self, expected_groups, require_ranking=False):
        if set(self.groups) != set(expected_groups) or any(not all(v[k] for k in ('target_points', 'anchor_points', 'response_points')) for v in self.groups.values()):
            raise ValueError('Complete declared point support in ALL predictive families required')
        rows = {g: {'ade_m': v['ade_sum']/v['target_points'], 'reference_ade_m': v['reference_sum']/v['anchor_points'],
            'paired_response_error_m': v['response_sum']/v['response_points'], 'target_points': v['target_points'],
            'response_points': v['response_points']} for g, v in self.groups.items()}
        result = {'calls': self.calls, 'groups': len(rows), 'by_group': rows,
            'group_equal_ade_m': float(np.mean([v['ade_m'] for v in rows.values()])),
            'group_equal_reference_ade_m': float(np.mean([v['reference_ade_m'] for v in rows.values()])),
            'group_equal_paired_response_error_m': float(np.mean([v['paired_response_error_m'] for v in rows.values()])),
            'anchor_response_exact_zero': self.anchor_zero}
        if require_ranking:
            if any(not v['kl_calls'] for v in self.groups.values()):
                raise ValueError('BASE ranking support missing from some predictive family')
            ranking = {g: {'calls': v['kl_calls'], 'ranking_kl': v['kl_sum']/v['kl_calls']} for g, v in self.groups.items()}
            result.update(group_equal_ranking_kl=float(np.mean([v['ranking_kl'] for v in ranking.values()])),
                ranking_by_group=ranking, ranking_complete_calls=sum(v['calls'] for v in ranking.values()))
        return result
