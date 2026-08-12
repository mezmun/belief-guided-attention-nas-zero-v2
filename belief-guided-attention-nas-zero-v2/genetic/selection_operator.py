from __future__ import division
import numpy as np
from scipy.stats import rankdata


class Selection(object):

    def RouletteSelection(self, _a, k, replace=False, rng=None):
        values = np.asarray(_a, dtype=float)
        rng = np.random if rng is None else rng
        if values.ndim != 1 or len(values) == 0:
            raise ValueError('fitness values must be a non-empty one-dimensional sequence')
        if np.any(values < 0) or not np.all(np.isfinite(values)):
            raise ValueError('fitness values must be finite and non-negative')
        if not replace and int(k) > len(values):
            raise ValueError('Cannot sample more unique items than available values')

        available = list(range(len(values)))
        selected = []
        for _ in range(int(k)):
            pool = available if not replace else list(range(len(values)))
            pool_values = np.asarray([values[index] for index in pool], dtype=float)
            total = float(pool_values.sum())
            if total <= 0:
                chosen = int(rng.choice(pool))
            else:
                u = float(rng.random_sample() * total)
                running = 0.0
                chosen = pool[-1]
                order = sorted(pool, key=lambda index: values[index], reverse=True)
                for index in order:
                    running += float(values[index])
                    if running > u:
                        chosen = index
                        break
            selected.append(chosen)
            if not replace:
                available.remove(chosen)
        return selected

    def WheelSelection(self, _fitness_values, k, replace=False, rng=None):
        fitness_values = np.asarray(_fitness_values, dtype=float)
        rng = np.random if rng is None else rng
        if fitness_values.ndim != 1 or len(fitness_values) == 0:
            raise ValueError('fitness values must be a non-empty one-dimensional sequence')
        if np.any(fitness_values < 0) or not np.all(np.isfinite(fitness_values)):
            raise ValueError('fitness values must be finite and non-negative')
        if not replace and int(k) > len(fitness_values):
            raise ValueError('Cannot sample more unique items than available values')
        total_fitness = float(np.sum(fitness_values))
        probabilities = None if total_fitness <= 0 else fitness_values / total_fitness
        idx_list = np.arange(len(fitness_values))
        return list(
            rng.choice(
                idx_list,
                int(k),
                replace=bool(replace),
                p=probabilities,
            )
        )
    @staticmethod
    def GetGeometricPseudoFitness(fitness_list, generation, total_generations, q_start=0.99, q_end=0.96):
        """
        Computes geometric pseudo-fitness scores based on ranked original fitness values.
        Higher fitness receives higher pseudo-fitness.
    
        Parameters:
            fitness_list (list): List of original fitness scores
            generation (int): Current generation number
            total_generations (int): Total number of generations
            q_start (float): Initial q value (default: 0.99)
            q_end (float): Final q value (default: 0.96)
    
        Returns:
            list: Pseudo-fitness values corresponding to original fitness scores
        """
        #print(f"generation = {generation}, type = {type(generation)}")
        #generation = int(generation[0]) if isinstance(generation, list) else int(generation)

        #generation = int(generation)

        #if total_generations == 0:
        #    raise ValueError("total_generations must be greater than 0")
            
        slope = (q_end - q_start) / total_generations
        q = q_start + slope * generation
        q = max(min(q, q_start), q_end)
    
        # High fitness → low rank → high pseudo-fitness
        ranks = rankdata([-f for f in fitness_list], method='average')
        pseudo_fitness = np.array([q ** (r - 1) for r in ranks])
    
        # Print summary
        max_idx = np.argmax(pseudo_fitness)
        min_idx = np.argmin(pseudo_fitness)
    
        print(f"[PseudoFitness] Gen {generation}/{total_generations} | q: {round(q, 4)} | "
              f"Max: {round(pseudo_fitness[max_idx], 5)} (fit: {fitness_list[max_idx]}) | "
              f"Min: {round(pseudo_fitness[min_idx], 5)} (fit: {fitness_list[min_idx]}) | "
              f"Ratio: {round(pseudo_fitness[max_idx] / pseudo_fitness[min_idx], 2)}")
    
        return pseudo_fitness.tolist()
if __name__ == '__main__':
    s = Selection()
    a = [1, 3, 2, 1, 4, 4, 5]
    selected_index = s.RouletteSelection(a, k=5)

    new_a =[a[i] for i in selected_index]
    print(list(np.asarray(a)[selected_index]))
    print(new_a)






