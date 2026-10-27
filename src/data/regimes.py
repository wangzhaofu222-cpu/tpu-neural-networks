from src.data.splits import train_subset


def nested_subsets(labels,train_indices,fractions,base_seed):
    return {r:train_subset(labels,train_indices,r,base_seed) for r in fractions}
