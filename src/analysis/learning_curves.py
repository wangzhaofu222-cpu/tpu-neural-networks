import math
import statistics


def first_crossover(fractions,deltas):
    return next((r for r,d in zip(fractions,deltas) if d>=0),None)


def paired_architecture(cnn,vit):
    delta=[v-c for c,v in zip(cnn,vit)]
    mean=statistics.mean(delta);sd=statistics.stdev(delta)
    established=(all(d>0 for d in delta) or all(d<0 for d in delta)) and abs(mean)>=2*sd
    return {'deltas':delta,'mean':mean,'sample_std':sd,'established':established}


def shift_metrics(clean,levels):
    return {'accuracies':levels,'relative_drops':[(clean-x)/(clean+1e-12) for x in levels],
            'mean_corrupted_accuracy':statistics.mean(levels)}


def leader_reversal(clean_delta,corrupted_delta):
    return clean_delta*corrupted_delta<0 and abs(clean_delta)>=.01 and abs(corrupted_delta)>=.01
