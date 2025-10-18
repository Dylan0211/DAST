from models.DAST.daily_profile.sax import SAX
from scipy.stats import gaussian_kde
import pickle
import random
import numpy as np


def combine_load_traces_with_augmented_remainder(df_synthetic_in_this_context, cl_model, num_of_noises):
    original_season = df_synthetic_in_this_context['nor_el_season']
    original_remainder = df_synthetic_in_this_context['nor_el_remainder']

    # get kde
    flag_augment_with_kde = True
    try:
        kde = gaussian_kde(original_remainder)
    except:
        flag_augment_with_kde = False

    original_season_units = []
    for i in range(original_season.shape[0] // 24):
        original_season_units.append(original_season[24 * i: 24 * (i + 1)])
    original_season_units = np.array(original_season_units)

    augmented_load_traces = []
    augmented_remainders = []
    for i in range(original_season_units.shape[0]):
        this_day_trace_without_remainder = np.array(original_season_units[i])

        if flag_augment_with_kde:
            # remainder augmentation with kde
            this_day_augmented_remainder = list(kde.resample(num_of_noises)[0]) + [0 for _ in range(24 - num_of_noises)]
            random.shuffle(this_day_augmented_remainder)
            this_day_augmented_remainder = np.array(this_day_augmented_remainder)
            this_day_trace_with_augmented_remainder = np.array(this_day_trace_without_remainder + this_day_augmented_remainder)
        else:
            # remainder augmentation with simple shuffle
            this_day_augmented_remainder = list(original_remainder.sample(n=num_of_noises)) + [0 for _ in range(24 - num_of_noises)]
            random.shuffle(this_day_augmented_remainder)
            this_day_augmented_remainder = np.array(this_day_augmented_remainder)
            this_day_trace_with_augmented_remainder = np.array(this_day_trace_without_remainder + this_day_augmented_remainder)
        for i in range(this_day_trace_with_augmented_remainder.shape[0]):
            if this_day_trace_with_augmented_remainder[i] > 1:
                this_day_trace_with_augmented_remainder[i] = 1
            elif this_day_trace_with_augmented_remainder[i] < 0:
                this_day_trace_with_augmented_remainder[i] = 0

        if cl_model is not None:
            num_iter = 0
            while cl_model.predict(this_day_trace_without_remainder, this_day_trace_with_augmented_remainder) > 0.5 and num_iter < 10:
                if flag_augment_with_kde:
                    # remainder augmentation with kde
                    this_day_augmented_remainder = list(kde.resample(num_of_noises)[0]) + [0 for _ in range(24 - num_of_noises)]
                    random.shuffle(this_day_augmented_remainder)
                    this_day_augmented_remainder = np.array(this_day_augmented_remainder)
                    this_day_trace_with_augmented_remainder = np.array(this_day_trace_without_remainder + this_day_augmented_remainder)
                else:
                    # remainder augmentation with simple shuffle
                    this_day_augmented_remainder = list(original_remainder.sample(n=num_of_noises)) + [0 for _ in range(24 - num_of_noises)]
                    random.shuffle(this_day_augmented_remainder)
                    this_day_augmented_remainder = np.array(this_day_augmented_remainder)
                    this_day_trace_with_augmented_remainder = np.array(this_day_trace_without_remainder + this_day_augmented_remainder)
                for i in range(this_day_trace_with_augmented_remainder.shape[0]):
                    if this_day_trace_with_augmented_remainder[i] > 1:
                        this_day_trace_with_augmented_remainder[i] = 1
                    elif this_day_trace_with_augmented_remainder[i] < 0:
                        this_day_trace_with_augmented_remainder[i] = 0
                num_iter += 1

        augmented_load_traces.append(this_day_trace_with_augmented_remainder)
        augmented_remainders.append(this_day_augmented_remainder)

    augmented_load_traces = np.array(augmented_load_traces)
    augmented_remainders = np.array(augmented_remainders)
    df_synthetic_in_this_context['nor_el_remainder'] = augmented_remainders.flatten()
    df_synthetic_in_this_context['nor_el'] = augmented_load_traces.flatten()

    return df_synthetic_in_this_context

