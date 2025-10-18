import random

"""
Deviation detection
"""
def detect_up_spike(sequence, start_point):
    up_spike_flag = False
    spike_length = 0

    # parameters
    # lower height / smaller down_scale ==> more deviation detected
    # height = 0.1
    up_scale = 0.3
    down_scale = 0.8

    # note: three points
    if start_point + 2 >= len(sequence):
        return up_spike_flag, spike_length
    s1 = sequence[start_point]
    s2 = sequence[start_point + 1]
    s3 = sequence[start_point + 2]

    if (s2 - s1) >= s2 * up_scale and (s2 - s3) >= (s2 - s1) * down_scale:
        up_spike_flag = True
        spike_length = 3

    # note: four points
    # if start_point + 3 >= len(sequence):
    #     return up_spike_flag, spike_length
    # s4 = sequence[start_point + 3]
    #
    # if ((s2 - s1) >= s2 * up_scale and s4 <= s3 <= s2 and (s2 - s4) >= (s2 - s1) * down_scale) or \
    #         (s3 >= s2 >= s1 and (s3 - s1) >= s3 * up_scale and (s3 - s4) >= (s3 - s1) * down_scale):
    #     up_spike_flag = True
    #     spike_length = 4

    return up_spike_flag, spike_length


def detect_down_spike(sequence, start_point):
    down_spike_flag = False
    spike_length = 0

    # parameters
    # lower down_scale / smaller up_scale ==> more deviation detected
    down_scale = 0.5
    up_scale = 0.3

    # note: three points
    if start_point + 2 >= len(sequence):
        return down_spike_flag, spike_length
    s1 = sequence[start_point]
    s2 = sequence[start_point + 1]
    s3 = sequence[start_point + 2]

    if (s1 - s2) >= s1 * down_scale and (s3 - s2) >= (s1 - s2) * up_scale:
        down_spike_flag = True
        spike_length = 3

    # note: four points
    # if start_point + 3 >= len(sequence):
    #     return down_spike_flag, spike_length
    # s4 = sequence[start_point + 3]
    #
    # if ((s1 - s2) >= s1 * down_scale and s2 <= s3 <= s4 and (s4 - s2) >= (s1 - s2) * up_scale) or \
    #         (s1 >= s2 >= s3 and (s1 - s3) >= s1 * down_scale and (s4 - s3) >= (s1 - s3) * up_scale):
    #     down_spike_flag = True
    #     spike_length = 4

    return down_spike_flag, spike_length


def detect_up_down_spike(sequence, start_point):
    """
    这里有个问题是 up 上去后 down 下来到什么程度，是不是一定要比原来的水平低
    """
    up_down_spike_flag = False
    spike_length = 0

    # parameters
    # lower first_scale / smaller second_scale ==> more deviation detected
    first_scale = 0.3
    second_scale = 0.8

    # note: four points
    if start_point + 3 >= len(sequence):
        return up_down_spike_flag, spike_length
    s1 = sequence[start_point]
    s2 = sequence[start_point + 1]
    s3 = sequence[start_point + 2]
    s4 = sequence[start_point + 3]

    if ((s2 - s1) >= s2 * first_scale and (s2 - s3) >= (s2 - s1) * (1 + second_scale) and (s4 - s3) >= (s2 - s3) * second_scale) or \
            ((s1 - s2) >= s1 * first_scale and (s3 - s2) >= (s1 - s2) * (1 + second_scale) and (s3 - s4) >= (s3 - s2) * second_scale):
        up_down_spike_flag = True
        spike_length = 4

    return up_down_spike_flag, spike_length

"""
Deviation removal
"""
def remove_spike_deviation(sequence, start_point, spike_length):
    if spike_length == 3:
        sequence[start_point + 1] = (sequence[start_point] + sequence[start_point + 2]) / 2
    elif spike_length == 4:
        sequence[start_point + 1] = (1 / 3) * (sequence[start_point + 3] - sequence[start_point]) + sequence[start_point]
        sequence[start_point + 2] = (1 / 3) * (sequence[start_point + 3] - sequence[start_point]) * 2 + sequence[start_point]

    return sequence

"""
Deviation insertion
"""
def randomly_insert_spike_deviation(sequence, original_spike_points, insert_spike_at_original_points_flag):
    max_load = max(sequence)
    min_load = min(sequence)
    height_bound = 0.5 * (max_load - min_load)

    num_of_inserted_spikes = random.randint(1, 3)
    for _ in range(num_of_inserted_spikes):
        # spike type:
        # 0 ---> up spike
        # 1 ---> down spike
        # 2 ---> up & down spike
        random_spike_type = random.randint(0, 2)
        if random_spike_type == 0 or random_spike_type == 1:
            random_spike_length = 3
        else:
            random_spike_length = 4
        random_spike_start_point = random.randint(0, len(sequence) - 1)
        if not insert_spike_at_original_points_flag:
            while random_spike_start_point in original_spike_points or (
                    random_spike_start_point + random_spike_length) >= len(sequence):
                random_spike_start_point = random.randint(0, len(sequence) - 1)

        random_height = (random.random() + 1) * height_bound
        random_height_extra = (random.random() + 1) * height_bound
        if random_spike_type == 0:
            if random_spike_length == 3:
                sequence[random_spike_start_point + 1] = sequence[random_spike_start_point] + random_height
                sequence[random_spike_start_point + 1] = 1.0 if sequence[random_spike_start_point + 1] > 1 else \
                sequence[random_spike_start_point + 1]
        elif random_spike_type == 1:
            if random_spike_length == 3:
                sequence[random_spike_start_point + 1] = sequence[random_spike_start_point] - random_height
                sequence[random_spike_start_point + 1] = 0. if sequence[random_spike_start_point + 1] < 0 else sequence[
                    random_spike_start_point + 1]
        else:
            if random.randint(0, 1) == 0:
                sequence[random_spike_start_point + 1] = sequence[random_spike_start_point] + random_height
                sequence[random_spike_start_point + 2] = sequence[random_spike_start_point] - random_height_extra
                sequence[random_spike_start_point + 1] = 1. if sequence[random_spike_start_point + 1] > 1 else sequence[
                    random_spike_start_point + 1]
                sequence[random_spike_start_point + 2] = 0. if sequence[random_spike_start_point + 2] < 0 else sequence[
                    random_spike_start_point + 2]
            else:
                sequence[random_spike_start_point + 1] = sequence[random_spike_start_point] - random_height
                sequence[random_spike_start_point + 2] = sequence[random_spike_start_point] + random_height_extra
                sequence[random_spike_start_point + 1] = 0. if sequence[random_spike_start_point + 1] < 0 else sequence[
                    random_spike_start_point + 1]
                sequence[random_spike_start_point + 2] = 1. if sequence[random_spike_start_point + 2] > 1 else sequence[
                    random_spike_start_point + 2]

    return sequence
