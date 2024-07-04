import random



def freq_mask(spec, num_masks=2, replace_with_zero=True):
    B, C, Freq, Time = spec.shape
    cloned = spec.clone()
    if random.random() < 0.5:
        num_masks = 1
    else:
        num_masks = 2

    for i in range(B):
        for _ in range(num_masks):
            if random.random() < 0.5:
                mask_len = Freq // 10
            else:
                mask_len = Freq // 5
            # f = random.randrange(0, Freq)
            f_zero = random.randrange(0, Freq - mask_len//2 - 1)

            # avoids randrange error if values are equal and range is empty
            # if (f_zero == Freq): continue

            mask_end = random.randrange(f_zero + mask_len//2, f_zero + mask_len)
            if (mask_end >= Freq):
                cloned[i, :, f_zero:, :] = 0
            if replace_with_zero:
                cloned[i, :, f_zero:mask_end, :] = 0
            else:
                cloned[i, :, f_zero:mask_end, :] = cloned[i, :, f_zero:mask_end, :].mean()

    return cloned


def time_mask(spec, num_masks=2, replace_with_zero=True):
    B, C, Freq, Time = spec.shape
    cloned = spec.clone()
    if random.random() < 0.5:
        num_masks = 1
    else:
        num_masks = 2
    for i in range(B):
        for _ in range(num_masks):
            if random.random() < 0.5:
                mask_len = Time // 10
            else:
                mask_len = Time // 5
            # t = random.randrange(0, Time)
            t_zero = random.randrange(0, Time - mask_len//2 -1)

            # avoids randrange error if values are equal and range is empty
            # if (t_zero == Time): continue

            mask_end = random.randrange(t_zero + mask_len//2, t_zero + mask_len)
            if (mask_end > Time):
                cloned[i, :, :, t_zero:] = 0
            if replace_with_zero:
                cloned[i, :, :, t_zero:mask_end] = 0
            else:
                cloned[i, :, :, t_zero:mask_end] = cloned[i, :, :, t_zero:mask_end].mean()

    return cloned

