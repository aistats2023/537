def dict_subset(d, keys):
    """ Returns a dictionary with requested subset of keys """
    if not set(keys).issubset(set(d.keys())):
        raise ValueError('Subset failed due to missing keys, d={} requested={}'.format(d.keys(), keys))
    return dict((k, d[k]) for k in keys)


def dict_set_if_none(d, key, new_value):
    """ If key's current value is None, modifies value to new_value """
    if key not in d:
        raise ValueError('Expected key={} in arguments'.format(key))
    if d[key] is None:
        d[key] = new_value
