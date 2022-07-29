import numpy as np


def multivalid_coverage(tau, x_train, y_train, f=(lambda x: 0), num_grid=100, group_fn=(lambda x: [True]), opt_rounds=1000):

    # initialize group structure on training dataaset
    num_groups = len(group_fn(x_train[0]))
    groups = [np.where(np.array([group_fn(x)[j] for x in x_train]))[0] for j in range(num_groups)]

    # initialize function-value distribution on training dataset
    grid = [(2*i - 1)/(2*num_grid) for i in range(1, num_grid+1)]

    def find_closest_grid_point_index(pred):
        pred_grid = (np.floor(pred*2*num_grid) + (np.floor(pred*2*num_grid)+1) % 2)/(2*num_grid)
        b_num = int((pred_grid * (2*num_grid) - 1)/2)
        return b_num

    f_val = np.array([grid[find_closest_grid_point_index(f(x))] for x in x_train]) 
    vals = [set(np.where(np.array([fval == val for fval in f_val]))[0]) for val in grid]

    # for each intersection of (group, value), 
    # y_num = number of training points there,
    # y_sum = sum of coverage indicators for training points there
    # => y_sum/y_num = empirical coverage in this intersection
    y_sum, y_num = np.zeros((num_groups, num_grid)), np.zeros((num_groups, num_grid))
    for i in range(num_groups):
        for j in range(num_grid):
            inds = list(vals[j].intersection(groups[i]))
            y_num[i][j] = len(inds)
            if inds:
                y_sum[i][j] = np.sum([int(y_train[ind] < grid[j]) for ind in inds])

    transform = []
    for step in range(opt_rounds):

        ### find most violated constraint
        violations = np.array([[y_num[i][j]*(tau - (tau if y_num[i][j] == 0 else y_sum[i][j]/y_num[i][j]))**2 for j in range(num_grid)] for i in range(num_groups)])
        g, val = np.unravel_index(np.argmax(violations, axis=None), violations.shape)

        print('Max violation in round', step, ' : ', violations[g, val])

        # indices of points to be re-valued at this round
        inds = np.array(list(vals[val].intersection(groups[g])))

        val_new = find_closest_grid_point_index(np.quantile(y_train[inds], tau))
        if val == val_new:
            break

        ### append update to transform[]
        transform.append((g, val, val_new))
        print('Update:', (g, val, val_new))

        # update vals
        vals[val]     -= set(inds)
        vals[val_new] |= set(inds)

        # update y_sum, y_num
        for i in range(num_groups):
            inds_group = np.array(list(set(inds).intersection(groups[i])))
            y_sum[i,val]     -= np.sum([int(y_train[ind] < grid[val])     for ind in inds_group])
            y_sum[i,val_new] += np.sum([int(y_train[ind] < grid[val_new]) for ind in inds_group])
            y_num[i,val]     -= len(inds_group)
            y_num[i,val_new] += len(inds_group)

    return transform

def eval_fn(x, patches, f=(lambda x: 0), num_grid=100, group_fn=(lambda x: [True])):

    grid = [(2*i - 1)/(2*num_grid) for i in range(1, num_grid+1)]
    relevant_patches = [patch for patch in patches if group_fn(x)[patch[0]]]

    ind_curr_value = 0
    for patch in relevant_patches:
        if patch[1] == ind_curr_value:
            ind_curr_value = patch[2]
    
    return grid[ind_curr_value]
