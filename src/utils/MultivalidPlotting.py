import numpy as np
import matplotlib.pyplot as plt

def plot_group_coverage(model, tau, x_test, y_test, num_groups, group_fn, multivalid=False, num_grid=100, verbose=True):

    if not multivalid:
        group_cov = np.zeros(num_groups)
        group_num = np.zeros(num_groups)

        for i in range(len(x_test)):
            pred = model(x_test[i])
            for j in range(num_groups):
                if group_fn(x_test[i])[j]:
                    group_num[j] += 1
                    group_cov[j] += int(pred > y_test[i])

        group_coverage = [group_cov[j]/group_num[j] for j in range(num_groups)]

        plt.figure(figsize=(10, 7))
        plt.bar(np.arange(num_groups), group_coverage, color='g', width=0.3, edgecolor='gray')
        plt.axhline(y=tau, c='r', linestyle='--', linewidth=2)
        plt.text(7.5, tau + 0.02, '  target coverage')
        plt.xlabel('Group Number')
        plt.xticks(range(num_groups), range(1, num_groups+1))
        plt.ylabel('Realized Group Coverage')
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.title('Performance of Optimal Conformal Predictor on Test Data')
        plt.show()

        if verbose:
            for j in range(num_groups):
                print('Coverage on group', j+1, ':', group_coverage[j])

    else:

        grid = [(2*i - 1)/(2*num_grid) for i in range(1, num_grid+1)]
        def find_closest_grid_point_index(pred):
            pred_grid = (np.floor(pred*2*num_grid) + (np.floor(pred*2*num_grid)+1) % 2)/(2*num_grid)
            b_num = int((pred_grid * (2*num_grid) - 1)/2)
            return b_num
        
        gb_cov = np.zeros((num_groups, num_grid))
        gb_num = np.zeros((num_groups, num_grid))

        for i in range(len(x_test)):
            pred = model(x_test[i])
            pred_bucket = find_closest_grid_point_index(pred)
            for j in range(num_groups):
                if group_fn(x_test[i])[j]:
                    gb_num[j][pred_bucket] += 1
                    gb_cov[j][pred_bucket] += int(pred > y_test[i])
        
        g_num = [sum([gb_num[j][b] for b in range(num_grid)]) for j in range(num_groups)]
        test_size = len(x_test)
        group_coverage = [ sum([ gb_num[j][b]*(tau - (tau if gb_num[j][b]==0 else gb_cov[j][b]/gb_num[j][b]))**2 for b in range(num_grid)])/test_size for j in range(num_groups)]
        
        plt.figure(figsize=(10, 7))
        plt.bar(np.arange(num_groups), group_coverage, color='g', width=0.3, edgecolor='gray')
        plt.xlabel('Group Number')
        plt.xticks(range(num_groups), range(1, num_groups+1))
        plt.ylabel('Realized Group Squared Coverage Loss')
        plt.title('Performance of Optimal Conformal Predictor on Test Data')
        plt.show()

        if verbose:
            for j in range(num_groups):
                print('Performance on group', j+1, ':', group_coverage[j])

def plot_pred_set_size(model, tau, x_test, y_test, num_groups, group_fn, mult_width, multivalid=False, num_grid=100, verbose=True):
    
    group_pred_set_size = np.zeros(num_groups)
    group_num = np.zeros(num_groups)

    for i in range(len(x_test)):
        pred = model(x_test[i])
        pred_set_size = 2 * pred * mult_width
        for j in range(num_groups):
            if group_fn(x_test[i])[j]:
                group_num[j] += 1
                group_pred_set_size[j] += pred_set_size

    group_pred_set_size = [group_pred_set_size[j]/group_num[j] for j in range(num_groups)]

    plt.figure(figsize=(10, 7))
    plt.bar(np.arange(num_groups), group_pred_set_size, color='g', width=0.3, edgecolor='gray')
    plt.xlabel('Group Number')
    plt.xticks(range(num_groups), range(1, num_groups+1))
    plt.ylabel('Mean prediction set size')
    plt.title('Performance of Conformal Predictor on Test Data')
    plt.show()

    if verbose:
        for j in range(num_groups):
            print('Average prediction set size on group', j+1, ':', group_pred_set_size[j])

def plot_all_group_coverage(model1, model2, tau, x_test, y_test, num_groups, group_fn, marginal_conformal_coverage, group_conformal_coverage, multivalid=False, num_grid=100, bar_width=0.25):
    group_cov1 = np.zeros(num_groups)
    group_num1 = np.zeros(num_groups)
    group_cov2 = np.zeros(num_groups)
    group_num2 = np.zeros(num_groups)

    for i in range(len(x_test)):
        pred1 = model1(x_test[i])
        pred2 = model2(x_test[i])
        for j in range(num_groups):
            if group_fn(x_test[i])[j]:
                group_num1[j] += 1
                group_cov1[j] += int(pred1 > y_test[i])
                group_num2[j] += 1
                group_cov2[j] += int(pred2 > y_test[i])

    group_coverage1 = [group_cov1[j]/group_num1[j] for j in range(num_groups)]
    group_coverage2 = [group_cov2[j]/group_num2[j] for j in range(num_groups)]

    br1 = np.arange(num_groups)
    br2 = [x + bar_width for x in br1]
    br3 = [x + bar_width for x in br2]
    br4 = [x + bar_width for x in br3]

    plt.figure(figsize=(10, 7))
    plt.bar(br1, marginal_conformal_coverage, color = 'b', width = bar_width, edgecolor = 'gray', label = 'Split-Conformal: Without groups', linewidth = 0.5)
    plt.bar(br2, group_conformal_coverage, color = 'm', width = bar_width, edgecolor = 'gray', label = 'Split-Conformal: With groups, conservative approach', linewidth = 0.5)
    plt.bar(br3, group_coverage1, color = 'c', width = bar_width, edgecolor = 'gray', label = 'One-shot group-accurate algorithm', linewidth = 0.5)
    plt.bar(br4, group_coverage2, color = 'g', width = bar_width, edgecolor = 'gray', label = 'Iterative multi-valid algorithm', linewidth = 0.5)
    plt.axhline(y=tau, c='r', linestyle='--', linewidth=2)
    # plt.text(8.5, tau + 0.02, '  target coverage')
    plt.xlabel('Group Number')
    plt.xticks(range(num_groups), range(1, num_groups+1))
    plt.ylabel('Realized Group Coverage')
    plt.yticks(np.arange(0, 1.4, 0.1))
    plt.legend()
    plt.title('Comparisons of group coverage using various methods on test data')
    plt.show()

def plot_all_pred_set_sizes(model1, model2, tau, x_test, y_test, num_groups, group_fn, mult_width, marginal_conformal_size, group_conformal_size, num_grid=100, multivalid=False, bar_width = 0.25):
    
    group_pred_set_size1 = np.zeros(num_groups)
    group_num1 = np.zeros(num_groups)
    group_pred_set_size2 = np.zeros(num_groups)
    group_num2 = np.zeros(num_groups)

    for i in range(len(x_test)):
        pred1 = model1(x_test[i])
        pred_set_size1 = 2 * pred1 * mult_width
        pred2 = model2(x_test[i])
        pred_set_size2 = 2 * pred2 * mult_width
        for j in range(num_groups):
            if group_fn(x_test[i])[j]:
                group_num1[j] += 1
                group_num2[j] += 1
                group_pred_set_size1[j] += pred_set_size1
                group_pred_set_size2[j] += pred_set_size2

    group_pred_set_size1 = [group_pred_set_size1[j]/group_num1[j] for j in range(num_groups)]
    group_pred_set_size2 = [group_pred_set_size2[j]/group_num2[j] for j in range(num_groups)]

    br1 = np.arange(num_groups)
    br2 = [x + bar_width for x in br1]
    br3 = [x + bar_width for x in br2]
    br4 = [x + bar_width for x in br3]

    plt.figure(figsize=(10, 7))
    plt.bar(br1, marginal_conformal_size , color = 'b', width = bar_width, edgecolor = 'gray', label = 'Split-Conformal: Without groups', linewidth = 0.5)
    plt.bar(br2, group_conformal_size, color = 'm', width = bar_width, edgecolor = 'gray', label = 'Split-Conformal: With groups, conservative approach', linewidth = 0.5)
    plt.bar(br3, group_pred_set_size1, color = 'c', width = bar_width, edgecolor = 'gray', label = 'One-shot group-accurate algorithm', linewidth = 0.5)
    plt.bar(br4, group_pred_set_size2, color = 'g', width = bar_width, edgecolor = 'gray', label = 'Iterative multivalid algorithm', linewidth = 0.5)
    plt.xlabel('Group Number')
    plt.xticks(range(num_groups), range(1, num_groups+1))
    plt.ylabel('Mean prediction set size')
    # maxYval = max(group_conformal_size)
    # plt.ylim(0,maxYval + 30000)
    plt.legend()
    plt.title('Comparisons of prediction set sizes using various methods on test data')
    plt.show()