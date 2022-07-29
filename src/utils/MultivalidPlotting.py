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
