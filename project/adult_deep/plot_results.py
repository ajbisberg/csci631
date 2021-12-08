import matplotlib.pyplot as plt

from data import DatasetChoice

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return smoothed


def save_results_fig(dataset_choice, fname, results):
    eps_list = sorted(list(results.keys()))
    acc_list = [results[ep][0] for ep in eps_list]
    dem_parity_list_gender = [results[ep][1] for ep in eps_list]
    dem_parity_list_race = [results[ep][2] for ep in eps_list]
    eq_opp_list_gender = [results[ep][3] for ep in eps_list]
    eq_opp_list_race = [results[ep][4] for ep in eps_list]

    plt.suptitle('DP-SGD Classifier Accuracy and Fairness Metrics vs Epsilon')
    plt.title('Dashed and Solid Lines for Standard and Private Classifiers', fontsize='small')
    if dataset_choice == DatasetChoice.ADULT:
        acc, dp_g, dp_r, eo_g, eo_r = (0.79104478, 0.30066256, 0.50714886, 0.26636978, 0.5203459)
    elif dataset_choice == DatasetChoice.HOUSING:
        acc, dp_g, dp_r, eo_g, eo_r = (0.68441297, 0.89828961, 0.7055424,  0.84102496, 0.61121419)
    plt.axhline(acc, color='r', linestyle='--')
    plt.axhline(dp_g, color='g', linestyle='--')
    plt.axhline(dp_r, color='b', linestyle='--')
    plt.axhline(eo_g, color='m', linestyle='--')
    plt.axhline(eo_r, color='y', linestyle='--')

    plt.semilogx(eps_list, smooth(acc_list,0.6), label='Accuracy', color='r')
    plt.semilogx(eps_list, smooth(dem_parity_list_gender,0.6), label='Dem Par Gender', color='g')
    plt.semilogx(eps_list, smooth(dem_parity_list_race,0.6), label='Dem Par Race', color='b')
    plt.semilogx(eps_list, smooth(eq_opp_list_gender,0.6), label='Eq Opp Gender', color='m')
    plt.semilogx(eps_list, smooth(eq_opp_list_race,0.6), label='Eq Opp Race', color='y')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim(0,1)
    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy / Fairness')

    plt.savefig(fname + '.png', dpi=150, bbox_inches='tight')
