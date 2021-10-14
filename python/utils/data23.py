import numpy as np
import pickle


def get_balanced_data(test_patient, ictal_ratio = 1.0, inter_ratio =1.0, non_ratio = 1.0):
    dir_path = '../input/chbmit'
    ictal = np.zeros((0, 1024, 23))
    inter_ictal = np.zeros((0, 1024, 23))
    non_ictal = np.zeros((0, 1024, 23))
    x_total = {"ictal": ictal, "inter_ictal": inter_ictal, "non_ictal": non_ictal}
    ratio = {"ictal": ictal_ratio, "inter_ictal": inter_ratio, "non_ictal": non_ratio}
    for pat in range(1, 24):
        if pat == test_patient:
            continue

        for mode in ["ictal", "inter_ictal", "non_ictal"]:
            for i in range(10):
                if np.random.rand() > ratio[mode]:
                    continue
                pickle_file = open("{}/{}/{}_{}.pickle".format(dir_path, mode, pat, i), "rb")
                data = pickle.load(pickle_file)
                x_total[mode] = np.concatenate((x_total[mode], data))
                pickle_file.close()

    X = np.concatenate((x_total["ictal"], x_total["inter_ictal"], x_total["non_ictal"]), axis=0)
    label = np.concatenate((np.ones(x_total["ictal"].shape[0]),
                            np.zeros(x_total["inter_ictal"].shape[0] + x_total["non_ictal"].shape[0])))
    print("Train shape: {}".format(X.shape))
    return X, label


def get_test_data(test_patient, root=''):
    dir_path = root + '../input/chbmit/{}.pickle'
    pickle_file = open(dir_path.format(test_patient), "rb")
    data = pickle.load(pickle_file)
    X = np.concatenate((data["ictal"], data["inter_ictal"], data["non_ictal"]), axis=0)
    label = np.concatenate((np.ones(data["ictal"].shape[0]), np.zeros(data["inter_ictal"].shape[0] + data["non_ictal"].shape[0])))
    pickle_file.close()
    return X, label
