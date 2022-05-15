import sys

import clustering
import data


def main(argv):
    # print("Part A: ")
    # df = data.load_data(argv[1])
    # data.add_new_columns(df)
    # data.data_analysis(df)

    print("Part B: ")
    ks = [2, 3, 5]
    df = data.load_data(argv[1])
    df_as_np = clustering.transform_data(df, ['cnt', 'hum'])
    labels,centroids = 0, 0
    for k in ks:
        print(f'k = {k}')
        labels, centroids = clustering.kmeans(df_as_np, k)  # printing the centroids
        print()
        path = f'C:/Users/user/PycharmProjects/HW2_Clustering/k_means_{k}.png'
        clustering.visualize_results(df_as_np, labels, centroids, path)



if __name__ == '__main__':
    main(sys.argv)
