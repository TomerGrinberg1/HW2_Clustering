import sys
import data


def main(argv):
    print("Part A: ")
    df = data.load_data(argv[1])
    data.add_new_columns(df)
    data.data_analysis(df)


if __name__ == '__main__':
    main(sys.argv)
