from model.utils1 import data
def main():
    input1, gt, data_radius, name = data.load_patch_data()

    print(data_radius)

if __name__ == "__main__":
    main()
