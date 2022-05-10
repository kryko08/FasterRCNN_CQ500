import ast
import random
import pandas as pd
import os
import shutil


def csv_split(csv_file, im_dir, dest_path):
    """
    Function drops all but one series from CT studies,
    as multiple series from the same study are not variable and cause over-fitting of the YOLOv5 model
    :param csv_file: path to csv annotation file
    :param im_dir: CT images directory
    :param dest_path: directory for new csv files
    :return: None
    """

    # Read csv file as pandas DataFrame
    df = pd.read_csv(csv_file)

    # List all Study Instance UID
    study_list = df["StudyInstanceUID"].to_list()
    study_list = list(set(study_list))

    check = True
    while check:
        # Split studies into train, validation a test subsets
        # Shuffle list
        random.shuffle(study_list)
        # Split list into two (train, test) uneven chunks (0.7 : 0.3)
        split_points = [0, int(len(study_list) * 0.7), len(study_list)]
        split_names = ["train", "test"]
        split_studies = [study_list[split_points[i]:split_points[i + 1]] for i in range(len(split_points) - 1)]
        named_split_studies = dict(zip(split_names, split_studies))

        # Randomly choose series from each study
        # Empty dict for split scans
        split_series = {}
        for subset in named_split_studies:
            subset_studies = named_split_studies[subset]

            series_list = []
            # Extract series
            for study in subset_studies:
                study_df = df[df["StudyInstanceUID"] == study]
                series_with_duplicates = study_df["SeriesInstanceUID"].tolist()
                # remove duplicates
                series = list(set(series_with_duplicates))
                # If there are multiple series within one study, drop all but one series
                if len(series) != 0:
                    random_index = random.randint(0, len(series)-1)
                    chosen_series = series[random_index]
                    series_list.append(chosen_series)
                else:
                    series_list.append(series[0])
            split_series[subset] = series_list

        # Split SOPInstanceUID
        sop_dict = {}
        for key, value in split_series.items():
            sop_df = df[df["SeriesInstanceUID"].isin(value)]
            sop_with_duplicates = sop_df["SOPInstanceUID"].tolist()
            # Remove duplicates
            sop = list(set(sop_with_duplicates))
            for record in sop:
                sop_dict[record] = key

        train_count = {"Intraparenchymal": 0, "Subdural": 0, "Epidural": 0, "Intraventricular": 0, "Chronic": 0,
                       "Subarachnoid": 0}

        test_count = {"Intraparenchymal": 0, "Subdural": 0, "Epidural": 0, "Intraventricular": 0, "Chronic": 0,
                      "Subarachnoid": 0}

        for key, value in sop_dict.items():
            sop_df = df[df["SOPInstanceUID"] == key]
            df_col = sop_df["labelName"].tolist()[0]
            if value == "train":
                train_count[df_col] += 1
            else:
                test_count[df_col] += 1

        total_train = 0
        for key, value in train_count.items():
            print(f"{value} annotations in train {key}")
            total_train += value
        print(f"Total of {total_train} annotations in train subset")

        total_test = 0
        for key, value in test_count.items():
            print(f"{value} annotations in test {key}")
            total_test += value
        print(f"Total of {total_test} annotations in test subset")

        user_input = input("Write these annotations? [y]/[n]")
        if user_input == "y":
            check = False

    train_df = pd.DataFrame(columns=["SOPInstanceUID", "x", "y", "width", "height", "label"])
    test_df = pd.DataFrame(columns=["SOPInstanceUID", "x", "y", "width", "height", "label"])

    for ind, row in df.iterrows():
        if row["SOPInstanceUID"] in sop_dict:
            data = ast.literal_eval(row["data"])
            x = data["x"]
            y = data["y"]
            width = data["width"]
            height = data["height"]
            if sop_dict[row["SOPInstanceUID"]] == "train":
                train_df = train_df.append({"SOPInstanceUID": row["SOPInstanceUID"], "x": x, "y": y, "width": width,
                                            "height": height, "label": row["labelName"]}, ignore_index=True)
            else:
                test_df = test_df.append({"SOPInstanceUID": row["SOPInstanceUID"], "x": x, "y": y, "width": width,
                                          "height": height, "label": row["labelName"]}, ignore_index=True)

    # Write csv from pandas Dataframe
    train = "train.csv"
    test = "test.csv"
    train_df.to_csv(os.path.join(dest_path, train))
    test_df.to_csv(os.path.join(dest_path, test))

    # Copy all images to target directory
    os.mkdir(os.path.join(dest_path, "train"))
    os.mkdir(os.path.join(dest_path, "test"))

    im_dir_list = os.listdir(im_dir)
    im_dir_list = [os.path.splitext(im)[0] for im in im_dir_list]

    missing = 0
    missing_files = []
    for key, value in sop_dict.items():
        if key in im_dir_list:
            if value == "train":
                shutil.copy(os.path.join(im_dir, f"{key}.jpg"), os.path.join(dest_path, f"train/{key}.jpg"))
            else:
                shutil.copy(os.path.join(im_dir, f"{key}.jpg"), os.path.join(dest_path, f"test/{key}.jpg"))
        else:
            missing += 1
            missing_files.append(key)
    print(missing, missing_files)



