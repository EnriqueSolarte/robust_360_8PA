import os
import pandas as pd
import numpy as np
from file_utilities import create_dir, create_file
from shutil import copyfile
import sys
from file_utilities import load_obj


def filename_in_results(**kwargs):
    scene_dir = os.path.join(kwargs["results_dir"], kwargs["scene"])
    sub_dirs = [d for d in os.listdir(scene_dir) if kwargs["key"] in d]
    if len(sub_dirs) < 1:
        return None
    result = [d for d in os.listdir(os.path.join(scene_dir, sub_dirs[0])) if kwargs["ext"] in d]
    return os.path.join(scene_dir, sub_dirs[0], result[0])


def get_dict_results(**kwargs):
    scenes = os.listdir(kwargs["results_dir"])
    list_dt = []
    for scene in scenes:
        if kwargs["seq"] is None:
            results_filename = filename_in_results(scene=scene, ext=".results", **kwargs)
        else:
            results_filename = filename_in_results(scene=scene + "/{}".format(kwargs["seq"]), ext=".results", **kwargs)
        if results_filename is None:
            continue
        list_dt.append(pd.DataFrame(load_obj(results_filename)))
        print("scene evaluated: {} - {} - {}".format(scene, len(list_dt[-1]), kwargs["key"]))
    if len(list_dt) < 1:
        exit(0)
    kwargs["results"] = pd.concat(list_dt)
    if kwargs.get("save", False):
        kwargs["save_results_dir"] = os.path.join(
            os.path.dirname(kwargs["results_dir"]),
            "results_{}".format(kwargs["key"]),
        )
        create_dir(kwargs["save_results_dir"], delete_previous=False)
        kwargs["results"].to_csv(kwargs["save_results_dir"] + "/results.csv", index=False)
    return kwargs


def get_results(**kwargs):
    scenes = os.listdir(kwargs["results_dir"])
    list_dt = []
    for scene in scenes:
        results_filename = filename_in_results(scene=scene + "/{}".format(kwargs["seq"]), ext=".txt", **kwargs)
        if results_filename is None:
            continue
        if kwargs.get("save", False):
            kwargs["save_results_dir"] = os.path.join(
                os.path.dirname(kwargs["results_dir"]),
                "results_{}".format(kwargs["key"]),
            )
            create_dir(kwargs["save_results_dir"], delete_previous=False)
            copyfile(results_filename,
                     os.path.join(kwargs["save_results_dir"],
                                  os.path.split(results_filename)[1]))
        list_dt.append(pd.read_csv(results_filename))
        print("scene evaluated: {} - {}".format(scene, len(list_dt[-1])))
    kwargs["results"] = pd.concat(list_dt)
    if kwargs.get("save", False):
        kwargs["results"].to_csv(kwargs["save_results_dir"] + "/results.txt")
    return kwargs


def eval_results(quantile, **kwargs):
    def print_results():
        if "time" in header:
            scale = scale_time
        else:
            scale = 1
        print("Q{} {},{},{} ".format(
            int(quantile * 100),
            header,
            np.quantile(data / scale, quantile),
            np.std(data / scale)
        ))

    results = kwargs["results"]
    try:
        scale_time = results["time_8pa"].values
    except:
        scale_time = 1
    if kwargs.get("save", False):
        create_file(kwargs["save_results_dir"] + "/summary_results_Q{}.csv".format(int(quantile * 100)),
                    delete_previous=True)
    all_headers = [h for h in results.columns if "kf" not in h]
    # all_headers.sort(key=lambda x: x[-1:-5])
    original_stdout = sys.stdout
    i = 0
    if kwargs.get("save", False):
        with open(kwargs["save_results_dir"] + "/summary_results_Q{}.csv".format(int(quantile * 100)), "w") as f:
            for header in all_headers:
                data = results[header].values
                for std in [f, original_stdout]:
                    sys.stdout = std
                    if i == 0:
                        print("Description, evaluation, std")
                        i += 1
                    print_results()
            for std in [f, original_stdout]:
                sys.stdout = std
                print("8PA time,{},{}".format(np.median(scale_time), np.std(scale_time)))
            # print(kwargs.get("extra", ""))
            sys.stdout = original_stdout
    else:
        for header in all_headers:
            data = results[header].values
            print_results()
        print("8PA time,{},{}".format(np.median(scale_time), np.std(scale_time)))
        # print(kwargs.get("extra", ""))


if __name__ == '__main__':
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    settings = dict(
        results_dir=results_dir,
        seq="0",
        # key="all_scenes_ransac_opt"
        # key="2020-10-10.19.55.25",
        # key="2020-10-10.18.47.37",
        key="2020-10-11.8.11.53",
        save=False
    )
    kwargs = get_results(**settings)
    eval_results(**kwargs)
