import os
import csv


def node_res_to_file(args, acc, std, earlyAcc, earlyStd):

    if not os.path.exists('./results'):
        print("Create Results File !!!")

        os.makedirs('./results')

    filename = "./results/node_res.csv"

    headerList = ["dataset", "num_hops", "num_layers", "mask_rate", "num_hidden", "batch_size", "pooler", "final-acc",
                  "final-std", "early-acc", "early-std"]

    with open(filename, "a+") as f:
        f.seek(0)
        header = f.read(7)
        if header != "dataset":
            dw = csv.DictWriter(f, delimiter=',',
                                fieldnames=headerList)
            dw.writeheader()

        line = "{},{},{},{},{},{},{},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
            args.dataset, args.num_hops, args.num_layers, args.mask_rate, args.num_hidden, args.batch_size, args.pooling, acc, std, earlyAcc, earlyStd
        )
        f.write(line)
