import os
import csv


def graph_results_to_file(args, acc, std):

    if not os.path.exists('./results'):
        print("Create Results File !!!")

        os.makedirs('./results')

    filename = "./results/{}.csv".format(args.dataset)

    headerList = ["dataset", "enc_type",  "num_layers", "pe_dim",
                  "trans_num_layers", "mask_rate", "num_subgraph",
                  "subgraph_lambda", "dec_type", "dec_num_layers",
                  "acc", "std"]

    with open(filename, "a+") as f:
        f.seek(0)
        header = f.read(7)
        if header != "dataset":
            dw = csv.DictWriter(f, delimiter=',',
                                fieldnames=headerList)
            dw.writeheader()

        line = "{},{},{},{},{},{},{},{},{},{},{:.4f},{:.4f}\n".format(
            args.dataset, args.enc_type, args.num_layers, args.pe_dim,
            args.trans_num_layers, args.mask_rate, args.num_subgraph,
            args.subgraph_lambda, args.dec_type, args.dec_num_layers, acc, std
        )
        f.write(line)
