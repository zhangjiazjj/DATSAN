import argparse
import time
import dgl
import torch
from dataset import EllipticDataset
from utils import Measure
from utils import GeneralizedCELoss1
from model import DATSAN
from tqdm import tqdm
import csv

def train(args, device):
    elliptic_dataset = EllipticDataset(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        self_loop=True,
        reverse_edge=True,
    )

    g, node_mask_by_time = elliptic_dataset.process()
    num_classes = elliptic_dataset.num_classes

    cached_subgraph = []
    cached_labeled_node_mask = []
    for i in range(len(node_mask_by_time)):
        # we add self loop edge when we construct full graph, not here
        node_subgraph = dgl.node_subgraph(graph=g, nodes=node_mask_by_time[i])
        cached_subgraph.append(node_subgraph.to(device))
        valid_node_mask = node_subgraph.ndata["label"] >= 0
        cached_labeled_node_mask.append(valid_node_mask)

    # 统计数据集结点数
    count_train =0
    count_val = 0
    count_test =0
    for i in range(0,31):
        a = cached_subgraph[i].ndata["label"]
        count_train+= torch.sum(a == 1)
    for i in range(31,36):
        a = cached_subgraph[i].ndata["label"]
        count_val+= torch.sum(a == 1)
    for i in range(36,49):
        a = cached_subgraph[i].ndata["label"]
        count_test+= torch.sum(a == 1)

    model = DATSAN(
        input_dim=166,
        hidden_dim=700,
        fcn_dim=32,
        num_classes=2,
        device=device
    )
    num_epochs=1000
    lr=0.001
    alpha=1.5 #{1.0, 1.3, 1.5, 1.7, 1.9, 2.1     1.5
    bete=0.5 #{0.1, 0.3, 0.5, 0.7, 0.9, 1.1}     0.5
    q=0.7
    wd=0.002
    loss_class_weight = [float(w) for w in args.loss_class_weight.split(",")]
    loss_class_weight = torch.Tensor(loss_class_weight).to(device)
    model = model.to(device)
    optimizer_ad = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_gce = GeneralizedCELoss1(q=q)
    criterion = torch.nn.CrossEntropyLoss()

    # 0-33，34-39，34-48
    train_max_index = 30
    valid_max_index = 35
    test_max_index = 48
    time_window_size = args.n_hist_steps

    valid_measure = Measure(
        num_classes=2, target_class=args.eval_class_id
    )
    test_measure = Measure(
        num_classes=2, target_class=args.eval_class_id
    )

    test_res_f1 = 0
    epochs = tqdm(range(num_epochs))
    max_auc = 0
    max_data = None
    for epoch in epochs:
        model.train()
        for i in range(time_window_size, train_max_index + 1):
            g_list = cached_subgraph[i - time_window_size : i + 1]
            optimizer_ad.zero_grad()
            permute = True
            pred_org_a, pred_org_b, _, pred_aug_bcak_b, data = model(g_list,cached_labeled_node_mask[i],permute)

            labels = data.y
            loss_ce_a = criterion(pred_org_a[data.train_mask & data.label_mask],
                                  labels[data.train_mask & data.label_mask].long())
            loss_ce_b = criterion(pred_org_b[data.train_mask & data.label_mask],
                                  labels[data.train_mask & data.label_mask].long() )
            loss_ce_weight = loss_ce_b / (loss_ce_b + loss_ce_a + 1e-8)
            loss_ce_anm = criterion(pred_org_a[data.train_anm], labels[data.train_anm].long())
            loss_ce_norm = criterion(pred_org_a[data.train_norm], labels[data.train_norm].long())
            loss_ce = loss_ce_weight * (loss_ce_anm + loss_ce_norm) / 2

            loss_gce = 0.5 * criterion_gce(pred_org_b[data.train_anm], labels[data.train_anm].long()) \
                       + 0.5 * criterion_gce(pred_org_b[data.train_norm], labels[data.train_norm].long())

            loss_gce_aug = 0.5 * criterion_gce(pred_aug_bcak_b[data.aug_train_anm], data.aug_y[data.aug_train_anm].long()) \
                           + 0.5 * criterion_gce(pred_aug_bcak_b[data.aug_train_norm], data.aug_y[data.aug_train_norm].long())

            loss = alpha * loss_ce + loss_gce + bete * loss_gce_aug
            loss.backward()
            optimizer_ad.step()
        # eval
        model.eval()

        for i in range(train_max_index + 1, valid_max_index + 1):
            g_list = cached_subgraph[i - time_window_size : i + 1]
            permute = True
            _, pred_org_b, _, _, data = model(g_list,cached_labeled_node_mask[i],permute)
            pred_b = pred_org_b[data.label_mask]
            labels = data.y
            labels=labels[data.label_mask]
            valid_measure.append_measures(pred_b, labels)

        cl_precision, cl_recall, cl_f1,auc = valid_measure.get_total_measure()
        valid_measure.update_best_f1(cl_f1, epoch)
        valid_measure.reset_info()

        print(
            "Eval Epoch {} | class {} | precision:{:.4f} | recall: {:.4f} | f1: {:.4f}".format(
                epoch, args.eval_class_id, cl_precision, cl_recall, cl_f1
            )
        )

        # if cur valid f1 score is best, do test
        # if epoch == valid_measure.target_best_f1_epoch:
        print(
            "###################Epoch {} Test###################".format(
                epoch
            )
        )
        for i in range(valid_max_index + 1, test_max_index + 1):
            g_list = cached_subgraph[i - time_window_size: i + 1]
            permute = True
            _, pred_org_b, _, _, data = model(g_list, cached_labeled_node_mask[i], permute)
            pred_b = pred_org_b[data.label_mask]
            labels = data.y
            labels = labels[data.label_mask]

            test_measure.append_measures(pred_b, labels)


        (
            cl_precisions,
            cl_recalls,
            cl_f1s,
        ) = test_measure.get_each_timestamp_measure()
        for index, (sub_p, sub_r, sub_f1) in enumerate(
                zip(cl_precisions, cl_recalls, cl_f1s)
        ):
            print(
                "  Test | Time {} | precision:{:.4f} | recall: {:.4f} | f1: {:.4f}".format(
                    valid_max_index + index + 2, sub_p, sub_r, sub_f1
                )
            )

        # get each epoch measure during test.
        cl_precision, cl_recall, cl_f1,auc = test_measure.get_total_measure()
        test_measure.update_best_f1(cl_f1, epoch)
        # reset measures for next test
        test_measure.reset_info()

        test_res_f1 = cl_f1

        print(
            "  Test | Epoch {} | class {} | precision:{:.4f} | recall: {:.4f} | f1: {:.4f} | auc: {}".format(
                epoch, args.eval_class_id, cl_precision, cl_recall, cl_f1,auc

            )
        )
        if auc['auc'] > max_auc:
            max_auc = auc['auc']
            with open('roc_elliptic.csv', 'w', newline='') as file:
                writer = csv.writer(file)

                # 写入列标题
                writer.writerow(['FPR', 'TPR'])

                # 写入FPR和TPR数据
                for fpr, tpr in zip(auc['fpr'], auc['tpr']):
                    writer.writerow([fpr, tpr])

                # 单独写入AUC值
                writer.writerow(['AUC', auc['auc']])


    print(
        "Best test f1 is {}, in Epoch {}".format(
            test_measure.target_best_f1, test_measure.target_best_f1_epoch
        )
    )

if __name__ == "__main__":
    argparser = argparse.ArgumentParser("DATSAN")
    argparser.add_argument(
        "--raw-dir",
        type=str,
        default="../data/epllic",
        help="Dir after unzip downloaded dataset, which contains 3 csv files.",
    )
    argparser.add_argument(
        "--processed-dir",
        type=str,
        default="../data/process/",
        help="Dir to store processed raw data.",
    )
    argparser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID. Use -1 for CPU training.",
    )
    argparser.add_argument("--num-epochs", type=int, default=10000)
    argparser.add_argument("--n-hidden", type=int, default=256)
    argparser.add_argument("--n-layers", type=int, default=2)
    argparser.add_argument(
        "--n-hist-steps",
        type=int,
        default=1,
        help="If it is set to 5, it means in the first batch,"
        "we use historical data of 0-4 to predict the data of time 5.",
    )
    argparser.add_argument("--lr", type=float, default=0.001)
    argparser.add_argument(
        "--loss-class-weight",
        type=str,
        default="0.35,0.65",
        help="Weight for loss function. Follow the official code,"

    )
    argparser.add_argument(
        "--eval-class-id",
        type=int,
        default=1,
        help="Class type to eval. On Elliptic, type 1(illicit) is the main interest.",
    )
    argparser.add_argument(
        "--patience", type=int, default=100, help="Patience for early stopping."
    )

    args = argparser.parse_args()

    if args.gpu >= 0:
        device = torch.device("cuda:%d" % args.gpu)
    else:
        device = torch.device("cpu")

    start_time = time.perf_counter()
    train(args, device)
    print("train time is: {}".format(time.perf_counter() - start_time))
