import os.path as osp
import os
from .evaluator import Eval_thread
from .dataloader import EvalDataset


def evaluate(args):

    pred_dir = args.save_test_path_root
    output_dir = args.save_dir

    method_names2 = args.methods1.split('+')

    threads = []
    gt_dir_all = []

    test_paths_total = [args.VSODtest_paths]
    task_total = ['VSOD']
    data_root = [args.VSODdata_root]

    for k in range(len(test_paths_total)):
        test_paths = test_paths_total[k].split('+')
        task = task_total[k]
        gt_dir = data_root[k]
        for dataset_setname in test_paths:

            dataset_name = dataset_setname

            for method2 in method_names2:
                if task == "VSOD" or task == "DVSOD":
                    pred_dir_all = osp.join(pred_dir, task, dataset_name, method2)
                    gt_dir_all = osp.join(gt_dir, dataset_setname)

                else:
                    if dataset_name in ['NJUD', 'NLPR', 'DUTLF-Depth', 'ReDWeb-S']:
                        gt_dir_all = osp.join(osp.join(gt_dir, dataset_setname))
                    elif dataset_name in ['VT5000']:
                        gt_dir_all = osp.join(osp.join(gt_dir, dataset_setname))
                    else:
                        gt_dir_all = osp.join(osp.join(gt_dir, dataset_setname))
                    pred_dir_all = osp.join(pred_dir, task, dataset_name, method2)

            loader = EvalDataset(pred_dir_all, gt_dir_all, task)
            thread = Eval_thread(loader, method2, dataset_setname, output_dir, cuda=True)
            threads.append(thread)
    for thread in threads:
        print(thread.run())