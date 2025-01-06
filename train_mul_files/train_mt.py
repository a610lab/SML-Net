import argparse
from utils.myload_data import *
import torch
from torch import optim
from torchvision import transforms
from torch.utils.data import RandomSampler
from utils.my_dataset import MyDataSet
from model_file import deeplab_mt
from utils.train_module import train_one_epoch_mt, evaluate

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_model(backbone_name):
    model = deeplab_mt.DeepLabV3Plus(model_name=backbone_name)
    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)
    seed_torch(args.seed)

    data_transform = {
        "train": transforms.Compose([
            transforms.Resize([96, 128]),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip()
            # transforms.Normalize([0.246890, 0.257212, 0.279224], [0.228331, 0.236681, 0.252307])
        ]),
        "val": transforms.Compose([
            transforms.Resize([96, 128]),
            transforms.ToTensor()
            # transforms.Normalize([0.244994, 0.255258, 0.277088], [0.227103, 0.235438, 0.251031])
        ])}

    train_u_dataset = MyDataSet("../data_csv/semi_unlabeled_20.csv",
                                transform=data_transform['train']
                                )
    train_l_dataset = MyDataSet("../data_csv/semi_labeled_20.csv",
                                transform=data_transform['train'],
                                nsample=len(train_u_dataset.images_paths)
                                )
    val_dataset = MyDataSet("../data_csv/validate_data.csv",
                            transform=data_transform['val']
                            )
    train_num = len(train_u_dataset) + len(train_l_dataset)
    print("using {} images for training.".format(train_num))

    train_l_loader = torch.utils.data.DataLoader(train_l_dataset,
                                                 shuffle=True,
                                                 batch_size=args.labeled_batch_size,
                                                 drop_last=True,
                                                 pin_memory=True,
                                                 num_workers=args.num_workers)
    train_u_loader = torch.utils.data.DataLoader(train_u_dataset,
                                                 shuffle=True,
                                                 batch_size=args.unlabeled_batch_size,
                                                 drop_last=True,
                                                 pin_memory=True,
                                                 num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.labeled_batch_size + args.unlabeled_batch_size,
                                             pin_memory=True,
                                             num_workers=args.num_workers)

    # Write the results to the CSV file
    fid_csv = open('../train_mul_save_file/' + args.model_name + '_' + str(args.seed) + '_20_mt_728.csv', 'w',
                   encoding='utf-8')
    csv_writer = csv.writer(fid_csv)
    csv_writer.writerow(["parameters", "batch_size", "learning_rate", "epochs"])
    csv_writer.writerow([" ", args.labeled_batch_size, args.lr, args.epochs])
    csv_writer.writerow(["epoch", "train_loss", "train_acc", "val_acc", ])

    # model_pth = '../train_mul_save_file/densenet_128_20_mt_1_100_872.pth'
    # ckpt = torch.load(model_pth, map_location=device)

    # 创建模型  构造优化器
    model_s = get_model(args.model_name).to(device)
    # model_s.load_state_dict(ckpt, strict=False)
    model_t = get_model(args.model_name).to(device)
    pg = [p for p in model_s.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=args.lr)

    best_acc = 0
    for epoch in range(args.epochs):
        train_loss = train_one_epoch_mt(model_s=model_s,
                                        model_t=model_t,
                                        optimizer=optimizer,
                                        labeled_data_loader=train_l_loader,
                                        unlabeled_data_loader=train_u_loader,
                                        device=device,
                                        epoch=epoch)
        # validate
        val_acc, val_dsc = evaluate(model=model_t,
                                    data_loader=val_loader,
                                    device=device,
                                    epoch=epoch)

        csv_writer.writerow(
            [epoch, train_loss, val_acc, val_dsc])
        if best_acc < val_acc:
            best_acc = val_acc
            # if epoch > 150:
            print("best acc is {:.3f}, dsc is {:.3f}".format(val_acc, val_dsc))
            torch.save(model_t.state_dict(),
                        '../train_mul_save_file/' + args.model_name + '_' + str(args.seed) + '_20_mt_728.pth')

    print()
    torch.cuda.empty_cache()
    fid_csv.close()
    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='densenet')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--unlabeled_batch_size', type=int, default=8)
    parser.add_argument('--labeled_batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--seed', type=int, default=128)
    opt = parser.parse_args()
    main(opt)
