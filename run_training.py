# head dims:512,512,512,512,512,512,512,512,128
# code is basicly:https://github.com/google-research/deep_representation_one_class
from pathlib import Path
from tqdm import tqdm
import datetime
import argparse

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


from dataset1 import MVTecAT, Repeat
from cutpaste1 import CutPasteNormal,CutPasteScar, CutPaste3Way, CutPasteUnion, cut_paste_collate_fn,CutPasteMask3Way,CutPasteMaskNormal,CutPasteMaskscar
from model1 import ProjectionNet
from eval import eval_model
from utils import str2bool,set_seed, seed_worker
import pandas as pd
import json  
from pathlib import Path
import argparse
import csv
import random  
import numpy as np
import os
import optuna
from optuna.trial import TrialState
# tensorboard --logdir logdirs
from cafl_loss import CAFLoss
from asl_focal_loss import create_loss, Cyclical_FocalLoss
def set_seed(seed: int = 42, deterministic: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        except Exception:
            pass

def seed_worker(worker_id: int):
    # Ensure each worker has reproducible but different sub-seeds on each start
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def run_training(data_type="screw",
                 model_dir="models",
                 epochs=256,
                 pretrained=True,
                 test_epochs=10,
                 freeze_resnet=20,
                 learninig_rate=0.03,
                 optim_name="SGD",
                 batch_size=64,
                 head_layer=2,
                 cutpate_type=CutPasteNormal,
                 device = "cuda",
                 workers=8,
                 size = 256,                 
                 seed: int = 42,
                 deterministic: bool = False,
                 save_name: str | None = None,
                 report_fn=None,
                 alpha=0.5,
            beta=1e-3,
            tau_w=0.5,
            gamma_pos=0.0,
            gamma_neg=4.0,
            gamma_hc=2.0,  # If set to 0, it will revert to ASLSingleLabel
            factor=2.0,    # It must be > 1, fc > 1
            ):

    torch.multiprocessing.freeze_support()
    # TODO: Use script params for hyperparameters
    # Temperature hyperparameter is currently not used

    set_seed(seed, deterministic)
    temperature = 0.2

    weight_decay = 0.00003
    momentum = 0.9
    # TODO: Use f-strings for the date as well
    # model_name = f"model-{data_type}" + '-{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now() )
    # Generate save name (new)
    if save_name is not None:
        model_name = save_name
    else:
        model_name = f"model-{data_type}" + '-{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now())

    csv_path0 = Path("eval") / model_name 
    csv_path = Path("eval") / model_name / "training_params.csv"
    csv_path0.mkdir(parents=True, exist_ok=True)
    params_fields = [
        "model_name", "data_type", "epochs", "pretrained", "test_epochs",
        "freeze_resnet", "learning_rate", "optim_name", "batch_size",
        "head_layer", "cutpaste_type", "device", "workers", "size", "timestamp"
    ]
    # Prepare a dict of current training parameters
    current_params = {
        "model_name": model_name,
        "data_type": data_type,
        "epochs": epochs,
        "pretrained": pretrained,
        "test_epochs": test_epochs,
        "freeze_resnet": freeze_resnet,
        "learning_rate": learninig_rate,
        "optim_name": optim_name,
        "batch_size": batch_size,
        "head_layer": head_layer,
        "cutpaste_type": cutpate_type,  # Save class name
        "device": device,
        "workers": workers,
        "size": size,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    }
    # If file does not exist, write header first
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=params_fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow(current_params)

    # Augmentation:
    min_scale = 1

    # Create Training Dataset and Dataloader
    after_cutpaste_transform = transforms.Compose([])
    after_cutpaste_transform.transforms.append(transforms.ToTensor())
    after_cutpaste_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225]))

    train_transform = transforms.Compose([])
    # train_transform.transforms.append(transforms.RandomResizedCrop(size, scale=(min_scale,1)))
    train_transform.transforms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
    # train_transform.transforms.append(transforms.GaussianBlur(int(size/10), sigma=(0.1,2.0)))
    train_transform.transforms.append(transforms.Resize((size,size)))
    train_transform.transforms.append(cutpate_type(transform = after_cutpaste_transform))
    # train_transform.transforms.append(transforms.ToTensor())

    train_data = MVTecAT("Data", data_type, transform = train_transform, size=int(size * (1/min_scale)))

    g = torch.Generator()
    g.manual_seed(seed)

    dataloader = DataLoader(Repeat(train_data, 3000), batch_size=batch_size, drop_last=True,
                            shuffle=True, num_workers=workers, collate_fn=cut_paste_collate_fn,
                            persistent_workers=True, pin_memory=True, prefetch_factor=5, worker_init_fn=seed_worker, generator=g)

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(Path("logdirs") / model_name)

    # Create Model:
    head_layers = [512]*head_layer+[128]
    if cutpate_type == CutPaste3Way:
        num_classes = 3
    elif cutpate_type == CutPasteMask3Way:
        num_classes = 3
    else:
        num_classes = 2
    model = ProjectionNet(pretrained=pretrained, head_layers=head_layers, num_classes=num_classes)
    model.to(device)

    if freeze_resnet > 0 and pretrained:
        model.freeze_resnet()

    #loss_fn = torch.nn.CrossEntropyLoss()
    # Build CFL (ASLSingleLabel if gamma_hc==0, otherwise Cyclical_FocalLoss)
    base_cfl = create_loss(
        gamma_pos=gamma_pos,
        gamma_neg=gamma_neg,
        gamma_hc=gamma_hc,
        epochs=epochs,
        factor=factor
    )

    # Wrap into CAFL
    loss_fn = CAFLoss(
        base_cfl=base_cfl,
        alpha=alpha,
        beta=beta,
        tau_w=tau_w
    )


    if optim_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learninig_rate, momentum=momentum,  weight_decay=weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, epochs)
        # scheduler = None
    elif optim_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learninig_rate, weight_decay=weight_decay)
        scheduler = None
    else:
        print(f"ERROR unkown optimizer: {optim_name}")
    results = [] 
    best_auc = 0
    step = 0
    num_batches = len(dataloader)

    def get_data_inf():
        while True:
            for out in enumerate(dataloader):
                yield out

    dataloader_inf = get_data_inf()

    # From paper: "Note that, unlike conventional definition for an epoch,
    #             we define 256 parameter update steps as one epoch."
    best_auc_optuna = -1.0  
    for step in tqdm(range(epochs)):
        epoch = int(step / 1)
        if epoch == freeze_resnet:
            model.unfreeze()
        
        batch_embeds = []
        batch_idx, data = next(dataloader_inf)
        xs = [x.to(device) for x in data]

        # Zero the parameter gradients
        optimizer.zero_grad()

        xc = torch.cat(xs, axis=0)
        embeds, logits ,cost_sspcab = model(xc)
        
#         embeds = F.normalize(embeds, p=2, dim=1)
#         embeds1, embeds2 = torch.split(embeds,x1.size(0),dim=0)
#         ip = torch.matmul(embeds1, embeds2.T)
#         ip = ip / temperature
#
#         y = torch.arange(0,x1.size(0), device=device)
#         loss = loss_fn(ip, torch.arange(0,x1.size(0), device=device))

        # Compute labels
        y = torch.arange(len(xs), device=device)
        y = y.repeat_interleave(xs[0].size(0))
        
        #loss = loss_fn(logits, y)#CE loss
        #CAFL
        loss, loss_cfl, loss_adv, loss_ssp = loss_fn(
        logits=logits,
        y=y,
        epoch=epoch,                 # Cyclical_FocalLoss need epoch
        cost_sspcab=cost_sspcab
    )
        
        # Regularize weights:
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step(epoch)
        
        writer.add_scalar('loss', loss.item(), step)
        
#         predicted = torch.argmax(ip,axis=0)
        predicted = torch.argmax(logits,axis=1)
#         print(logits)
#         print(predicted)
#         print(y)
        accuracy = torch.true_divide(torch.sum(predicted==y), predicted.size(0))
        writer.add_scalar('acc', accuracy, step)
        if scheduler is not None:
            writer.add_scalar('lr', scheduler.get_last_lr()[0], step)
        
        # Save embeddings for validation:
        if test_epochs > 0 and epoch % test_epochs == 0:
            batch_embeds.append(embeds.cpu().detach())

        writer.add_scalar('epoch', epoch, step)

        # Run tests
        if test_epochs > 0 and epoch % test_epochs == 0:
            # Run AUC calculation
            # TODO: Create dataset only once.
            # TODO: Train predictor here or in the model class itself. Should not be in the eval part.
            # TODO: We might not want to use the training data because of dropout etc., but it should give an indication of the model performance???
            # batch_embeds = torch.cat(batch_embeds)
            # print(batch_embeds.shape)
            model.eval()
            roc_auc = eval_model(model_name, defect_type=data_type, device=device,
                                save_plots=True,
                                size=size,
                                show_training_data=False,
                                model=model,
                                cutpate_type=cutpate_type,
                                seed=seed,
                                deterministic=deterministic, head_layer=head_layer)
                                # train_embed=batch_embeds)
            model.train()
            # [OPTUNA] Report to the hyperparameter tuner
            if report_fn is not None:
                report_fn(step, float(roc_auc))

            if roc_auc > best_auc_optuna:
                best_auc_optuna = roc_auc

            writer.add_scalar('eval_auc', roc_auc, step)
            print('AUC of the network now:%f '% roc_auc)
            print('AUC of best:%f ' % best_auc)
            results.append({
            "defect_type": data_type,
            "auc_score": roc_auc
            })

            if roc_auc > best_auc:
                # print(roc_auc)
                print('AUC of the network best:%f ' %roc_auc)
                best_auc = roc_auc
                # torch.save(model.state_dict(), model_dir_best / f"{model_name}.tch")

    df = pd.DataFrame(results)

    csv_path = Path("eval") / model_name / "auc_results.csv"
    df.to_csv(csv_path, index=False)
    # torch.save(model.state_dict(), model_dir / f"{model_name}.tch")
    # Save final model
    final_path = model_dir / f"{model_name}.tch"
    torch.save(model.state_dict(), final_path)
    return best_auc_optuna, str(final_path)

# tensorboard --logdir logdirs
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training defect detection as described in the CutPaste Paper.')
    parser.add_argument('--type', default="grid",
                        help='MVTec defection dataset type to train seperated by , (default: "all": train all defect types)')

    parser.add_argument('--epochs', default=400, type=int,
                        help='number of epochs to train the model , (default: 256)')
    
    parser.add_argument('--model_dir', default="models",
                        help='output folder of the models , (default: models)')
    
    parser.add_argument('--no-pretrained', dest='pretrained', default=True, action='store_false',
                        help='use pretrained values to initalize ResNet18 , (default: True)')
    
    parser.add_argument('--test_epochs', default=10, type=int,
                        help='interval to calculate the auc during trainig, if -1 do not calculate test scores, (default: 10)')                  

    parser.add_argument('--freeze_resnet', default=20, type=int,
                        help='number of epochs to freeze resnet (default: 20)')
    
    parser.add_argument('--lr', default=0.0005, type=float,
                        help='learning rate (default: 0.03)')

    parser.add_argument('--optim', default="sgd",
                        help='optimizing algorithm values:[sgd, adam] (dafault: "sgd")')

    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch size, real batchsize is depending on cut paste config normal cutaout has effective batchsize of 2x batchsize (dafault: "64")')   

    parser.add_argument('--head_layer', default=2, type=int,
                    help='number of layers in the projection head (default: 1)')
    
    parser.add_argument('--variant', default="masknormal", choices=['normal', 'scar', '3way', 'union', "mask3way", "masknormal", "maskscar"],
                        help='cutpaste variant to use (dafault: "3way")')
    
    parser.add_argument('--cuda', default=True, type=str2bool,
                    help='use cuda for training (default: False)')
    
    parser.add_argument('--workers', default=4, type=int,
                        help="number of workers to use for data loading (default:8)")

    parser.add_argument('--seed', default=42, type=int,
                        help='global random seed (default: 42)')
    parser.add_argument('--deterministic', default=False, type=str2bool,
                        help='force deterministic algorithms (may slow down)')
    parser.add_argument('--alpha', default=0.5, type=float,
                   )
    parser.add_argument('--beta', default=1e-3, type=float,
                   )
    parser.add_argument('--tau_w', default=0.5, type=float,
                   )
    parser.add_argument('--gamma_pos', default=0.0, type=float,
                   )
    parser.add_argument('--gamma_neg', default=4.0, type=float,
                   )
    parser.add_argument('--gamma_hc', default=2.0, type=float,
                   )
    parser.add_argument('--factor', default=2.0, type=float,
                   )
    args = parser.parse_args()
    print(args)

    all_types = ['bottle',
             'cable',
             'capsule',
             'carpet',
             'grid',
             'hazelnut',
             'leather',
             'metal_nut',
             'pill',
             'screw',
             'tile',
             'toothbrush',
             'transistor',
             'wood',
             'zipper']

    if args.type == "all":
        types = all_types
    else:
        types = args.type.split(",")
    
    variant_map = {'normal': CutPasteNormal, 'scar': CutPasteScar, '3way': CutPaste3Way, 'union': CutPasteUnion,
                   'mask3way': CutPasteMask3Way, "masknormal": CutPasteMaskNormal, "maskscar": CutPasteMaskscar}
    variant = variant_map[args.variant]
    
    device = "cuda" if args.cuda else "cpu"
    print(f"using device: {device}")
    
    # Create model dir
    Path(args.model_dir).mkdir(exist_ok=True, parents=True)
    # Save config
    with open(Path(args.model_dir) / "run_config.txt", "w") as f:
        f.write(str(args))
    
    for data_type in types:
        print(f"training {data_type}")
        best_auc, path = run_training(data_type,
                     model_dir=Path(args.model_dir),
                     epochs=args.epochs,
                     pretrained=args.pretrained,
                     test_epochs=args.test_epochs,
                     freeze_resnet=args.freeze_resnet,
                     learninig_rate=args.lr,
                     optim_name=args.optim,
                     batch_size=args.batch_size,
                     head_layer=args.head_layer,
                     device=device,
                     cutpate_type=variant,
                     workers=args.workers,
                     seed=args.seed,
                     deterministic=args.deterministic,
                     alpha=args.alpha,
                     beta=args.beta,
                     tau_w=args.tau_w,
                     gamma_pos=args.gamma_pos,
                     gamma_neg=args.gamma_neg,
                     gamma_hc=args.gamma_hc,
                     factor=args.factor
                     )
        print("best_auc:", best_auc, "path:", path)   
