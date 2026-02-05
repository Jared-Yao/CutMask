from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from dataset1 import MVTecAT
from cutpaste1 import CutPaste
from model1 import ProjectionNet
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from cutpaste1 import CutPasteNormal, CutPasteScar, CutPaste3Way, CutPasteUnion, cut_paste_collate_fn, CutPasteMask3Way, CutPasteMaskNormal, CutPasteMaskscar
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
import numpy as np
from collections import defaultdict
from density import GaussianDensitySklearn, GaussianDensityTorch
import pandas as pd
from utils import str2bool, set_seed, seed_worker

import os, random
from sklearn.preprocessing import StandardScaler

import random


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
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

test_data_eval = None
test_transform = None
cached_type = None
test_data_eval = None
test_transform = None
cached_type = None

def get_train_embeds(model, size, defect_type, transform, device, seed: int = 42):
    # Train data / train KDE
    test_data = MVTecAT("Data", defect_type, size, transform=transform, mode="train")
  
    g = torch.Generator(); g.manual_seed(seed)

    dataloader_train = DataLoader(test_data, batch_size=64,
                            shuffle=False, num_workers=0, generator=g)
    train_embed = []
    with torch.no_grad():
        for x in dataloader_train:
            embed, logit,_ = model(x.to(device))

            train_embed.append(embed.cpu())
    train_embed = torch.cat(train_embed)
    return train_embed

def eval_model(modelname, defect_type, device="cuda", save_plots=True, size=256, show_training_data=True, model=None, train_embed=None, head_layer=1, density=GaussianDensityTorch(),cutpate_type=CutPasteNormal, seed: int = 42, deterministic: bool = True):

    set_seed(seed, deterministic)
    # Create test dataset
    global test_data_eval,test_transform, cached_type

    # TODO: Cache is only useful during training. Do we need it here?
    if test_data_eval is None or cached_type != defect_type:
        cached_type = defect_type
        test_transform = transforms.Compose([])
        test_transform.transforms.append(transforms.Resize((size,size)))
        test_transform.transforms.append(transforms.ToTensor())
        test_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225]))
        test_data_eval = MVTecAT("Data", defect_type, size, transform = test_transform, mode="test")

    g = torch.Generator(); g.manual_seed(seed)
    dataloader_test = DataLoader(test_data_eval, batch_size=64,
                                    shuffle=False, num_workers=0,generator=g)

    # Create model
    if model is None:
        print(f"loading model {modelname}")
        head_layers = [512]*head_layer+[128]
        print(head_layers)
        weights = torch.load(modelname)
        classes = weights["out.weight"].shape[0]
        model = ProjectionNet(pretrained=False, head_layers=head_layers, num_classes=classes)
        model.load_state_dict(weights)
        model.to(device)
        model.eval()

    # Get embeddings for test data
    labels = []
    embeds = []

    with torch.no_grad():
        for x, label in dataloader_test:
            embed, logit ,_ = model(x.to(device))

            # Save
            embeds.append(embed.cpu())
            labels.append(label.cpu())
    labels = torch.cat(labels)
    embeds = torch.cat(embeds)

    if train_embed is None:
        train_embed = get_train_embeds(model, size, defect_type, test_transform, device,seed=seed)  # same as test_eval

    # Normalize embeddings
    embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
    train_embed = torch.nn.functional.normalize(train_embed, p=2, dim=1)

    # Create eval plot directory
    if save_plots:
        eval_dir = Path("eval") / modelname
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot t-SNE
        # Also show some of the training data

        show_training_data = True
        if show_training_data:
            # Augmentation setting
            # TODO: Put all of this into a separate function usable for both training and evaluation.
            #       It's pretty ugly to just copy-paste the code here.
            min_scale = 0.5

            # Create Training Dataset and Dataloader
            after_cutpaste_transform = transforms.Compose([])
            test_transform.transforms.append(transforms.Resize((size,size)))
            after_cutpaste_transform.transforms.append(transforms.ToTensor())
            after_cutpaste_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]))

            train_transform = transforms.Compose([])
            # train_transform.transforms.append(transforms.RandomResizedCrop(size, scale=(min_scale,1)))
            # train_transform.transforms.append(transforms.GaussianBlur(int(size/10), sigma=(0.1,2.0)))
            train_transform.transforms.append(cutpate_type(transform=after_cutpaste_transform))
            # train_transform.transforms.append(transforms.ToTensor())
            #
            """train_embeds,train_labels = create_train_tsne(model, size, defect_type, train_transform, device,cutpate_type,seed=seed)


            # For t-SNE we encode training data as 2, and augmented data as 3
            #test label + train label
            tsne_labels = torch.cat([labels, train_labels])
            tsne_embeds = torch.cat([embeds, train_embeds])"""
        else:
            tsne_labels = labels
            tsne_embeds = embeds
        # plot_tsne(tsne_labels, tsne_embeds, eval_dir / "tsne.png")
    else:
        eval_dir = Path("unused")
    
    print(f"using density estimation {density.__class__.__name__}")
    density.fit(train_embed)
    distances = density.predict(embeds)
    # TODO: Set a threshold on Mahalanobis distances and use "real" probabilities

    roc_auc = plot_roc(labels, distances, eval_dir / "roc_plot.png", modelname=modelname, save_plots=save_plots)
    
    return roc_auc
    

def plot_roc(labels, scores, filename, modelname="", save_plots=False):

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC
    if save_plots:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver operating characteristic {modelname}')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig(filename)
        plt.close()

    return roc_auc



def create_train_tsne(model, size, defect_type, transform, device,cutpate_type,seed):
    # Train data / train KDE
    test_data = MVTecAT("Data", defect_type, size, transform=transform, mode="train")
    g2 = torch.Generator(); g2.manual_seed(seed)
    dataloader_train = DataLoader(test_data, batch_size=32,
                            shuffle=True,collate_fn=cut_paste_collate_fn,persistent_workers=True,worker_init_fn=seed_worker,generator=g2)

    # Inference on training data
    train_labels = []
    train_embeds = []
    with torch.no_grad():
        # Generate labels:
        y1 = torch.tensor([0])  # origin
        y2 = torch.tensor([2])  # cutpastenormal
        y3 = torch.tensor([3])  # cutpastescar
        y4 = torch.tensor([4])  # cutmasknormal
        y5 = torch.tensor([5])  # cutmaskscar
        for x in dataloader_train:

            if cutpate_type == CutPaste3Way:
                x1, x2, x3 = x
                x = torch.cat([x1,x2, x3])
                y3 = y3.repeat_interleave(x3.size(0))
                y1 = y1.repeat_interleave(x1.size(0))
                y2 = y2.repeat_interleave(x2.size(0))
                y = torch.cat([y1,y2, y3])
         
            elif cutpate_type == CutPasteNormal:
                x1,x2 =x 
                x=torch.cat([x1,x2])
                y1 = y1.repeat_interleave(x1.size(0))     
                y2 = y2.repeat_interleave(x2.size(0))   
                y = torch.cat([y1,y2])   
            elif cutpate_type == CutPasteScar:
                x1, x3 = x
                x = torch.cat([x1,x3])
                y1 = y1.repeat_interleave(x1.size(0))
                y3 = y3.repeat_interleave(x3.size(0))
                y = torch.cat([ y1,y3])
            elif cutpate_type == CutPasteMaskNormal:
                x1,x4 =x 
                x=torch.cat([x1,x4])
                y1 = y1.repeat_interleave(x1.size(0))     
                y4 = y4.repeat_interleave(x4.size(0))   
                y = torch.cat([y1,y4])    
            elif cutpate_type == CutPasteMaskscar:
                x1,x5 =x 
                x=torch.cat([x1,x5])
                y1 = y1.repeat_interleave(x1.size(0))     
                y5 = y5.repeat_interleave(x5.size(0))   
                y = torch.cat([y1,y5])        

            # x = torch.cat([x1, x2], axis=0)
            # embed, logit,cost_sspcab = model(x.to(device))
            embed, logit,_ = model(x.to(device))

            # y = y.to(dtype=torch.int).flatten()
            # Save
            train_embeds.append(embed.cpu())
            train_labels.append(y.cpu())
            # Only take a small subset of data
            break
    train_labels = torch.cat(train_labels)
    train_embeds = torch.cat(train_embeds)

    # train_embeds = torch.nn.functional.normalize(train_embeds, p=2, dim=1)
    return train_embeds,train_labels

def plot_tsne(labels, embeds, filename):
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=1000)
    embeds, labels = shuffle(embeds, labels)
    tsne_results = tsne.fit_transform(embeds)
    fig, ax = plt.subplots(1, figsize=(10, 8))  # Adjust figure size to better display the legend
    
    # Define a base color list (extend if needed)
    colormap = ["b", "r", "g", "c", "y", "m", "k", "orange", "purple", "pink"]
    
    # Get all unique labels to ensure all classes are covered
    unique_labels = sorted(torch.unique(labels).numpy())
    
    # Check whether colors are sufficient; extend if not (avoid index errors)
    if len(unique_labels) > len(colormap):
        # Dynamically add colors using matplotlib's built-in color cycle
        from matplotlib import cm
        extra_colors = [cm.tab10(i) for i in range(len(colormap), len(unique_labels))]
        colormap.extend(extra_colors)
    
    # Plot scatter points for each class and add legend entries
    for label in unique_labels:
        # Select samples for the current class
        mask = (labels == label)
        ax.scatter(
            tsne_results[mask, 0], 
            tsne_results[mask, 1], 
            color=colormap[label], 
            label=f"CLASS{label}",  # Legend label format: CLASS0, CLASS1...
            alpha=0.7, 
            s=50 
        )
    
    # Add legend and adjust its position to avoid overlapping the plot
    ax.legend(title="Categories", loc='best', fontsize=10)
    # Add title and axis labels
    ax.set_title("t-SNE Visualization of Embeddings")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    
    # Save the image
    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval models')
    parser.add_argument('--type', default="tile",
                        help='MVTec defection dataset type to train seperated by , (default: "all": train all defect types)')

    parser.add_argument('--model_dir', default="models",
                    help=' directory contating models to evaluate (default: models)')
    
    parser.add_argument('--cuda', default=True, type=str2bool,
                    help='use cuda for model predictions (default: False)')

    parser.add_argument('--head_layer', default=2, type=int,
                    help='number of layers in the projection head (default: 8)')

    parser.add_argument('--density', default="sklearn", choices=["torch", "sklearn"],
                    help='density implementation to use. See `density.py` for both implementations. (default: torch)')

    parser.add_argument('--save_plots', default=True, type=str2bool,
                    help='save TSNE and roc plots')

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)

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
    
    device = "cuda" if args.cuda else "cpu"

    density_mapping = {
        "torch": GaussianDensityTorch,
        "sklearn": GaussianDensitySklearn
    }
    density = density_mapping[args.density]

    # Find models
    model_names = [list(Path(args.model_dir).glob(f"model-{data_type}*"))[0] for data_type in types if len(list(Path(args.model_dir).glob(f"model-{data_type}*"))) > 0]
    if len(model_names) < len(all_types):
        print("warning: not all types present in folder")

    obj = defaultdict(list)
    for model_name, data_type in zip(model_names, types):
        print(f"evaluating {data_type}")

        roc_auc = eval_model(model_name, data_type, save_plots=args.save_plots, device=device, head_layer=args.head_layer, density=density(),seed=args.seed, deterministic=args.deterministic)
        print(f"{data_type} AUC: {roc_auc}")
        obj["defect_type"].append(data_type)
        obj["roc_auc"].append(roc_auc)
    
    # Save pandas dataframe
    eval_dir = Path("eval") / args.model_dir
    eval_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(obj)
    df.to_csv(str(eval_dir) + "_perf.csv")
