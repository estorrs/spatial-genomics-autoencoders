import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from scipy.spatial import ConvexHull
from skimage.draw import polygon2mask
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ColorJitter, RandomCrop, RandomRotation, CenterCrop, Compose, Normalize
from torchvision.transforms import InterpolationMode

from spatial_genomics_autoencoders.data.utils import flexible_rescale, create_circular_mask


def adata_from_visium(fp, min_genes=200, min_cells=3,
                      filter_mt=True, only_highly_variable=True,
                      include_genes=None):
    if isinstance(fp, str):
        if fp.split('.')[-1] == 'h5ad':
            a = sc.read_h5ad(fp)
        else:
            a = sc.read_visium(fp)
        a.var_names_make_unique()

    if min_genes is not None:
        sc.pp.filter_cells(a, min_genes=min_genes)
    if min_cells is not None:
        sc.pp.filter_genes(a, min_cells=min_cells)

    if filter_mt:
        a = a[:, [False if x[:3]=='MT-' else True
                  for x in a.var.index]]
    
    keep = []
    if only_highly_variable:
        norm = a.copy()
        sc.pp.normalize_total(norm, target_sum=1e4)
        sc.pp.log1p(norm)
        sc.pp.highly_variable_genes(norm)
        # norm = norm[:, norm.var.highly_variable]
        # norm = sc.pp.scale(norm)
        keep += [g for x, g in zip(norm.var.highly_variable, norm.var.index) if x]
        if include_genes is not None:
            keep += include_genes
    elif include_genes is not None:
        keep += include_genes
    else:
        keep = a.var.index.to_list()

    keep = sorted(set(keep))
    a = a[:, keep]

    a.obsm['spatial'] = a.obsm['spatial'].astype(int)

    return a


def project_expression(labeled, exp, voxel_idxs):
    """Project spot expression onto a labeled image."""
    new = torch.zeros((labeled.shape[-2], labeled.shape[-1], exp.shape[1]), dtype=exp.dtype)
    for i, idx in enumerate(voxel_idxs):
        new[labeled.squeeze()==idx] = exp[i]
    return new


def incorporate_hi_res(adata, he, scale=.05, is_fullres=True, trim=True):
    """Incorporate full-res H&E into anndata object."""
    spot_diameter_fullres = next(iter(
        adata.uns['spatial'].values()))['scalefactors']['spot_diameter_fullres']
    tissue_hires_scalef = next(iter(
        adata.uns['spatial'].values()))['scalefactors']['tissue_hires_scalef']

    if is_fullres:
        spot_diameter = spot_diameter_fullres
        spatial_obsm = adata.obsm['spatial']
    else:
        spot_diameter = spot_diameter_fullres * tissue_hires_scalef
        spatial_obsm = (adata.obsm['spatial'] * tissue_hires_scalef).astype(int)
    spot_diameter, spot_radius = int(spot_diameter), int(spot_diameter / 2)

    c_min, r_min = np.min(spatial_obsm, axis=0) - spot_radius
    c_max, r_max = np.max(spatial_obsm, axis=0) + spot_radius

    if trim:
        adata.uns['trimmed'] = he[r_min:r_max, c_min:c_max]
        adata.obsm['spatial_trimmed'] = spatial_obsm + np.asarray([-c_min, -r_min])
    else:
        adata.uns['trimmed'] = he
        adata.obsm['spatial_trimmed'] = spatial_obsm

    adata.uns[f'trimmed_{scale}'] = rearrange(
        flexible_rescale(rearrange(adata.uns['trimmed'], 'h w c -> c h w'), scale=scale),
        'c h w -> h w c')
    adata.obsm[f'spatial_trimmed_{scale}'] = (adata.obsm['spatial_trimmed'] * scale).astype(int)

    sr = int(scale * spot_radius)

    labeled_img = np.zeros(
        (adata.uns[f'trimmed_{scale}'].shape[0], adata.uns[f'trimmed_{scale}'].shape[1]),
        dtype=np.uint32)
    footprint = create_circular_mask(sr * 2, sr * 2)

    for i, (c, r) in enumerate(adata.obsm[f'spatial_trimmed_{scale}']):
        r, c = int(r), int(c)
        rect = np.zeros((sr * 2, sr * 2))
        # print(rect.shape, r, c)
        rect[footprint>0] = i + 1
        r_size, c_size = min(labeled_img.shape[0], r+sr) - max(0, r-sr), min(labeled_img.shape[1], c+sr) - max(0, c-sr)
        labeled_img[max(0, r-sr):max(0, r-sr) + r_size, max(0, c-sr):max(0, c-sr) + c_size] = rect[:r_size, :c_size]
    adata.uns[f'trimmed_{scale}_labeled_img'] = labeled_img

    return adata


def create_masks(labeled_mask, voxel_idxs, max_area, thresh=.25):
    voxel_idxs = torch.unique(labeled_mask)[1:].to(torch.long)
    lm = labeled_mask.squeeze()
    masks = torch.zeros((len(voxel_idxs), lm.shape[-2], lm.shape[-1]), dtype=torch.bool)
    for i, l in enumerate(voxel_idxs):
        m = masks[i]
        m[lm==l] = 1

    keep = masks.sum(dim=(-1,-2)) / max_area > thresh

    if keep.sum() > 0:
        masks = masks[keep]
        voxel_idxs = voxel_idxs[keep]
        return masks, voxel_idxs

    return None, None


def get_valid_coords(xy, shape):
    hull = ConvexHull(xy)
    hull_xy = xy[hull.vertices]
    mask = polygon2mask(shape, hull_xy)
    xs, ys = np.where(m)
    coords = np.vstack((xs, ys)).transpose()
    return coords


class OverlaidHETransform(object):
    def __init__(self, p=.95, size=(256,256), degrees=180,
                 brightness=.1, contrast=.1, saturation=.1, hue=.1,
                 normalize=True, means=(0.771, 0.651, 0.752), stds=(0.229, 0.288, 0.224)):
        self.size = size
        self.color_transform = ColorJitter(brightness=brightness, contrast=contrast,
                                           saturation=saturation, hue=hue)
        self.spatial_transform = Compose([
            RandomCrop((size[0] * 2, size[1] * 2), padding=size, padding_mode='reflect'),
            RandomRotation(degrees),
            CenterCrop(size)
        ])
        
        if normalize:
            self.normalize = Normalize(means, stds) # from HT397B1-H2 ffpe H&E image
        else:
            self.normalize = nn.Identity()
 
        self.p = p
        
    def __call__(self, he, overlay):
        """
        he - (3, H, W)
        overlay - (n, H, W)
        """
        # idx = torch.randint(0, coords.shape[0], (1,)).item()
        # top, left = coords[idx]

        x = torch.concat((he, overlay))
        # x = TF.crop(x, top, left, self.size[0] * 2, self.size[1] * 2)
        x = self.spatial_transform(x)
        he, overlay = x[:3], x[3:]
        if torch.rand(size=(1,)) < self.p:
            he = self.color_transform(he)
            
        he = self.normalize(he)
        
        return he, overlay


class NormalizeHETransform(object):
    def __init__(self, normalize=True,
                 means=(0.771, 0.651, 0.752), stds=(0.229, 0.288, 0.224)):
        if normalize:
            self.normalize = Normalize(means, stds) # from HT397B1-H2 ffpe H&E image
        else:
            self.normalize = nn.Identity()
  
    def __call__(self, he):
        """
        he - (3, H, W)
        """
        return self.normalize(he)


class STDataset(Dataset):
    """Registration Dataset"""
    def __init__(self, adata, he, size=(256, 256), transform=OverlaidHETransform(),
                 he_scale=.5, exp_size=(16, 16), length=None, is_fullres=True):
        self.he_scale = he_scale
        self.exp_size = exp_size
        self.size = size
        self.is_fullres = is_fullres

        self.adata = incorporate_hi_res(adata, he, scale=self.he_scale, is_fullres=is_fullres, trim=False)
        self.genes = self.adata.var.index.to_list()
        self.length = self.adata.shape[0] if length is None else length

        # images
        he = self.adata.uns[f'trimmed_{self.he_scale}']
        labeled_img = self.adata.uns[f'trimmed_{self.he_scale}_labeled_img']

        he = rearrange(torch.tensor(he, dtype=torch.float32), 'h w c -> c h w')
        he -= he.min()
        he /= he.max()

        self.he = he
        self.labeled_img = torch.tensor(labeled_img.astype(np.int32)).unsqueeze(dim=0)
        self.max_voxel_area = int(
            self.labeled_img[self.labeled_img==1].sum() / (self.size[0] / self.exp_size[0])**2)
        
        # self.valid_coords = get_valid_coords(
        #     self.adata.obsm['spatial_trimmed_{self.he_scale}'][:, [1, 0]],
        #     (he.shape[-2], he.shape[-1]))[:, [1, 0]]

        self.transform = transform

        # expression
        if 'sparse' in str(type(self.adata.X)):
            self.X = torch.tensor(self.adata.X.toarray().astype(np.int32))
        else:
            self.X = torch.tensor(self.adata.X.astype(np.int32))

        self.max_voxels_per_sample = self.__determine_max_voxels()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            # images
            context_he, context_labeled_img = self.transform(self.he, self.labeled_img)
            context_labeled_img = context_labeled_img.to(torch.int32)

            he = TF.center_crop(context_he, self.size)
            labeled_img = TF.center_crop(context_labeled_img, self.size)

            context_he = TF.resize(context_he, self.size, antialias=True)
            context_labeled_img = TF.resize(
                context_labeled_img, self.size, interpolation=InterpolationMode.NEAREST_EXACT)

            voxel_idxs = torch.unique(labeled_img).to(torch.long)
            voxel_idxs = voxel_idxs[voxel_idxs!=0]
            n_voxels = len(voxel_idxs)

            # downsampling labeled img
            labeled_img = TF.resize(
                labeled_img, self.exp_size, interpolation=InterpolationMode.NEAREST_EXACT)
            masks, voxel_idxs = create_masks(labeled_img, voxel_idxs, self.max_voxel_area, thresh=.25)

            if voxel_idxs is not None:
                break

        voxel_idxs -= 1
        X = self.X[voxel_idxs]
        voxel_idxs += 1

        padding = self.max_voxels_per_sample - len(voxel_idxs)
        if padding < 0:
            raise RuntimeError(f'more voxels than max voxel size: {len(voxel_idxs)}\
, {self.max_voxels_per_sample} . increase max voxel size to nearest power of 2.')
        voxel_idxs = F.pad(voxel_idxs, (0, padding))
        X = torch.concat((X, torch.zeros((padding, X.shape[1]))))
        masks = torch.concat(
            (masks, torch.zeros((padding, masks.shape[-2], masks.shape[-1]), dtype=torch.bool)))

        return {
            'he': he,
            'labeled_img': labeled_img,
            'context_he': context_he,
            'context_labeled_img': context_labeled_img,
            'voxel_idxs': voxel_idxs,
            'masks': masks,
            'exp': X,
            'n_voxels': torch.tensor([n_voxels])
        }
    
    def __determine_max_voxels(self, n=20):
        pool = []
        for i in range(n):
            context_he, context_labeled_img = self.transform(self.he, self.labeled_img)
            context_labeled_img = context_labeled_img.to(torch.int32)

            he = TF.center_crop(context_he, self.size)
            labeled_img = TF.center_crop(context_labeled_img, self.size)

            context_he = TF.resize(context_he, self.size, antialias=True)
            context_labeled_img = TF.resize(
                context_labeled_img, self.size, interpolation=InterpolationMode.NEAREST_EXACT)

            voxel_idxs = torch.unique(labeled_img).to(torch.long)
            voxel_idxs = voxel_idxs[voxel_idxs!=0]
            n_voxels = len(voxel_idxs)
            pool.append(n_voxels)
        max_voxels = np.max(pool)
        return 2**int(np.sqrt(max_voxels) + 2)

    def sanity_check(self, gene='IL7R'):
        d = self[0]
        print(f'keys: {d.keys()}')

        img = rearrange(d['he'].clone().detach().cpu().numpy(), 'c h w -> h w c')
        img -= img.min()
        img /= img.max()
        plt.imshow(img)
        plt.title('H&E')
        plt.show()

        plt.imshow(d['labeled_img'][0])
        plt.title('labeled image')
        plt.show()

        img = rearrange(d['context_he'].clone().detach().cpu().numpy(), 'c h w -> h w c')
        img -= img.min()
        img /= img.max()
        plt.imshow(img)
        plt.title('context H&E')
        plt.show()

        plt.imshow(d['context_labeled_img'][0])
        plt.title('context labeled image')
        plt.show()

        print('voxel idxs: ' + str(d['voxel_idxs']))
        print('labeled image voxel idxs: ' + str(torch.unique(d['labeled_img'])))
        print('context labeled image voxel idxs: ', str(torch.unique(d['context_labeled_img'])))

        print(f'masks shape: ' + str(d['masks'].shape))
        plt.imshow(d['masks'][0])
        plt.title('first voxel')
        plt.show()
        plt.imshow(d['masks'].sum(0))
        plt.title('summed masks')
        plt.show()

        print('expression counts shape: ' + str(d['exp'].shape))
        print(d['exp'])

        if gene in self.genes:
            idx =  self.genes.index(gene)
            x = project_expression(d['labeled_img'], d['exp'][:, idx:idx + 1], d['voxel_idxs'])
            plt.imshow(x[..., 0])
            plt.title(gene)
            plt.show()


class MultisampleSTDataset(Dataset):
    def __init__(self, ds_dict):
        super().__init__()
        self.samples = list(ds_dict.keys())
        self.ds_dict = ds_dict

        self.mapping = [(k, i) for k, ds in ds_dict.items() for i in zip(range(len(ds)))]
        self.ds_to_max_voxels = {k:len(ds[0]['voxel_idxs']) for k, ds in ds_dict.items()}
        self.max_voxels = np.max(list(self.ds_to_max_voxels.values()))
        self.ds_to_padding = {k:self.max_voxels - v for k, v in self.ds_to_max_voxels.items()}

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        k, i = self.mapping[idx]

        d = self.ds_dict[k][i]

        for key in ['voxel_idxs']:
            d[key] = F.pad(d[key], (0, self.ds_to_padding[k]))

        for key in ['exp']:
            d[key] = F.pad(d[key], (0, 0, 0, self.ds_to_padding[k]))

        for key in ['masks']:
            d[key] = F.pad(d[key], (0, 0, 0, 0, 0, self.ds_to_padding[k]))

        return d
    

class HEPredictionDataset(Dataset):
    """Tiled H&E dataset"""
    def __init__(self, he, size=256, context_res=2., transform=OverlaidHETransform(),
                 scale=.5, overlap_pct=.5, exp_size=16):
        self.scale = scale
        self.size = size
        self.exp_size = exp_size
        self.overlap_pct = overlap_pct
        self.step_size = int(size * overlap_pct)
        self.tile_width = int(size * context_res)
        self.transform = transform

        he = rearrange(torch.tensor(he, dtype=torch.float32), 'h w c -> c h w')
        self.full_res_he_shape = he.shape
        he -= he.min()
        he /= he.max()
        he = TF.resize(he, (int(he.shape[-2] * scale), int(he.shape[-1] * scale)))

        self.scaled_he_shape = he.shape
        he = self.transform(he)
        self.padding = int((context_res * size / 2) - (size / 2) + (overlap_pct * size / 2))

        he = TF.pad(he, self.padding, padding_mode='reflect')
        self.he = he

        n_steps_width = self.he.shape[-1] // self.step_size
        n_steps_height = self.he.shape[-2] // self.step_size # assuming square tile
        self.coord_to_tile = {}
        self.coords = []
        self.absolute_coords = []
        for i in range(n_steps_height):
            for j in range(n_steps_width):
                r, c = i * self.step_size, j * self.step_size
                tile = TF.crop(self.he, r, c, self.tile_width, self.tile_width)
                self.coord_to_tile[(i, j)] = tile
                self.coords.append((i, j))
                self.absolute_coords.append((r, c))

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        i, j = self.coords[idx]
        r, c = self.absolute_coords[idx]
        context_he = self.coord_to_tile[(i, j)]
        he = TF.center_crop(context_he, (self.size, self.size))
        context_he = TF.resize(context_he, (self.size, self.size))

        return {
            'coord': torch.tensor([i, j], dtype=torch.long),
            'absolute': torch.tensor([r, c], dtype=torch.long),
            'he': he,
            'context_he': context_he,
        }
    
    def retile(self, coord_to_tile, trim_to_original=True, scale_to_original=False):
        tile = next(iter(coord_to_tile.values()))
        max_i = max([i for i, _ in coord_to_tile.keys()])
        max_j = max([j for _, j in coord_to_tile.keys()])
        new = None
        for i in range(max_i):
            row = None
            for j in range(max_j):
                tile = coord_to_tile[i, j]
                tile = TF.center_crop(
                    tile,
                    (int(self.exp_size * self.overlap_pct), int(self.exp_size * self.overlap_pct)))
                if row is None:
                    row = tile
                else:
                    row = torch.concat((row, tile), dim=-1)
            if new is None:
                new = row
            else:
                new = torch.concat((new, row), dim=-2)
        if trim_to_original:
            ratio = self.exp_size / self.size
            h = int(self.scaled_he_shape[-2] * ratio)
            w = int(self.scaled_he_shape[-1] * ratio)
            new = new[..., :h, :w]
        if scale_to_original:
            new = TF.resize(new, (self.full_res_he_shape[-2], self.full_res_he_shape[-1]))

        return new


    def sanity_check(self):
        print(f'num tiles: {len(self.coords)}, step size: {self.step_size}, tile width: {self.tile_width}')

        d = self[0]
        print(f'keys: {d.keys()}')

        img = rearrange(d['he'].clone().detach().cpu().numpy(), 'c h w -> h w c')
        img -= img.min()
        img /= img.max()
        plt.imshow(img)
        plt.title('H&E')
        plt.show()

        img = rearrange(d['context_he'].clone().detach().cpu().numpy(), 'c h w -> h w c')
        img -= img.min()
        img /= img.max()
        plt.imshow(img)
        plt.title('context H&E')
        plt.show()

        img = rearrange(self.retile(self.coord_to_tile), 'c h w -> h w c')
        img -= img.min()
        img /= img.max()
        plt.imshow(img)
        plt.title('retiled to full res H&E')
        plt.show()

        print(f'original H&E shape: {self.full_res_he_shape}')
        print(f'retiled H&E shape: {img.shape}')