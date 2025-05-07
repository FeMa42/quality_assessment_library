import os
import glob
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, List

from metrics.helpers import (
    compare_bounding_boxes,
    compare_px_area,
    compare_outline_normals,
    compare_summed_outline_normals,
)
from car_quality_estimator.car_quality_metric import load_car_quality_score


class Metrics:
    """
    Compute a configurable subset of semantic + distribution metrics.
    """

    def __init__(
        self,
        device: Optional[str] = "cuda",
        clip_model_path: str = "openai/clip-vit-large-patch14",
        clip_cache_dir: Optional[str] = None,
        compute_distribution_metrics: bool = True,
        metric_list: Optional[List[str]] = None,
        distribution_list: Optional[List[str]] = None,
    ):
        from torchmetrics.functional.image import (
            peak_signal_noise_ratio,           # PSNR
            learned_perceptual_image_patch_similarity,  # LPIPS
            structural_similarity_index_measure,  # SSIM
            spectral_distortion_index,           # D_lambda
            error_relative_global_dimensionless_synthesis,  # ERGAS
            relative_average_spectral_error,     # RASE
            root_mean_squared_error_using_sliding_window,  # RMSE_wind
            spectral_angle_mapper,               # SAM
            multiscale_structural_similarity_index_measure,  # MS-SSIM
            universal_image_quality_index,       # UQI
            visual_information_fidelity,         # VIF
            spatial_correlation_coefficient,     # SCC
        )
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.inception import InceptionScore
        from torchmetrics.image.kid import KernelInceptionDistance
        from transformers import CLIPImageProcessor, CLIPModel

        self.device = device
        self.compute_distribution_metrics = compute_distribution_metrics

        self._all_metrics = {
            "MSE": lambda inp, tgt: F.mse_loss(inp.to(torch.float32), tgt.to(torch.float32)).to(inp.dtype),
            "CLIP-S": self._make_clip_score_fn(clip_model_path, clip_cache_dir),
            "Spectral_MSE": self._make_spectral_mse(),
            "D_lambda": spectral_distortion_index,
            "ERGAS": error_relative_global_dimensionless_synthesis,
            "PSNR": peak_signal_noise_ratio,
            "RASE": relative_average_spectral_error,
            "RMSE_wind": root_mean_squared_error_using_sliding_window,
            "SAM": spectral_angle_mapper,
            "MS-SSIM": multiscale_structural_similarity_index_measure,
            "SSIM": structural_similarity_index_measure,
            "UQI": universal_image_quality_index,
            "VIF": visual_information_fidelity,
            "LPIPS": learned_perceptual_image_patch_similarity,
            "SCC": spatial_correlation_coefficient,
        }

        if metric_list is not None:
            self.metrics = {k: v for k, v in self._all_metrics.items() if k in metric_list}
        else:
            self.metrics = dict(self._all_metrics)

        self._all_distributions = {
            "FID": self._FID_score,
            "IS": self._IS_score,
            "KID": self._KID_score,
        }
        if compute_distribution_metrics:
            self.fid_metric = FrechetInceptionDistance(feature=64, normalize=True).to(device)
            self.is_metric  = InceptionScore(normalize=True).to(device)
            self.kid_metric = KernelInceptionDistance(subset_size=21, normalize=True).to(device)

            if distribution_list is not None:
                self.distribution_metrics = {
                    k: self._all_distributions[k]
                    for k in distribution_list
                    if k in self._all_distributions
                }
            else:
                self.distribution_metrics = dict(self._all_distributions)
        else:
            self.distribution_metrics = {}

        print(f"[Metrics] will compute: {list(self.metrics.keys())}")
        if self.compute_distribution_metrics:
            print(f"[Metrics] will compute distributions: {list(self.distribution_metrics.keys())}")

        self.result              = torch.zeros(len(self.metrics), device=device)
        self.result_distribution = (
            torch.zeros(len(self.distribution_metrics), device=device)
            if compute_distribution_metrics
            else None
        )
        self.total = 0

        self._clip_proc = CLIPImageProcessor.from_pretrained(
            clip_model_path,
            cache_dir=(clip_cache_dir or os.path.expanduser("~/.cache/clip"))
        )
        self._clip_model = CLIPModel.from_pretrained(
            clip_model_path,
            cache_dir=(clip_cache_dir or os.path.expanduser("~/.cache/clip"))
        ).to(device)

    def _make_clip_score_fn(self, clip_model_path, clip_cache_dir):
        def _CLIP_score(input, target):
            if input.shape[1] == 4:  
                input  = input[:, :3]
                target = target[:, :3]
            inp = self._clip_proc(input * 0.5 + 0.5, return_tensors="pt")["pixel_values"]
            tgt = self._clip_proc(target * 0.5 + 0.5, return_tensors="pt")["pixel_values"]
            with torch.no_grad():
                emb_i = self._clip_model.get_image_features(inp.to(self.device))
                emb_t = self._clip_model.get_image_features(tgt.to(self.device))
                sim   = F.cosine_similarity(emb_i, emb_t).mean().item()
            torch.cuda.empty_cache()
            return sim
        return _CLIP_score

    def _make_spectral_mse(self):
        def spectral_mse(input, target):
            dtype = input.dtype
            f1 = torch.fft.fft2(input.to(torch.float32))
            f2 = torch.fft.fft2(target.to(torch.float32))
            return ((f1.abs() - f2.abs())**2).mean().to(dtype)
        return spectral_mse

    def _FID_score(self, input=None, target=None):
        if input is None:
            return self.fid_metric.compute()
        self.fid_metric.update(target.to(self.device), real=True)
        self.fid_metric.update(input.to(self.device), real=False)
        return self.fid_metric.compute()

    def _IS_score(self, input=None, target=None):
        if input is None:
            return self.is_metric.compute()[0]
        self.is_metric.update(input.to(self.device))
        return self.is_metric.compute()[0]

    def _KID_score(self, input=None, target=None):
        if input is None:
            return self.kid_metric.compute()[0]
        self.kid_metric.update(target.to(self.device), real=True)
        self.kid_metric.update(input.to(self.device), real=False)
        return self.kid_metric.compute()[0]

    def compute_image(self, input: torch.Tensor, target: torch.Tensor):
        assert input.shape == target.shape
        input  = input.to(self.device) * 2 - 1
        target = target.to(self.device) * 2 - 1

        agg = torch.zeros(len(self.metrics), device=self.device)
        for i in range(input.shape[0]):
            vals = [fn(input[i:i+1], target[i:i+1]) for fn in self.metrics.values()]
            agg += torch.tensor(vals, device=self.device)

        if self.compute_distribution_metrics:
            dist_vals = [fn(input, target) for fn in self.distribution_metrics.values()]
            self.result_distribution += torch.tensor(dist_vals, device=self.device)

        self.result += agg / input.shape[0]
        self.total += 1

        out = dict(zip(self.metrics.keys(), (self.result / self.total).tolist()))
        if self.compute_distribution_metrics:
            out.update(dict(zip(
                self.distribution_metrics.keys(),
                (self.result_distribution / self.total).tolist()
            )))
        return out


class GeometryMetrics:
    """
    Compute a configurable subset of geometric metrics.
    """

    def __init__(self, num_points: int = 100, metric_list: Optional[List[str]] = None):
        self.num_points  = num_points
        self.metric_list = metric_list
        print(f"[GeometryMetrics] metric_list = {self.metric_list}")

        self.reset()

    def reset(self):
        self.area_diffs          = []
        self.bbox_metrics        = []
        self.outline_diffs       = []
        self.outline_sq          = []
        self.summed_diffs        = []
        self.summed_sq           = []
        self.total = 0

    def compute_image_pair(self, img1: Image.Image, img2: Image.Image):
        res = {}
        if self.metric_list is None or "Rel_Pixel_Area_Diff" in self.metric_list:
            a = compare_px_area(img1, img2)["relative_difference"]
            self.area_diffs.append(a)
            res["Rel_Pixel_Area_Diff"] = a

        if self.metric_list is None or "Rel_BB_Aspect_Ratio_Diff" in self.metric_list:
            bb = compare_bounding_boxes(img1, img2)["aspect_percent"]
            self.bbox_metrics.append(bb)
            res["Rel_BB_Aspect_Ratio_Diff"] = bb

        o = compare_outline_normals(img1, img2, num_points=self.num_points)
        if self.metric_list is None or "Outline_Normals_Angle_Diff" in self.metric_list:
            od = o["average_angle_difference_degrees"]
            self.outline_diffs.append(od)
            res["Outline_Normals_Angle_Diff"] = od
        if self.metric_list is None or "Squared_Outline_Normals_Angle_Diff" in self.metric_list:
            osq = o["average_angle_difference_squared_degrees"]
            self.outline_sq.append(osq)
            res["Squared_Outline_Normals_Angle_Diff"] = osq

        so = compare_summed_outline_normals(img1, img2, num_points=self.num_points)
        if self.metric_list is None or "Summed_Outline_Normals_Angle_Diff" in self.metric_list:
            sd = so["average_summed_angle_difference_degrees"]
            self.summed_diffs.append(sd)
            res["Summed_Outline_Normals_Angle_Diff"] = sd
        if self.metric_list is None or "Squared_Summed_Outline_Normals_Angle_Diff" in self.metric_list:
            ssq = so["average_summed_angle_difference_squared_degrees"]
            self.summed_sq.append(ssq)
            res["Squared_Summed_Outline_Normals_Angle_Diff"] = ssq

        self.total += 1
        return res

    def get_average_metrics(self):
        out = {}
        if self.metric_list is None or "Rel_Pixel_Area_Diff" in self.metric_list:
            out["Rel_Pixel_Area_Diff"] = float(np.mean(self.area_diffs)) if self.area_diffs else None
        if self.metric_list is None or "Rel_BB_Aspect_Ratio_Diff" in self.metric_list:
            out["Rel_BB_Aspect_Ratio_Diff"] = float(np.mean(self.bbox_metrics)) if self.bbox_metrics else None
        if self.metric_list is None or "Outline_Normals_Angle_Diff" in self.metric_list:
            out["Outline_Normals_Angle_Diff"] = float(np.mean(self.outline_diffs)) if self.outline_diffs else None
        if self.metric_list is None or "Squared_Outline_Normals_Angle_Diff" in self.metric_list:
            out["Squared_Outline_Normals_Angle_Diff"] = float(np.mean(self.outline_sq)) if self.outline_sq else None
        if self.metric_list is None or "Summed_Outline_Normals_Angle_Diff" in self.metric_list:
            out["Summed_Outline_Normals_Angle_Diff"] = float(np.mean(self.summed_diffs)) if self.summed_diffs else None
        if self.metric_list is None or "Squared_Summed_Outline_Normals_Angle_Diff" in self.metric_list:
            out["Squared_Summed_Outline_Normals_Angle_Diff"] = float(np.mean(self.summed_sq)) if self.summed_sq else None
        out["Image_Pairs"] = self.total
        return out


class CarQualityMetrics:
    """
    Wraps no-reference CarQualityScore and filters to requested sub-metrics.
    """

    def __init__(
        self,
        use_combined_embedding_model: bool = True,
        device: Optional[str] = None,
        batch_size: int = 32,
        metrics_list: Optional[List[str]] = None,
    ):
        self.metric = load_car_quality_score(
            device=device,
            use_combined_embedding_model=use_combined_embedding_model,
            batch_size=batch_size,
        )
        self.metrics_list = metrics_list
        print(f"[CarQualityMetrics] metrics_list = {self.metrics_list}")


    def compute_folder_metrics(self, folder: str) -> dict:
        paths = sorted(glob.glob(os.path.join(folder, "*.png")))
        if not paths:
            raise ValueError(f"No PNG images found in {folder!r}")
        imgs = [Image.open(p).convert("RGB") for p in paths]
        raw = self.metric.compute_scores_no_reference(imgs)

        cleaned = {}
        for k, v in raw.items():
            cleaned[k] = int(v) if k == "num_samples" else float(np.array(v))

        if self.metrics_list is not None:
            cleaned = {k: cleaned[k] for k in cleaned if k in self.metrics_list}
        return cleaned

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore

class GlobalImagePairDataset(Dataset):
    def __init__(self, gt_root, gen_root, transform=None):
        super().__init__()
        self.transform = transform or transforms.ToTensor()
        self.pairs = []
        for obj in sorted(os.listdir(gt_root)):
            gt_dir  = os.path.join(gt_root, obj)
            gen_dir = os.path.join(gen_root, obj)
            if not os.path.isdir(gt_dir) or not os.path.isdir(gen_dir):
                continue
            for fn in sorted(os.listdir(gt_dir)):
                if fn.lower().endswith(".png"):
                    gt_path  = os.path.join(gt_dir,  fn)
                    gen_path = os.path.join(gen_dir, fn)
                    if os.path.exists(gen_path):
                        self.pairs.append((gt_path, gen_path))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        gt_path, gen_path = self.pairs[idx]
        gt = Image.open(gt_path).convert("RGB")
        gn = Image.open(gen_path).convert("RGB")
        return self.transform(gt), self.transform(gn)


def compute_global_distribution_metrics(
    gt_root: str,
    gen_root: str,
    distribution_list: Optional[List[str]] = None,
    device: str = "cuda",
    batch_size: int = 128,
    num_workers: int = 4,
    compute_on_cpu: bool = False,
) -> dict:
    """
    Batched FID/IS/KID over ALL matching pairs under gt_root/gen_root.
    """
    ds = GlobalImagePairDataset(gt_root, gen_root, transform=transforms.ToTensor())
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(not compute_on_cpu),
    )

    dev_metrics = "cpu" if compute_on_cpu else device

    objs = {}
    if distribution_list is None or "FID" in distribution_list:
        objs["FID"] = FrechetInceptionDistance(
            feature=64, normalize=True, compute_on_cpu=compute_on_cpu
        ).to(dev_metrics)
    if distribution_list is None or "KID" in distribution_list:
        objs["KID"] = KernelInceptionDistance(
            subset_size=21, normalize=True, compute_on_cpu=compute_on_cpu
        ).to(dev_metrics)
    if distribution_list is None or "IS" in distribution_list:
        objs["IS"] = InceptionScore(
            normalize=True, compute_on_cpu=compute_on_cpu
        ).to(dev_metrics)

    for gt_batch, gen_batch in loader:
        if not compute_on_cpu:
            gt_batch  = gt_batch.to(device)
            gen_batch = gen_batch.to(device)
        for name, m in objs.items():
            if name == "FID":
                m.update(gt_batch, real=True)
                m.update(gen_batch, real=False)
            elif name == "KID":
                m.update(gt_batch, real=True)
                m.update(gen_batch, real=False)
            else:  # IS
                m.update(gen_batch)

    out = {}
    if "FID" in objs:
        out["FID"] = objs["FID"].compute().item()
    if "KID" in objs:
        out["KID"] = objs["KID"].compute()[0].item()
    if "IS" in objs:
        out["IS"] = objs["IS"].compute()[0].item()

    return out

