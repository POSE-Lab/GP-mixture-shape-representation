import gpytorch
import torch
import tyro
import common
from os.path import join as jn
import pickle
from utils.model_3d import (
    load_point_cloud,
    distance_from_centers,
    direction_distance_given_class,
    load_skeleton_points,
    xyz_from_direction_distance_class,
    write_ply,
)
from utils.metrics import Metrics
from utils.io import make_dir, log_debug
from model import ExactGPModel
import dataclasses
import numpy as np
from typing import List
import os
import tqdm
import json


@dataclasses.dataclass
class Infer_args:

    # Class name to infer
    class_name: str
    # reference point selection method
    ref_point_selection_method: str = "kmeans"
    # perform poisson meshing
    mesh: bool = False


def infer(class_name: str, test_model: str, ref_point_selection_method="kmeans", mesh: bool = False):

    points = load_point_cloud(
        jn(common.MODELS_PATH, "test", class_name, test_model.split("/")[0] + ".ply")
    )

    if ref_point_selection_method == "kmeans":
        # load the trained k-means classifier weights for the specific model
        with open(
            jn(common.RESULTS_PATH, class_name, test_model, "kmeans.pkl"), "rb"
        ) as f:
            kmeans = pickle.load(f)
        ref_points = kmeans.cluster_centers_
        class_idxs = kmeans.predict(points.astype("double"))
    elif ref_point_selection_method == "skeleton":
        skeleton_points_path = jn(
            common.MODELS_PATH, "skeletons", test_model.split("/")[0] + ".ply"
        )
        class_idxs, ref_points = load_skeleton_points(points, skeleton_points_path)

    # get the distance from each point from their correspoding reference points
    distances = distance_from_centers(points, ref_points, class_idxs)
    _, phi_thetas, ds, sorted_indices = direction_distance_given_class(
        points, distances, ref_points, class_idxs
    )

    predicted_3d_points, total_predicted, total_gt = [], [], []

    # load the trained gps
    for rfp in range(0, ref_points.shape[0]):
        log_debug(f"[bold][green] Making predictions for reference point {rfp}")

        gp_model_path = jn(
            common.RESULTS_PATH, class_name, test_model, "gps", f"gp_model_{rfp}.pth"
        )
        gp_model = ExactGPModel.load_from_file(gp_model_path)
        gp_model = gp_model.to("cuda")
        gp_model.eval()

        phi_thetas_test = torch.tensor(
            phi_thetas[rfp], dtype=torch.float32, device="cuda"
        )

        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if phi_thetas_test.shape[0] > 0:
                observed_pred = gp_model.likelihood(gp_model(phi_thetas_test))
                mean = observed_pred.mean
                lower, upper = observed_pred.confidence_region()

                # compute chamfer distance per point-cloud class
                xyz_predicted = xyz_from_direction_distance_class(
                    phi_thetas[rfp], mean.cpu(), ref_points, rfp
                )
                total_predicted.append(xyz_predicted)

            mean, lower, upper = mean.cpu(), lower.cpu(), upper.cpu()

            predicted_3d_points.append(
                xyz_from_direction_distance_class(
                    phi_thetas[rfp], mean, ref_points, rfp
                )
            )

    log_debug(["[bold][green] Done making predictions"])
    make_dir(jn(common.RESULTS_PATH, class_name, test_model, "infer"))

    # load the scale factor of the normalized model
    with open(jn(common.MODELS_PATH, "normalized", "scaling_factors.json"), "r") as f:
        scaling_factors = json.load(f)
    pred_points = np.concatenate(predicted_3d_points, axis=0)
    write_ply(
        pred_points,
        jn(common.RESULTS_PATH, class_name, test_model, "infer", "est_points.ply"),
    )
    if mesh:
        log_debug("[bold][green] Performing poisson meshing...")
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pred_points)
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(k=6)
        poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        o3d.io.write_triangle_mesh(jn(common.RESULTS_PATH, class_name, test_model, "infer", "est_mesh.ply"), poisson_mesh)
        
    log_debug("[bold][green] Exporting predicted 3D points...")

    torch.cuda.empty_cache()


if __name__ == "__main__":

    args = tyro.cli(Infer_args)
    make_dir(jn(common.RESULTS_PATH, args.class_name))
    make_dir(jn(common.MODELS_PATH, "est_models"))
    model_dirs = [
        model
        for model in os.listdir(jn(common.RESULTS_PATH, args.class_name))
        if os.path.isdir(jn(common.RESULTS_PATH, args.class_name, model))
    ]
    with tqdm.tqdm(
        total=len(model_dirs),
        desc=f"Inferring class {args.class_name}",
    ) as pbar:
        for model in model_dirs:
            for cls in os.listdir(
                jn(common.RESULTS_PATH, args.class_name, model.split(".")[0])
            ):
                try:
                    if os.path.isdir(
                        jn(
                            common.RESULTS_PATH,
                            args.class_name,
                            model.split(".")[0],
                            cls,
                        )
                    ):
                        infer(
                            args.class_name,
                            model.split(".")[0] + "/" + cls,
                            ref_point_selection_method=args.ref_point_selection_method,
                            mesh=args.mesh,
                        )
                except:
                    continue
            pbar.update(1)
