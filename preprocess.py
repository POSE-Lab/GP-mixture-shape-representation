from utils.model_3d import (
    sample_visible_surface,
    load_mesh,
    fitModel2UnitSphere,
    get_model_size,
)
from utils.io import make_dir, log_event
import tyro
import dataclasses
from typing import List
import os
import common
from os.path import join as jn
from rich.console import Console
import open3d as o3d
import shutil
import json

CONSOLE = Console()


@dataclasses.dataclass
class sample_points_args:

    # class name
    class_name: str

    # path to original meshes folder
    data_path: str = common.MODELS_PATH + "/original"

    # normalize the point cloud
    normalize: bool = True

    # Radius of sphere to normalize model (e.g. 1.0 for unit sphere)
    scale: float = 1.0

    # Number of sampling points for train and test splits
    num_samples: List[int] = dataclasses.field(default_factory=lambda: [10000, 250000])


def run(args):

    make_dir(common.MODELS_PATH + "/train")
    make_dir(common.MODELS_PATH + "/test")

    make_dir(common.MODELS_PATH + "/train/" + f"/{args.class_name}")
    make_dir(common.MODELS_PATH + "/test/" + f"/{args.class_name}")
    make_dir(common.MODELS_PATH + "/normalized" + f"/{args.class_name}")

    for model in os.listdir(jn(args.data_path, args.class_name)):
        
        # Load existing scaling factors
        scaling_factors_path = jn(
            common.MODELS_PATH, "normalized", "scaling_factors.json"
        )
        if os.path.exists(scaling_factors_path):
                with open(scaling_factors_path, "r") as f:
                    scaling_factors = json.load(f)
        else:
            scaling_factors = {}

        if args.normalize:

            # save the normalized model
            original_model = load_mesh(
                jn(common.MODELS_PATH, "original", args.class_name, model)
            )
            normalized_vetcies, scale, max_dist = fitModel2UnitSphere(
                original_model[0], scale=args.scale
            )

            

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(normalized_vetcies)
            mesh.triangles = o3d.utility.Vector3iVector(original_model[1])

            # Optional: Compute normals for better visualization
            mesh.compute_vertex_normals()

            # Step 4: Save the mesh to a file (e.g., .ply format)
            o3d.io.write_triangle_mesh(
                common.MODELS_PATH
                + "/normalized"
                + f"/{args.class_name}/{os.path.basename(model)}",
                mesh,
            )
        else:
            scale = 1.0
            max_dist = 1.0
            # just coppy the original model into the normalized models path
            shutil.copy(
                jn(args.data_path, args.class_name, model),
                jn(
                    args.data_path.replace("original", "normalized"),
                    args.class_name,
                    model,
                ),
            )

        # Update the scaling factors dictionary
        if args.class_name not in scaling_factors:
            scaling_factors[args.class_name] = {}
        scaling_factors[args.class_name][model] = {'scale': scale, 'max_dist': max_dist}

        # Save the updated scaling factors back to the file
        with open(scaling_factors_path, "w") as f:
            json.dump(scaling_factors, f, indent=4)

        log_event(f"Sampling points for model {model}")

        # sample training points
        sample_visible_surface(
            num_sample_centers=100,
            mesh_path=jn(
                args.data_path.replace("original", "normalized"),
                args.class_name,
                model,
            ),
            num_train=args.num_samples[0],
            num_test=args.num_samples[1],
            model_class=args.class_name,
            scale_fn=get_model_size,
            model_name=model,
        )

    log_event("Done sampling points.")


if __name__ == "__main__":
    args = tyro.cli(sample_points_args)
    run(args)
