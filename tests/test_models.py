import pytest
import torch

from sidenet import DGCNN, PointNet2, SideNet, build_model


@pytest.mark.parametrize("input_mode,input_dim", [("xy", 2), ("xyz", 3), ("xyzs", 4)])
@pytest.mark.parametrize(
    "architecture", ["transformer", "dgcnn", "pointnet", "pointnet2"]
)
@pytest.mark.parametrize("num_points", [1, 2, 7, 17])
def test_all_models_support_declared_input_modes_and_small_sets(
    architecture, input_mode, input_dim, num_points
):
    model = build_model(
        {
            "architecture": architecture,
            "d_model": 32,
            "nhead": 4,
            "num_layers": 1,
            "dim_feedforward": 32,
            "dropout": 0.0,
            "dgcnn_k": 16,
            "dgcnn_dims": [16, 16],
        },
        {"input_mode": input_mode, "num_classes": 2},
    ).eval()
    output = model(torch.randn(num_points, input_dim))
    assert output.shape == (num_points, 2)


def test_pointnet2_eval_is_deterministic():
    model = PointNet2(input_dim=3, dropout=0.0).eval()
    points = torch.randn(11, 3)
    assert torch.equal(model(points), model(points))


def test_dgcnn_single_point_projects_to_hidden_dimensions():
    model = DGCNN(input_dim=3, hidden_dims=(8, 16), k=16).eval()
    assert model(torch.randn(1, 3)).shape == (1, 2)


def test_transformer_supports_non_default_model_width():
    model = SideNet(input_mode="xyz", d_model=48, nhead=4, num_layers=1).eval()
    assert model(torch.randn(5, 3)).shape == (5, 2)
