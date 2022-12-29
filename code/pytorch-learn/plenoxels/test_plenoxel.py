import unittest

import torch

from volumetric_rendering_with_tv_pruning import PlenoxelModel
from volumetric_rendering_with_tv_pruning import Voxel
from volumetric_rendering_with_tv_pruning import VoxelAccess
from volumetric_rendering_with_tv_pruning import VoxelGrid
from volumetric_rendering_with_tv_pruning import modify_grad


class PlenoxelTest(unittest.TestCase):
    ALL_ONES = torch.ones(VoxelGrid.VOXEL_DIMENSION)
    ALL_ZEROES = torch.zeros(VoxelGrid.VOXEL_DIMENSION)

    def setUp(self):
        self.world = VoxelGrid.build_with_voxel(1, 1, 1, torch.ones(VoxelGrid.VOXEL_DIMENSION))

    def test_can_access_voxels_using_world_coordinates(self):
        self.assertTrue(torch.equal(PlenoxelTest.ALL_ONES, self.world.at(0, 0, 0)))

    def test_returns_empty_tensor_if_index_is_out_of_bounds(self):
        self.assertTrue(torch.equal(PlenoxelTest.ALL_ZEROES, self.world.at(0, 1, 1)))

    def test_can_scale_up_world(self):
        upscaled_world = self.world.scale_up()
        self.assertEqual((2, 2, 2), upscaled_world.voxel_grid.shape)
        self.assertTrue(torch.equal(PlenoxelTest.ALL_ONES, upscaled_world.voxel_by_position(0, 0, 0)))
        self.assertTrue(torch.equal(PlenoxelTest.ALL_ONES, upscaled_world.voxel_by_position(0, 0, 1)))
        self.assertTrue(torch.equal(PlenoxelTest.ALL_ONES, upscaled_world.voxel_by_position(0, 1, 0)))
        self.assertTrue(torch.equal(PlenoxelTest.ALL_ONES, upscaled_world.voxel_by_position(0, 1, 1)))
        self.assertTrue(torch.equal(PlenoxelTest.ALL_ONES, upscaled_world.voxel_by_position(1, 0, 0)))
        self.assertTrue(torch.equal(PlenoxelTest.ALL_ONES, upscaled_world.voxel_by_position(1, 0, 1)))
        self.assertTrue(torch.equal(PlenoxelTest.ALL_ONES, upscaled_world.voxel_by_position(1, 1, 0)))
        self.assertTrue(torch.equal(PlenoxelTest.ALL_ONES, upscaled_world.voxel_by_position(1, 1, 1)))

    def test_can_prune_voxel(self):
        world = VoxelGrid.build_with_voxel(1, 1, 1, torch.tensor([0.0001] + [1] * (VoxelGrid.VOXEL_DIMENSION - 1)))
        world.prune([0, 0, 0])
        self.check_pruned(world.voxel_by_position(0, 0, 0))

    def test_will_not_prune_voxel_if_opacity_is_above_threshold(self):
        opaque_voxel = torch.tensor([0.1] + [1] * (VoxelGrid.VOXEL_DIMENSION - 1))
        world = VoxelGrid.build_with_voxel(1, 1, 1, opaque_voxel)
        world.prune([0, 0, 0])
        self.assertTrue(torch.equal(opaque_voxel, world.voxel_by_position(0, 0, 0)))
        self.assertFalse(world.voxel_by_position(0, 0, 0).requires_grad)
        self.assertFalse(Voxel.is_pruned(world.voxel_by_position(0, 0, 0)))

    def test_can_propagate_pruned_property_through_scaling_up(self):
        world = VoxelGrid.build_with_voxel(1, 1, 1, torch.tensor([0.0001] + [1] * (VoxelGrid.VOXEL_DIMENSION - 1)))
        world.prune([0, 0, 0])
        upscaled_world = world.scale_up()
        self.assertEqual((2, 2, 2), upscaled_world.voxel_grid.shape)
        self.check_pruned(upscaled_world.voxel_by_position(0, 0, 0))
        self.check_pruned(upscaled_world.voxel_by_position(0, 0, 0))
        self.check_pruned(upscaled_world.voxel_by_position(0, 0, 1))
        self.check_pruned(upscaled_world.voxel_by_position(0, 1, 0))
        self.check_pruned(upscaled_world.voxel_by_position(0, 1, 1))
        self.check_pruned(upscaled_world.voxel_by_position(1, 0, 0))
        self.check_pruned(upscaled_world.voxel_by_position(1, 0, 1))
        self.check_pruned(upscaled_world.voxel_by_position(1, 1, 0))
        self.check_pruned(upscaled_world.voxel_by_position(1, 1, 1))

    def test_can_propagate_pruning_into_parameters(self):
        world = VoxelGrid.build_with_voxel(1, 1, 1, torch.tensor([0.0001] + [1] * (VoxelGrid.VOXEL_DIMENSION - 1)))
        world.prune([0, 0, 0])
        model = PlenoxelModel(world)
        for p in model.parameters():
            self.check_pruned(p)

    def test_modify_grad_does_not_touch_pruned_voxels(self):
        original_world = VoxelGrid.build_with_voxel(2, 2, 2,
                                                    torch.tensor([0.0001] + [1] * (VoxelGrid.VOXEL_DIMENSION - 1)))
        for _, _, _, v in original_world.all_voxels():
            Voxel.prune(v)
        model = PlenoxelModel(original_world)
        world = model.parameter_world
        view_points = [[0, 0, 0]]
        ray_sample_positions = [[1, 1, 1]]
        voxel_pointers = [[0, 8, 1]]
        all_voxels = world.voxel_grid.flatten().tolist()
        all_voxel_positions = [
            torch.tensor([0, 0, 0]),
            torch.tensor([0, 0, 1]),
            torch.tensor([0, 1, 0]),
            torch.tensor([0, 1, 1]),
            torch.tensor([1, 0, 0]),
            torch.tensor([1, 0, 1]),
            torch.tensor([1, 1, 0]),
            torch.tensor([1, 1, 1]),
        ]
        access = VoxelAccess(view_points, ray_sample_positions, voxel_pointers, all_voxels, all_voxel_positions)
        modify_grad(world, access)

        for p in model.parameters():
            self.check_pruned(p)

    def check_pruned(self, voxel):
        self.assertTrue(torch.equal(PlenoxelTest.ALL_ZEROES, voxel))
        self.assertFalse(voxel.requires_grad)
        self.assertTrue(Voxel.is_pruned(voxel))
