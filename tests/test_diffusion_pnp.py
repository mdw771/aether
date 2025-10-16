import argparse

import torch
import matplotlib.pyplot as plt
import pytest
import ptychi.api as pcapi

import aether.api as api

import test_utils as tutils


@pytest.fixture(autouse=True)
def mock_diffusion_model(monkeypatch):
    import aether.api.task as api_task
    import aether.reconstructors.pnp as pnp

    class DummyUnet:
        def __init__(self):
            self.dtype = torch.float32

    class DummyPipe:
        def __init__(self):
            self.device = "cpu"
            self.unet = DummyUnet()

        def to(self, _):
            # Always keep the dummy pipe on CPU to avoid CUDA dependency in tests.
            self.device = "cpu"
            return self

    class DummyModelLoader:
        def __init__(self, *args, **kwargs):
            self.pipe = DummyPipe()

        def load(self):
            return None

    def dummy_project_to_prior(self):
        self.v = self.x.detach().clone()
        self.num_prior_projections += 1

    monkeypatch.setattr(api_task.fio, "HuggingFaceModelLoader", DummyModelLoader)
    monkeypatch.setattr(pnp, "HuggingFaceModelLoader", DummyModelLoader)
    monkeypatch.setattr(pnp.PnPLEDITSPPReconstructor, "project_to_prior", dummy_project_to_prior)


class TestDiffusionPnP(tutils.TungstenDataTester):
    
    @staticmethod
    def create_ptychi_options_lsqml(patterns, pixel_size_m, probe, pos_x, pos_y):
        options = pcapi.LSQMLOptions()

        options.data_options.data = patterns

        options.object_options.initial_guess = torch.ones([1, 1024, 1024], dtype=torch.complex64)
        options.object_options.pixel_size_m = pixel_size_m
        options.object_options.remove_object_probe_ambiguity.enabled = False

        options.probe_options.initial_guess = probe
        options.probe_options.optimizable = True

        options.probe_position_options.optimizable = True
        options.probe_position_options.position_x_px = pos_x
        options.probe_position_options.position_y_px = pos_y

        options.reconstructor_options.batch_size = 100
        options.reconstructor_options.num_epochs = 5

        return options
    
    def test_diffusion_pnp(self):
        self.atol = 1e-1
        self.rtol = 0
        self.trigger_on_mean_abs_diff = True
        
        self.setup_ptychi(cpu_only=False)

        data, probe, pixel_size_m, positions_px = self.load_tungsten_data(pos_type='true')
        
        options = api.PnPOptions()

        options.object_options.initial_guess = torch.ones([1, 1024, 1024], dtype=torch.complex64)
        options.object_options.pixel_size_m = pixel_size_m

        options.reconstructor_options.data_projection_options = self.create_ptychi_options_lsqml(
            data, pixel_size_m, probe, positions_px[:, 1], positions_px[:, 0]
        )
        
        options.reconstructor_options.prior_projection_options = api.LEDITSPPOptions()
        options.reconstructor_options.prior_projection_options.num_inference_steps = 100
        options.reconstructor_options.prior_projection_options.editing_prompt = "dot grid"
        options.reconstructor_options.prior_projection_options.remove_concept = True
        options.reconstructor_options.prior_projection_options.text_guidance_scale = 7
        options.reconstructor_options.prior_projection_options.unwrap_phase_before_editing = True
        options.reconstructor_options.prior_projection_options.stats_matching_threshold = 0.01
        options.reconstructor_options.prior_projection_options.model_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        options.reconstructor_options.prior_projection_options.match_stats_of_prior_projected_image = True
        options.reconstructor_options.prior_projection_options.generator_seed = 123
        options.reconstructor_options.num_data_projection_epochs = 5
        options.reconstructor_options.num_epochs = 2
        options.reconstructor_options.use_prior_projected_data_as_final_result = False
        options.reconstructor_options.proximal_penalty = 1e-4
        options.reconstructor_options.update_relaxation = 0.9
        options.reconstructor_options.batch_size = 100
        task = api.PnPPtychographyTask(options)
        task.run()
        
        return
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    tester = TestDiffusionPnP()
    tester.setup_method(name="", generate_data=False, generate_gold=args.generate_gold, debug=True)
    tester.test_diffusion_pnp()
