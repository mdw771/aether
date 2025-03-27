import argparse
import pytest

import torch
import matplotlib.pyplot as plt
import firefly.api as api

import test_utils as tutils


class TestGuidedLatentFlowMatching(tutils.TungstenDataTester):
    
    @pytest.mark.local()
    def test_guided_latent_flow_matching(self):
        self.setup_ptychi(cpu_only=False)

        data, probe, pixel_size_m, positions_px = self.load_tungsten_data(pos_type='true')
        options = api.GuidedDiffusionOptions()

        options.data_options.data = data

        options.object_options.initial_guess = torch.ones([1, 1024, 1024], dtype=torch.complex64)
        options.object_options.pixel_size_m = pixel_size_m

        options.probe_options.initial_guess = probe

        options.probe_position_options.position_x_px = positions_px[:, 1]
        options.probe_position_options.position_y_px = positions_px[:, 0]

        options.reconstructor_options.model_path = "stabilityai/stable-diffusion-3.5-medium"
        options.reconstructor_options.num_inference_steps = 40
        options.reconstructor_options.text_guidance_scale = 15
        options.reconstructor_options.physical_guidance_scale = 1e-1
        options.reconstructor_options.physical_guidance_method = api.enums.PhysicalGuidanceMethods.SCORE
        options.reconstructor_options.time_travel_plan.stride = torch.inf
        options.reconstructor_options.prompt = "a binary pattern of intertwined lines"

        task = api.GuidedDiffusionPtychographyTask(options)
        task.run()

        plt.figure()
        plt.imshow(task.reconstructor.parameter_group.object.data.angle().detach().cpu().numpy()[0])
        plt.show()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    tester = TestGuidedLatentFlowMatching()
    tester.setup_method(name="", generate_data=False, generate_gold=args.generate_gold, debug=True)
    tester.test_guided_latent_flow_matching()
